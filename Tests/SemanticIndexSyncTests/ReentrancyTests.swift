import XCTest
@testable import SemanticIndexSync

/// `IndexCoordinator.drainMigration` awaits the embedding provider, and the actor
/// is reentrant across that `await`. These tests mutate the index *while a drain
/// is genuinely parked inside the provider call* — not before it and not after —
/// and assert nothing is written back on the strength of the pre-await snapshot.
///
/// The rendezvous is a continuation, not a sleep, so the interleaving is the one
/// the test asked for on every run rather than most runs on an idle machine.
final class ReentrancyTests: XCTestCase {

    private let device = DeviceID("phone")
    private let epoch = EmbeddingEpoch(modelIdentifier: "system.text", revision: 1, dimension: 32)

    func testDocumentDeletedMidEmbedDoesNotGetAVectorWrittenBack() async {
        let gate = Gate()
        let coordinator = IndexCoordinator(
            device: device,
            provider: GatedProvider(epoch: epoch, gate: gate)
        )
        await coordinator.upsert(
            document: DocumentID("doomed"),
            passages: ["This passage is about to be deleted mid-flight."],
            hash: ContentHash("v1")
        )

        let drain = Task { await coordinator.drainMigration(budget: .background, conditions: DeviceConditions()) }

        // Wait until the provider is genuinely suspended inside `embed`.
        await gate.waitUntilEntered()
        await coordinator.delete(document: DocumentID("doomed"))
        await gate.release()

        _ = await drain.value

        let violations = await coordinator.invariantViolations()
        XCTAssertTrue(violations.isEmpty, "Violations: \(violations)")
        let counts = await coordinator.vectorCountsByEpoch
        XCTAssertTrue(
            counts.isEmpty,
            "A vector was written for a document deleted across the suspension point: \(counts)"
        )
        let result = await coordinator.search("deleted mid-flight", limit: 5)
        XCTAssertTrue(result.hits.isEmpty)
    }

    func testDocumentEditedMidEmbedKeepsTheChunkQueuedInsteadOfStoringStaleVector() async {
        let gate = Gate()
        let coordinator = IndexCoordinator(
            device: device,
            provider: GatedProvider(epoch: epoch, gate: gate)
        )
        await coordinator.upsert(
            document: DocumentID("edited"),
            passages: ["Original text."],
            hash: ContentHash("v1")
        )

        let drain = Task { await coordinator.drainMigration(budget: .background, conditions: DeviceConditions()) }

        await gate.waitUntilEntered()
        await coordinator.upsert(
            document: DocumentID("edited"),
            passages: ["Replacement text that was never embedded."],
            hash: ContentHash("v2")
        )
        await gate.release()

        _ = await drain.value

        // The vector produced describes "Original text.", which is gone. It must
        // not be stored against the new content hash.
        let counts = await coordinator.vectorCountsByEpoch
        XCTAssertTrue(counts.isEmpty, "A stale vector was stored against edited content: \(counts)")
        let remainingCheck = await coordinator.migrationProgress.remaining
        XCTAssertEqual(remainingCheck, 1)
        let violationsCheck = await coordinator.invariantViolations()
        XCTAssertTrue(violationsCheck.isEmpty, "Violations: \(violationsCheck)")

        let result = await coordinator.search("replacement", limit: 5)
        XCTAssertEqual(result.completeness.semanticallyCovered, 0)
        XCTAssertEqual(result.completeness.awaitingReindex, 1)
    }

    func testEpochBumpMidEmbedDiscardsVectorsForTheAbandonedSpace() async {
        let gate = Gate()
        let coordinator = IndexCoordinator(
            device: device,
            provider: GatedProvider(epoch: epoch, gate: gate)
        )
        await coordinator.upsert(
            document: DocumentID("d"),
            passages: ["A passage caught between two embedding spaces."],
            hash: ContentHash("v1")
        )

        let drain = Task { await coordinator.drainMigration(budget: .background, conditions: DeviceConditions()) }

        await gate.waitUntilEntered()
        let bumped = EmbeddingEpoch(modelIdentifier: "system.text", revision: 2, dimension: 32)
        await coordinator.adopt(provider: DeterministicEmbeddingProvider(epoch: bumped))
        await gate.release()

        _ = await drain.value

        let counts = await coordinator.vectorCountsByEpoch
        XCTAssertNil(counts[epoch], "A vector for the abandoned epoch was committed: \(counts)")
        let epochAfterBump = await coordinator.currentEpoch
        XCTAssertEqual(epochAfterBump, bumped)
        let remainingCheck = await coordinator.migrationProgress.remaining
        XCTAssertEqual(remainingCheck, 1)
        let violationsCheck = await coordinator.invariantViolations()
        XCTAssertTrue(violationsCheck.isEmpty, "Violations: \(violationsCheck)")
    }
}

/// Many writers hitting one coordinator at once, then a full invariant audit.
///
/// A concurrency test with no concurrent writer proves nothing, so every task
/// here genuinely mutates shared state: upserts, deletes, remote merges and
/// drains all race against each other.
final class ConcurrentWriterTests: XCTestCase {

    private let epoch = EmbeddingEpoch(modelIdentifier: "system.text", revision: 1, dimension: 32)

    func testInvariantsHoldUnderConcurrentWritersAndDrains() async {
        let coordinator = IndexCoordinator(
            device: DeviceID("phone"),
            provider: DeterministicEmbeddingProvider(epoch: epoch)
        )
        let documentCount = 24

        await withTaskGroup(of: Void.self) { group in
            for index in 0..<documentCount {
                group.addTask {
                    let id = DocumentID("doc-\(index)")
                    await coordinator.upsert(
                        document: id,
                        passages: ["Passage \(index) about queues, budgets and epochs.", "Second passage \(index)."],
                        hash: ContentHash("doc-\(index)-v1")
                    )
                }
            }
            // Drains racing the writers.
            for _ in 0..<8 {
                group.addTask {
                    _ = await coordinator.drainMigration(budget: .background, conditions: DeviceConditions())
                }
            }
            // Deletes racing both.
            for index in stride(from: 0, to: documentCount, by: 3) {
                group.addTask {
                    await coordinator.delete(document: DocumentID("doc-\(index)"))
                }
            }
            // Remote merges racing everything.
            for index in stride(from: 1, to: documentCount, by: 5) {
                group.addTask {
                    let record = DocumentRecord.live(
                        DocumentID("doc-\(index)"),
                        hash: ContentHash("doc-\(index)-remote"),
                        by: DeviceID("laptop")
                    )
                    _ = await coordinator.applyRemote(
                        records: [record],
                        passages: [DocumentID("doc-\(index)"): ["Remote passage \(index)."]]
                    )
                }
            }
            await group.waitForAll()
        }

        let violations = await coordinator.invariantViolations()
        XCTAssertTrue(violations.isEmpty, "Invariant violations after concurrent load: \(violations)")

        // Drain whatever is left and confirm the index is coherent afterwards.
        for _ in 0..<64 {
            let outcome = await coordinator.drainMigration(budget: .foregroundInteractive, conditions: DeviceConditions())
            if case .progressed = outcome { continue }
            break
        }

        let result = await coordinator.search("queues budgets epochs", limit: 10)
        XCTAssertTrue(result.completeness.isComplete, "Summary: \(result.completeness.summary)")
        XCTAssertEqual(result.completeness.awaitingReindex, 0)
        let violationsCheck = await coordinator.invariantViolations()
        XCTAssertTrue(violationsCheck.isEmpty, "Violations: \(violationsCheck)")
        XCTAssertFalse(result.hits.isEmpty)
    }

    /// Two replicas exchanging manifests concurrently must end up agreeing.
    func testTwoReplicasConvergeAfterConcurrentExchange() async {
        let left = IndexCoordinator(device: DeviceID("phone"), provider: DeterministicEmbeddingProvider(epoch: epoch))
        let right = IndexCoordinator(device: DeviceID("laptop"), provider: DeterministicEmbeddingProvider(epoch: epoch))

        for index in 0..<6 {
            await left.upsert(document: DocumentID("shared-\(index)"), passages: ["left \(index)"], hash: ContentHash("v-\(index)"))
            await right.upsert(document: DocumentID("shared-\(index)"), passages: ["right \(index)"], hash: ContentHash("w-\(index)"))
        }

        let leftManifest = await left.exportManifest()
        let rightManifest = await right.exportManifest()

        async let leftMerge: Void = { _ = await left.applyRemote(records: rightManifest) }()
        async let rightMerge: Void = { _ = await right.applyRemote(records: leftManifest) }()
        _ = await (leftMerge, rightMerge)

        let leftFinal = await left.exportManifest()
        let rightFinal = await right.exportManifest()

        XCTAssertEqual(leftFinal.count, rightFinal.count)
        for (a, b) in zip(leftFinal, rightFinal) {
            XCTAssertEqual(a.id, b.id)
            XCTAssertEqual(a.state, b.state, "Replicas disagree on \(a.id).")
            XCTAssertEqual(a.version, b.version, "Replicas disagree on the history of \(a.id).")
        }
    }
}
