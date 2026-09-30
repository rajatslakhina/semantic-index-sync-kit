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

    /// Two drains overlapping in time must not embed the same chunks twice.
    ///
    /// Without in-flight tracking, the second pass takes the same prefix of the
    /// queue while the first is suspended inside the provider, both write the
    /// same vectors, and both add their work to the migration counter — so a
    /// three-passage corpus reports six of six migrated. That fabricated
    /// denominator drives the demo's progress bar, and the duplicated embed
    /// burns exactly the battery budget `WorkBudget` exists to conserve.
    func testConcurrentDrainsDoNotEmbedTheSameChunksTwice() async {
        let gate = Gate()
        let coordinator = IndexCoordinator(
            device: device,
            provider: GatedProvider(epoch: epoch, gate: gate)
        )
        await coordinator.upsert(
            document: DocumentID("d"),
            passages: ["alpha passage", "beta passage", "gamma passage"],
            hash: ContentHash("v1")
        )

        let first = Task { await coordinator.drainMigration(budget: .foregroundInteractive, conditions: DeviceConditions()) }
        await gate.waitUntilEntered()

        // Asserted synchronously while the first pass is genuinely parked inside
        // `embed`, against the very function `drainMigration` uses to choose its
        // batch.
        //
        // Deliberately *not* by starting a second `drainMigration` here: without
        // the guard, that call parks on the same gate and never returns, so the
        // regression would make this test **hang** rather than fail. XCTest has
        // no per-test timeout, so a hang burns a CI job to its ceiling with no
        // failing assertion and no diagnostic. Racing it against a timeout does
        // not help either — `withTaskGroup` waits for every child before its
        // scope exits, so a stuck child blocks the race too. Asserting the
        // selection directly is what turns the regression into a named failure.
        let availableWhileParked = await coordinator.availableBatch(limit: 10)
        XCTAssertTrue(
            availableWhileParked.isEmpty,
            "A second pass would take \(availableWhileParked.count) chunks the first pass is already embedding."
        )

        await gate.release()
        let firstOutcome = await first.value
        guard case .finished(let progress) = firstOutcome else {
            return XCTFail("Expected completion, got \(firstOutcome)")
        }

        XCTAssertEqual(progress.completed, 3, "Three passages, three embeddings — not six.")
        XCTAssertEqual(progress.total, 3)
        XCTAssertEqual(progress.remaining, 0)

        let counts = await coordinator.vectorCountsByEpoch
        XCTAssertEqual(counts[epoch], 3)
        let violations = await coordinator.invariantViolations()
        XCTAssertTrue(violations.isEmpty, "Violations: \(violations)")
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

        // The outcome is the point of this test, so it is asserted rather than
        // discarded. Reporting this as a *budget* deferral would be wrong twice
        // over — the budget admitted the pass, and `adopt` rebuilt the queue —
        // so it has its own case.
        let outcome = await drain.value
        guard case .abandoned(let supersededBy) = outcome else {
            return XCTFail("Expected .abandoned, got \(outcome)")
        }
        XCTAssertEqual(supersededBy, bumped)

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

extension ReentrancyTests {

    /// `adopt` rebuilds the queue underneath a pass that is still suspended in
    /// the provider. If it does not also release that pass's claim, `inFlight`
    /// no longer describes anything in the queue, and the next pass skips a
    /// chunk nothing is working on — reporting `.alreadyDraining` about a pass
    /// that no longer exists.
    func testEpochBumpReleasesTheInFlightClaimOfTheAbandonedPass() async {
        let gate = Gate()
        let coordinator = IndexCoordinator(
            device: device,
            provider: GatedProvider(epoch: epoch, gate: gate)
        )
        await coordinator.upsert(
            document: DocumentID("d"),
            passages: ["first passage", "second passage"],
            hash: ContentHash("v1")
        )

        let drain = Task { await coordinator.drainMigration(budget: .foregroundInteractive, conditions: DeviceConditions()) }
        await gate.waitUntilEntered()

        let bumped = EmbeddingEpoch(modelIdentifier: "system.text", revision: 9, dimension: 32)
        await coordinator.adopt(provider: DeterministicEmbeddingProvider(epoch: bumped))

        // Checked while the abandoned pass is still parked — this is the window
        // the bug lived in.
        let duringViolations = await coordinator.invariantViolations()
        XCTAssertTrue(duringViolations.isEmpty, "Violations while a pass is abandoned mid-flight: \(duringViolations)")

        await gate.release()
        _ = await drain.value

        let afterViolations = await coordinator.invariantViolations()
        XCTAssertTrue(afterViolations.isEmpty, "Violations: \(afterViolations)")
        let remaining = await coordinator.migrationProgress.remaining
        XCTAssertEqual(remaining, 2)
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

        // The migration counter must describe the corpus, not the number of
        // times a racing pass happened to run. A count above the live chunk
        // total is the signature of duplicated work.
        let progress = await coordinator.migrationProgress
        let liveCount = await coordinator.liveChunks.count
        XCTAssertEqual(progress.total, liveCount, "Migration total diverged from the corpus size.")
        let vectorTotal = await coordinator.vectorCountsByEpoch.values.reduce(0, +)
        XCTAssertEqual(vectorTotal, liveCount)
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
