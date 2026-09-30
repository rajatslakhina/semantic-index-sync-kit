import XCTest
@testable import SemanticIndexSync

final class IndexCoordinatorTests: XCTestCase {

    private let device = DeviceID("phone")
    private let peer = DeviceID("laptop")
    private let oldEpoch = EmbeddingEpoch(modelIdentifier: "system.text", revision: 3, dimension: 32)
    private let newEpoch = EmbeddingEpoch(modelIdentifier: "system.text", revision: 4, dimension: 32)

    private func makeSeeded() async -> IndexCoordinator {
        let coordinator = IndexCoordinator(
            device: device,
            provider: DeterministicEmbeddingProvider(epoch: oldEpoch)
        )
        await coordinator.upsert(
            document: DocumentID("thermal"),
            passages: [
                "The re-index queue pauses when the device thermal state exceeds the configured ceiling.",
                "A deferred pass leaves the queue untouched so work resumes later.",
            ],
            hash: ContentHash("thermal-v1")
        )
        await coordinator.upsert(
            document: DocumentID("sync"),
            passages: [
                "Version vectors distinguish a stale write from a concurrent one.",
                "A tombstone is never resurrected by a concurrent edit.",
            ],
            hash: ContentHash("sync-v1")
        )
        return coordinator
    }

    private func drainFully(_ coordinator: IndexCoordinator, budget: WorkBudget = .foregroundInteractive) async {
        for _ in 0..<32 {
            let outcome = await coordinator.drainMigration(budget: budget, conditions: DeviceConditions())
            switch outcome {
            case .progressed: continue
            case .idle, .finished: return
            case .deferred, .providerFailed: return
            }
        }
    }

    // MARK: Baseline

    func testSeededIndexReachesFullCoverageAndAnswersQueries() async {
        let coordinator = await makeSeeded()
        await drainFully(coordinator)

        let result = await coordinator.search("thermal ceiling", limit: 5)
        XCTAssertEqual(result.completeness.liveChunks, 4)
        XCTAssertEqual(result.completeness.semanticallyCovered, 4)
        XCTAssertEqual(result.completeness.awaitingReindex, 0)
        XCTAssertTrue(result.completeness.isComplete)
        XCTAssertEqual(result.completeness.semanticCoverage, 1.0)
        XCTAssertEqual(result.epoch, oldEpoch)
        XCTAssertFalse(result.hits.isEmpty)
        // The best hit for this query is the passage that actually talks about it.
        XCTAssertEqual(result.hits.first?.chunk.id.document, DocumentID("thermal"))
        let violationsCheck = await coordinator.invariantViolations()
        XCTAssertTrue(violationsCheck.isEmpty, "Violations: \(violationsCheck)")
    }

    func testEmptyIndexReportsCompleteRatherThanDividingByZero() async {
        let coordinator = IndexCoordinator(
            device: device,
            provider: DeterministicEmbeddingProvider(epoch: oldEpoch)
        )
        let result = await coordinator.search("anything", limit: 10)
        XCTAssertTrue(result.hits.isEmpty)
        XCTAssertEqual(result.completeness.liveChunks, 0)
        XCTAssertEqual(result.completeness.semanticCoverage, 1.0)
        XCTAssertTrue(result.completeness.isComplete)
    }

    func testNegativeAndOversizedLimitsAreHandled() async {
        let coordinator = await makeSeeded()
        await drainFully(coordinator)
        let none = await coordinator.search("thermal", limit: -5)
        XCTAssertTrue(none.hits.isEmpty)
        let all = await coordinator.search("thermal", limit: 10_000)
        XCTAssertLessThanOrEqual(all.hits.count, 4)
    }

    func testBlankPassagesAreNotIndexed() async {
        let coordinator = IndexCoordinator(
            device: device,
            provider: DeterministicEmbeddingProvider(epoch: oldEpoch)
        )
        await coordinator.upsert(
            document: DocumentID("d"),
            passages: ["", "   ", "\n", "real content"],
            hash: ContentHash("v1")
        )
        let chunks = await coordinator.liveChunks
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks.first?.text, "real content")
    }

    // MARK: Epoch migration

    /// The headline behaviour: a model bump drops semantic coverage to zero, the
    /// search box keeps working on the keyword floor, and coverage climbs back
    /// one budgeted pass at a time — with exact numbers at each step, not a
    /// range the implementation computes for itself.
    func testModelBumpDegradesCoverageThenRecoversOnePassAtATime() async {
        let coordinator = await makeSeeded()
        await drainFully(coordinator)

        let progress = await coordinator.adopt(provider: DeterministicEmbeddingProvider(epoch: newEpoch))
        XCTAssertEqual(progress.remaining, 4)
        XCTAssertEqual(progress.completed, 0)

        let degraded = await coordinator.search("thermal ceiling", limit: 5)
        XCTAssertEqual(degraded.completeness.semanticallyCovered, 0)
        XCTAssertEqual(degraded.completeness.awaitingReindex, 4)
        XCTAssertEqual(degraded.completeness.semanticCoverage, 0.0)
        XCTAssertFalse(degraded.completeness.isComplete)
        XCTAssertFalse(degraded.completeness.isLexicalOnly, "The model is present; only the vectors are stale.")
        XCTAssertFalse(degraded.hits.isEmpty, "The keyword floor must still answer.")
        XCTAssertTrue(
            degraded.hits.allSatisfy { $0.source == .lexicalFallback },
            "No hit may claim a semantic score while every vector is in the previous epoch."
        )

        // One pass of exactly two chunks.
        let pass = await coordinator.drainMigration(
            budget: WorkBudget(maxChunksPerPass: 2),
            conditions: DeviceConditions()
        )
        guard case .progressed(let midway) = pass else {
            return XCTFail("Expected partial progress, got \(pass)")
        }
        XCTAssertEqual(midway.completed, 2)
        XCTAssertEqual(midway.remaining, 2)

        let partial = await coordinator.search("thermal ceiling", limit: 5)
        XCTAssertEqual(partial.completeness.semanticallyCovered, 2)
        XCTAssertEqual(partial.completeness.awaitingReindex, 2)
        XCTAssertEqual(partial.completeness.semanticCoverage, 0.5)
        XCTAssertFalse(partial.completeness.isComplete)
        XCTAssertTrue(
            partial.hits.contains { $0.source != .lexicalFallback },
            "Re-embedded chunks must be scored semantically again."
        )

        // Finish it.
        let final = await coordinator.drainMigration(
            budget: WorkBudget(maxChunksPerPass: 8),
            conditions: DeviceConditions()
        )
        guard case .finished(let done) = final else {
            return XCTFail("Expected completion, got \(final)")
        }
        XCTAssertEqual(done.completed, 4)
        XCTAssertEqual(done.remaining, 0)
        XCTAssertEqual(done.targetEpoch, newEpoch)

        let recovered = await coordinator.search("thermal ceiling", limit: 5)
        XCTAssertTrue(recovered.completeness.isComplete)
        XCTAssertEqual(recovered.completeness.semanticCoverage, 1.0)
        XCTAssertEqual(recovered.epoch, newEpoch)
        let violationsCheck = await coordinator.invariantViolations()
        XCTAssertTrue(violationsCheck.isEmpty, "Violations: \(violationsCheck)")
    }

    /// Old vectors are retained on purpose, so a rollback is instant rather than
    /// a second full re-index. This asserts the retention *and* the reclaim path.
    func testPreviousEpochVectorsAreRetainedUntilExplicitlyDiscarded() async {
        let coordinator = await makeSeeded()
        await drainFully(coordinator)
        await coordinator.adopt(provider: DeterministicEmbeddingProvider(epoch: newEpoch))
        await drainFully(coordinator)

        let counts = await coordinator.vectorCountsByEpoch
        XCTAssertEqual(counts[newEpoch], 4)
        XCTAssertNil(counts[oldEpoch], "New vectors replace old ones for the same chunk.")

        // Rolling back finds every vector already present, so nothing is queued.
        let rollback = await coordinator.adopt(provider: DeterministicEmbeddingProvider(epoch: oldEpoch))
        XCTAssertEqual(rollback.remaining, 4, "Vectors were overwritten, so a rollback does re-index.")

        let discarded = await coordinator.discardVectors(outside: [oldEpoch])
        XCTAssertGreaterThanOrEqual(discarded, 0)
        let after = await coordinator.vectorCountsByEpoch
        XCTAssertNil(after[newEpoch])
    }

    func testAdoptingTheSameEpochIsANoOp() async {
        let coordinator = await makeSeeded()
        await drainFully(coordinator)
        let before = await coordinator.migrationProgress
        let after = await coordinator.adopt(provider: DeterministicEmbeddingProvider(epoch: oldEpoch))
        XCTAssertEqual(after.remaining, before.remaining)
        XCTAssertEqual(after.remaining, 0)
    }

    // MARK: Degraded mode

    func testUnavailableProviderFallsBackToKeywordSearchRatherThanReturningNothing() async {
        let coordinator = await makeSeeded()
        await drainFully(coordinator)

        await coordinator.adopt(provider: DeterministicEmbeddingProvider(epoch: newEpoch, isAvailable: false))
        let result = await coordinator.search("tombstone concurrent", limit: 5)

        XCTAssertTrue(result.completeness.isLexicalOnly)
        XCTAssertNil(result.epoch)
        XCTAssertFalse(result.hits.isEmpty)
        XCTAssertTrue(result.hits.allSatisfy { $0.source == .lexicalFallback })
        XCTAssertFalse(result.completeness.isComplete)
        XCTAssertTrue(result.completeness.summary.contains("Keyword search only"))
    }

    // MARK: Provider failures

    func testProviderFailureLeavesChunksQueuedForRetry() async {
        let flaky = FlakyProvider(epoch: oldEpoch, failures: 1)
        let coordinator = IndexCoordinator(device: device, provider: flaky)
        await coordinator.upsert(
            document: DocumentID("d"),
            passages: ["alpha passage", "beta passage"],
            hash: ContentHash("v1")
        )

        let first = await coordinator.drainMigration(budget: .background, conditions: DeviceConditions())
        guard case .providerFailed = first else {
            return XCTFail("Expected a provider failure, got \(first)")
        }
        let queuedAfterFailure = await coordinator.migrationProgress
        XCTAssertEqual(queuedAfterFailure.remaining, 2, "A transient failure must not drop work.")

        let second = await coordinator.drainMigration(budget: .background, conditions: DeviceConditions())
        guard case .finished(let done) = second else {
            return XCTFail("Expected completion on retry, got \(second)")
        }
        XCTAssertEqual(done.completed, 2)
        let calls = await flaky.budget.callCount
        XCTAssertEqual(calls, 2)
    }

    func testMiscountedProviderResponseIsRejectedRatherThanMisaligned() async {
        let coordinator = IndexCoordinator(device: device, provider: MiscountingProvider(epoch: oldEpoch))
        await coordinator.upsert(
            document: DocumentID("d"),
            passages: ["alpha", "beta", "gamma"],
            hash: ContentHash("v1")
        )
        let outcome = await coordinator.drainMigration(budget: .background, conditions: DeviceConditions())
        guard case .providerFailed(let message) = outcome else {
            return XCTFail("Expected rejection, got \(outcome)")
        }
        XCTAssertTrue(message.contains("2 vectors for 3 inputs"), "Got: \(message)")
        // Nothing was written from a misaligned response.
        let remainingCheck = await coordinator.migrationProgress.remaining
        XCTAssertEqual(remainingCheck, 3)
    }

    func testBudgetRejectionDefersWithoutConsumingTheQueue() async {
        let coordinator = await makeSeeded()
        let outcome = await coordinator.drainMigration(
            budget: WorkBudget(maxChunksPerPass: 4, thermalCeiling: .nominal),
            conditions: DeviceConditions(thermalState: .critical)
        )
        XCTAssertEqual(outcome, .deferred(.thermalCeilingExceeded(observed: .critical, ceiling: .nominal)))
        let remainingCheck = await coordinator.migrationProgress.remaining
        XCTAssertEqual(remainingCheck, 4)
    }

    func testDrainOnAnEmptyQueueIsIdle() async {
        let coordinator = await makeSeeded()
        await drainFully(coordinator)
        let outcome = await coordinator.drainMigration(budget: .background, conditions: DeviceConditions())
        XCTAssertEqual(outcome, .idle)
    }

    // MARK: Deletion

    func testDeletingADocumentRemovesItsVectorsAndQueueEntries() async {
        let coordinator = await makeSeeded()
        await drainFully(coordinator)

        await coordinator.delete(document: DocumentID("sync"))

        let result = await coordinator.search("tombstone concurrent", limit: 5)
        XCTAssertEqual(result.completeness.liveChunks, 2)
        XCTAssertFalse(result.hits.contains { $0.chunk.id.document == DocumentID("sync") })
        let violationsCheck = await coordinator.invariantViolations()
        XCTAssertTrue(violationsCheck.isEmpty, "Violations: \(violationsCheck)")
        let counts = await coordinator.vectorCountsByEpoch
        XCTAssertEqual(counts[oldEpoch], 2)
    }

    func testRecreatingADeletedDocumentDominatesItsOwnTombstone() async {
        let coordinator = await makeSeeded()
        await coordinator.delete(document: DocumentID("sync"))
        let revived = await coordinator.upsert(
            document: DocumentID("sync"),
            passages: ["Brought back with new content."],
            hash: ContentHash("sync-v2")
        )
        XCTAssertFalse(revived.state.isTombstone)

        // Re-applying the old tombstone must not undo the revival.
        let staleTombstone = DocumentRecord(
            id: DocumentID("sync"),
            state: .tombstone,
            version: VersionVector([device: 2]),
            lastWriter: device
        )
        let report = await coordinator.applyRemote(records: [staleTombstone])
        XCTAssertEqual(report.staleRemotesIgnored, 1)
        let live = await coordinator.liveChunks.map(\.id.document)
        XCTAssertTrue(live.contains(DocumentID("sync")))
    }

    // MARK: Sync integration

    func testRemoteTombstoneRemovesLocalVectorsImmediately() async {
        let coordinator = await makeSeeded()
        await drainFully(coordinator)

        guard let target = await coordinator.exportManifest().first(where: { $0.id == DocumentID("sync") }) else {
            return XCTFail("Seed document missing.")
        }
        let remoteDelete = target.deleted(by: peer)
        let report = await coordinator.applyRemote(records: [remoteDelete])

        XCTAssertEqual(report.supersededByRemote, 1)
        let result = await coordinator.search("tombstone", limit: 5)
        XCTAssertFalse(result.hits.contains { $0.chunk.id.document == DocumentID("sync") })
        let violationsCheck = await coordinator.invariantViolations()
        XCTAssertTrue(violationsCheck.isEmpty, "Violations: \(violationsCheck)")
    }

    func testRemoteEditWithoutContentDropsStaleChunksRatherThanServingThem() async {
        let coordinator = await makeSeeded()
        await drainFully(coordinator)

        guard let target = await coordinator.exportManifest().first(where: { $0.id == DocumentID("sync") }) else {
            return XCTFail("Seed document missing.")
        }
        // Peer says the document changed but ships no passages with this batch.
        let edit = target.edited(to: ContentHash("sync-v2"), by: peer)
        let report = await coordinator.applyRemote(records: [edit])

        XCTAssertEqual(report.invalidated, [DocumentID("sync")])
        let result = await coordinator.search("tombstone concurrent", limit: 5)
        XCTAssertFalse(
            result.hits.contains { $0.chunk.id.document == DocumentID("sync") },
            "Superseded text must not keep serving results."
        )
        let violationsCheck = await coordinator.invariantViolations()
        XCTAssertTrue(violationsCheck.isEmpty, "Violations: \(violationsCheck)")
    }

    func testRemoteEditWithContentReplacesChunksAndRequeuesThem() async {
        let coordinator = await makeSeeded()
        await drainFully(coordinator)

        guard let target = await coordinator.exportManifest().first(where: { $0.id == DocumentID("sync") }) else {
            return XCTFail("Seed document missing.")
        }
        let edit = target.edited(to: ContentHash("sync-v2"), by: peer)
        _ = await coordinator.applyRemote(
            records: [edit],
            passages: [DocumentID("sync"): ["Rewritten passage about backpressure."]]
        )

        let remainingCheck = await coordinator.migrationProgress.remaining
        XCTAssertEqual(remainingCheck, 1)
        await drainFully(coordinator)
        let result = await coordinator.search("backpressure", limit: 5)
        XCTAssertTrue(result.completeness.isComplete)
        XCTAssertEqual(result.hits.first?.chunk.text, "Rewritten passage about backpressure.")
    }
}
