import XCTest
@testable import SemanticIndexSyncUI
import SemanticIndexSync

/// The demo app's own logic, under test.
///
/// This exists because of a specific near-miss: an earlier version of
/// `syncFromPeer` built the peer's records from the *current* local version, so
/// they strictly dominated, `Reconciler` never reached its `.concurrent` branch,
/// and the whole demonstration was empty — while the on-screen copy, the view
/// model's own doc comment and the README all said otherwise. Nothing caught it,
/// because the view model was behind `#if canImport(SwiftUI)` and therefore
/// untestable on CI. It no longer is, and these tests exercise the shipped code
/// rather than a re-implementation of it.
final class OfflineEditScenarioTests: XCTestCase {

    private let local = DeviceID("iphone")
    private let peer = DeviceID("ipad")

    private func seeded() -> [DocumentRecord] {
        [
            DocumentRecord.live(DocumentID("a"), hash: ContentHash("a-v1"), by: local),
            DocumentRecord.live(DocumentID("b"), hash: ContentHash("b-v1"), by: local),
        ]
    }

    func testEmptyManifestProducesNoPlan() {
        XCTAssertNil(OfflineEditScenario.make(from: [], peer: peer))
    }

    /// The load-bearing property, asserted directly on the shipped builder.
    func testPeerRecordsAreConcurrentWithTheLocalEditsNotNewerThanThem() {
        let live = seeded()
        guard let plan = OfflineEditScenario.make(from: live, peer: peer) else {
            return XCTFail("Expected a plan.")
        }
        XCTAssertEqual(plan.localEdits.count, 2)
        XCTAssertEqual(plan.peerRecords.count, 2)

        for edit in plan.localEdits {
            guard let base = live.first(where: { $0.id == edit.id }),
                  let peerRecord = plan.peerRecords.first(where: { $0.id == edit.id }) else {
                return XCTFail("Plan is missing \(edit.id).")
            }
            // What the local device's version becomes once it applies its edit.
            let localAfterEdit = base.edited(to: edit.hash, by: local)
            XCTAssertEqual(
                VersionVector.order(localAfterEdit.version, peerRecord.version), .concurrent,
                "\(edit.id) must be a genuine conflict; a dominating peer record would be handled correctly by last-writer-wins too, and would prove nothing."
            )
        }
    }

    func testThePeerDeletesOneDocumentAndEditsTheOther() {
        guard let plan = OfflineEditScenario.make(from: seeded(), peer: peer) else {
            return XCTFail("Expected a plan.")
        }
        XCTAssertEqual(plan.peerRecords.filter { $0.state.isTombstone }.count, 1)
        XCTAssertEqual(plan.peerRecords.filter { !$0.state.isTombstone }.count, 1)
        XCTAssertTrue(plan.peerRecords.allSatisfy { $0.lastWriter == peer })

        // Only the surviving edit ships content; a tombstone has none.
        XCTAssertEqual(plan.peerPassages.count, 1)
        XCTAssertNil(plan.peerPassages[DocumentID("a")])
        XCTAssertNotNil(plan.peerPassages[DocumentID("b")])
    }

    /// The peer's edit must win its content-hash tie-break deterministically, or
    /// the shipped passages would not be the ones indexed.
    func testThePeerEditSortsAboveTheLocalEdit() {
        guard let plan = OfflineEditScenario.make(from: seeded(), peer: peer) else {
            return XCTFail("Expected a plan.")
        }
        guard let peerEdit = plan.peerRecords.first(where: { !$0.state.isTombstone }),
              let peerHash = peerEdit.state.contentHash,
              let localEdit = plan.localEdits.first(where: { $0.id == peerEdit.id }) else {
            return XCTFail("Plan is missing the concurrent edit.")
        }
        XCTAssertGreaterThan(peerHash, localEdit.hash)
    }

    func testASingleLiveDocumentStillProducesAConcurrentDeleteConflict() {
        let base = DocumentRecord.live(DocumentID("only"), hash: ContentHash("v1"), by: local)
        guard let plan = OfflineEditScenario.make(from: [base], peer: peer) else {
            return XCTFail("Expected a plan.")
        }
        XCTAssertEqual(plan.localEdits.count, 1)
        XCTAssertEqual(plan.peerRecords.count, 1)
        XCTAssertTrue(plan.peerRecords[0].state.isTombstone)
        XCTAssertTrue(plan.peerPassages.isEmpty)

        // The counts above would all hold for a peer record built from the
        // *current* local version — the exact regression this file exists to
        // catch — so the causal relationship is asserted here too.
        let localAfterEdit = base.edited(to: plan.localEdits[0].hash, by: local)
        XCTAssertEqual(
            VersionVector.order(localAfterEdit.version, plan.peerRecords[0].version), .concurrent,
            "A dominating peer delete would be handled correctly by last-writer-wins too, and would prove nothing."
        )
    }
}

/// End-to-end through the real view model, against the real coordinator.
@MainActor
final class IndexWorkbenchModelTests: XCTestCase {

    private func makeConfiguration() -> WorkbenchConfiguration {
        WorkbenchConfiguration(
            localDevice: DeviceID("iphone"),
            peerDevice: DeviceID("ipad"),
            baselineEpoch: EmbeddingEpoch(modelIdentifier: "m", revision: 3, dimension: 64),
            upgradedEpoch: EmbeddingEpoch(modelIdentifier: "m", revision: 4, dimension: 64),
            seeds: [
                SeedDocument(id: "alpha", passages: ["thermal budget pauses the queue", "deferred passes resume later"], revision: "v1"),
                SeedDocument(id: "beta", passages: ["version vectors order concurrent writes", "tombstones are never resurrected"], revision: "v1"),
            ],
            backgroundBudget: WorkBudget(maxChunksPerPass: 2),
            interactiveBudget: .foregroundInteractive,
            initialQuery: "thermal budget"
        )
    }

    /// The first thing on screen must be a working search over a covered index,
    /// not an empty state waiting for a tap.
    func testLaunchStateIsFullyIndexedAndHasResults() async {
        let model = IndexWorkbenchModel(configuration: makeConfiguration())
        await model.start()

        guard let result = model.result else { return XCTFail("No result after start().") }
        XCTAssertEqual(result.completeness.liveChunks, 4)
        XCTAssertEqual(result.completeness.semanticallyCovered, 4)
        XCTAssertTrue(result.completeness.isComplete)
        XCTAssertFalse(result.hits.isEmpty)
        XCTAssertFalse(model.isOnUpgradedEpoch)
    }

    /// The headline sequence: bump, degrade, recover one budgeted pass at a time.
    func testModelBumpDegradesThenBudgetedPassesRecoverCoverage() async {
        let model = IndexWorkbenchModel(configuration: makeConfiguration())
        await model.start()

        await model.toggleModelRevision()
        XCTAssertTrue(model.isOnUpgradedEpoch)
        XCTAssertEqual(model.result?.completeness.semanticallyCovered, 0)
        XCTAssertEqual(model.result?.completeness.awaitingReindex, 4)
        XCTAssertEqual(model.result?.completeness.isComplete, false)
        // The keyword floor still answers.
        XCTAssertFalse(model.result?.hits.isEmpty ?? true)
        XCTAssertTrue(model.result?.hits.allSatisfy { $0.source == .lexicalFallback } ?? false)

        await model.runBackgroundPass()   // budget is 2 per pass
        XCTAssertEqual(model.result?.completeness.semanticallyCovered, 2)

        await model.runBackgroundPass()
        XCTAssertEqual(model.result?.completeness.semanticallyCovered, 4)
        XCTAssertEqual(model.result?.completeness.isComplete, true)
    }

    /// And the rollback is the payoff for keying vectors by chunk *and* epoch:
    /// it must cost no re-embedding at all.
    func testRollingBackRestoresFullCoverageWithoutReindexing() async {
        let model = IndexWorkbenchModel(configuration: makeConfiguration())
        await model.start()
        await model.toggleModelRevision()
        await model.runBackgroundPass()
        await model.runBackgroundPass()
        XCTAssertEqual(model.result?.completeness.isComplete, true)

        await model.toggleModelRevision()   // roll back
        XCTAssertFalse(model.isOnUpgradedEpoch)
        XCTAssertEqual(model.progress?.remaining, 0, "A rollback must queue no work.")
        XCTAssertEqual(model.result?.completeness.isComplete, true)
        XCTAssertEqual(model.result?.completeness.semanticallyCovered, 4)
    }

    /// A refusal must be reported rather than silently doing nothing — and the
    /// queue must survive it.
    func testThermalCeilingDefersTheBackgroundPassWithoutLosingWork() async {
        let model = IndexWorkbenchModel(configuration: makeConfiguration())
        await model.start()
        await model.toggleModelRevision()

        model.thermalState = .serious
        await model.runBackgroundPass()
        XCTAssertEqual(model.progress?.remaining, 4, "A deferred pass must not consume the queue.")
        XCTAssertEqual(model.result?.completeness.semanticallyCovered, 0)
        XCTAssertTrue(
            model.log.contains { $0.detail.contains("thermal serious exceeds ceiling fair") },
            "The refusal must name the observed state and the ceiling. Log: \(model.log.map(\.detail))"
        )

        // Cooling down lets it proceed.
        model.thermalState = .nominal
        await model.runBackgroundPass()
        XCTAssertEqual(model.result?.completeness.semanticallyCovered, 2)
    }

    /// The sync scenario, end to end through the shipped code path.
    func testSyncFromPeerResolvesConcurrentMergesAndUpholdsTheTombstone() async {
        let model = IndexWorkbenchModel(configuration: makeConfiguration())
        await model.start()

        await model.syncFromPeer()
        guard let entry = model.log.first else { return XCTFail("No log entry.") }
        XCTAssertTrue(entry.title.hasPrefix("Synced from"))
        XCTAssertTrue(
            entry.detail.contains("concurrent merges resolved 2"),
            "Both documents must conflict. Got: \(entry.detail)"
        )
        XCTAssertTrue(
            entry.detail.contains("tombstones upheld 1"),
            "The peer's offline delete must win its conflict. Got: \(entry.detail)"
        )
        XCTAssertTrue(entry.detail.contains("stale ignored 0"))
    }

    /// Repeated syncs retire documents, so the reset has to genuinely restore.
    func testResetRestoresTheCorpusAfterRepeatedSyncs() async {
        let model = IndexWorkbenchModel(configuration: makeConfiguration())
        await model.start()
        XCTAssertEqual(model.result?.completeness.liveChunks, 4)

        await model.syncFromPeer()
        await model.syncFromPeer()
        XCTAssertEqual(model.result?.completeness.liveChunks, 0, "Both documents have been retired.")

        await model.reset()
        XCTAssertEqual(model.result?.completeness.liveChunks, 4)
        XCTAssertEqual(model.result?.completeness.isComplete, true)
        XCTAssertFalse(model.isOnUpgradedEpoch)
    }

    /// The activity log is bounded, so a long session cannot grow it without limit.
    func testActivityLogIsBounded() async {
        let model = IndexWorkbenchModel(configuration: makeConfiguration())
        await model.start()
        for _ in 0..<60 { await model.runBackgroundPass() }
        XCTAssertLessThanOrEqual(model.log.count, 40)
    }
}
