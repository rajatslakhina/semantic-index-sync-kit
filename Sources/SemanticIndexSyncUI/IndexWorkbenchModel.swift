import Foundation
import Observation
import SemanticIndexSync

/// Builds a pair of histories that are genuinely concurrent.
///
/// Pure and separated from the view model on purpose: this is the one piece of
/// demo logic whose correctness is not obvious by reading it, and putting it
/// behind a function makes it directly testable. Get it wrong — build the peer's
/// records from the *current* local version instead of the shared ancestor — and
/// they strictly dominate, last-writer-wins would handle them correctly too, and
/// the demonstration silently becomes worthless while still looking right.
public enum OfflineEditScenario {

    public struct LocalEdit: Sendable {
        public let id: DocumentID
        public let passages: [String]
        public let hash: ContentHash
    }

    public struct Plan: Sendable {
        public let localEdits: [LocalEdit]
        public let peerRecords: [DocumentRecord]
        public let peerPassages: [DocumentID: [String]]
    }

    /// `nil` when there is nothing live to diverge over.
    public static func make(from live: [DocumentRecord], peer: DeviceID) -> Plan? {
        guard let first = live.first else { return nil }

        var localEdits: [LocalEdit] = [
            LocalEdit(
                id: first.id,
                passages: ["Edited here while offline: the background pass now reports a typed refusal."],
                hash: ContentHash("\(first.id.raw)-local-edit")
            )
        ]
        // Built from the history the two devices last shared — NOT from the
        // version that local edit produces.
        var peerRecords: [DocumentRecord] = [
            DocumentRecord(
                id: first.id,
                state: .tombstone,
                version: first.version.incrementing(peer),
                lastWriter: peer
            )
        ]
        var peerPassages: [DocumentID: [String]] = [:]

        if let second = live.dropFirst().first {
            localEdits.append(
                LocalEdit(
                    id: second.id,
                    passages: ["Edited here while offline: conflicting with the peer's own revision."],
                    hash: ContentHash("\(second.id.raw)-local-edit")
                )
            )
            peerRecords.append(
                DocumentRecord(
                    id: second.id,
                    // Sorts above the local edit's hash, so the peer wins the
                    // content-hash tie-break deterministically.
                    state: .live(ContentHash("\(second.id.raw)-zz-peer-edit")),
                    version: second.version.incrementing(peer),
                    lastWriter: peer
                )
            )
            peerPassages[second.id] = [
                "Revised on the peer device: thermal throttling now pauses the re-index queue instead of dropping it."
            ]
        }

        return Plan(localEdits: localEdits, peerRecords: peerRecords, peerPassages: peerPassages)
    }
}

/// One line in the activity log the workbench shows.
public struct ActivityEntry: Identifiable, Sendable, Equatable {
    public enum Kind: Sendable, Equatable { case info, success, warning }
    public let id = UUID()
    public let kind: Kind
    public let title: String
    public let detail: String
}

/// Drives the workbench. Deliberately thin: it sequences calls into
/// ``IndexCoordinator`` and turns their return values into view state. None of
/// the ordering, epoch or budget rules live here — those are the library's, and
/// duplicating them in a view model is how the two drift apart.
@MainActor
@Observable
public final class IndexWorkbenchModel {

    public private(set) var query: String
    public private(set) var result: QueryResult?
    public private(set) var progress: MigrationProgress?
    public private(set) var epochCounts: [(epoch: String, count: Int)] = []
    public private(set) var activeEpochLabel: String = "—"
    public private(set) var log: [ActivityEntry] = []
    public private(set) var isWorking = false
    /// Which of the two configured spaces is active. The demo toggles rather
    /// than latching, so the rollback path — the thing retention buys — can
    /// actually be shown.
    public private(set) var isOnUpgradedEpoch = false

    public var thermalState: ThermalState = .nominal
    public var isLowPowerModeEnabled = false

    private let configuration: WorkbenchConfiguration
    private var coordinator: IndexCoordinator

    public init(configuration: WorkbenchConfiguration) {
        self.configuration = configuration
        self.query = configuration.initialQuery
        self.coordinator = IndexCoordinator(
            device: configuration.localDevice,
            provider: DeterministicEmbeddingProvider(epoch: configuration.baselineEpoch)
        )
    }

    public func setQuery(_ text: String) { query = text }

    /// Rebuilds the workbench from scratch.
    ///
    /// The peer always wins its concurrent delete, so repeatedly tapping sync
    /// walks the corpus down to nothing — correct behaviour, and a dead end for
    /// anyone exploring. This puts it back.
    public func reset() async {
        isWorking = true
        defer { isWorking = false }
        coordinator = IndexCoordinator(
            device: configuration.localDevice,
            provider: DeterministicEmbeddingProvider(epoch: configuration.baselineEpoch)
        )
        isOnUpgradedEpoch = false
        log.removeAll()
        result = nil
        progress = nil
        epochCounts = []
        await start()
    }

    /// Seeds the corpus, embeds all of it, and runs the opening query, so the
    /// first thing on screen is a working search over a fully covered index
    /// rather than an empty state waiting for a tap.
    public func start() async {
        guard log.isEmpty else { return }
        let alreadyWorking = isWorking
        isWorking = true
        defer { isWorking = alreadyWorking }

        for seed in configuration.seeds {
            await coordinator.upsert(document: seed.id, passages: seed.passages, hash: seed.hash)
        }
        note(.info, "Seeded \(configuration.seeds.count) documents", await chunkSummary())

        await drainFully(budget: configuration.interactiveBudget, label: "Initial embedding")
        await refresh()
        note(.success, "Index ready", "All passages embedded in \(configuration.baselineEpoch.description).")
    }

    public func runQuery() async {
        isWorking = true
        defer { isWorking = false }
        await refresh()
    }

    /// Simulates the OS shipping a new revision of the on-device model — and,
    /// on a second tap, that rollout being pulled again.
    ///
    /// The rollback is the interesting half: because vectors are keyed by chunk
    /// *and* epoch, the previous space is still on disk, so going back restores
    /// full coverage with no re-index at all. That is the payoff for the disk
    /// the forward migration deliberately spends.
    public func toggleModelRevision() async {
        isWorking = true
        defer { isWorking = false }

        let target = isOnUpgradedEpoch ? configuration.baselineEpoch : configuration.upgradedEpoch
        let rollingBack = isOnUpgradedEpoch
        let progress = await coordinator.adopt(provider: DeterministicEmbeddingProvider(epoch: target))
        isOnUpgradedEpoch.toggle()

        if rollingBack {
            note(
                .success,
                "Rolled back to \(target.description)",
                progress.remaining == 0
                    ? "Coverage restored instantly: \(progress.completed) passages still had vectors in this space, so nothing needed re-embedding."
                    : "\(progress.remaining) passages still need re-embedding."
            )
        } else {
            note(
                .warning,
                "Embedding model bumped to \(target.description)",
                "\(progress.remaining) passages now sit in the previous space and are excluded from semantic scoring until re-embedded. The old vectors are kept, not deleted."
            )
        }
        await refresh()
    }

    /// One budgeted background pass, under whatever conditions the toggles say.
    public func runBackgroundPass() async {
        isWorking = true
        defer { isWorking = false }

        let conditions = DeviceConditions(
            thermalState: thermalState,
            isOnExternalPower: false,
            batteryFraction: 0.74,
            isLowPowerModeEnabled: isLowPowerModeEnabled
        )
        let outcome = await coordinator.drainMigration(
            budget: configuration.backgroundBudget,
            conditions: conditions
        )

        switch outcome {
        case .idle:
            note(.info, "Nothing queued", "Every live passage already has a vector in the active epoch.")
        case .deferred(let rejection):
            note(.warning, "Pass deferred", "\(rejection). The queue is untouched and resumes on the next pass.")
        case .alreadyDraining(let value):
            note(.info, "Another pass is already running", "\(value.remaining) passages are in flight; this pass took no duplicate work.")
        case .progressed(let value):
            note(.info, "Background pass ran", "\(value.completed) of \(value.total) re-embedded; \(value.remaining) remaining.")
        case .finished(let value):
            note(.success, "Migration complete", "All \(value.total) passages are now in \(value.targetEpoch.description).")
        case .abandoned(let epoch):
            note(.warning, "Pass abandoned", "The active space changed to \(epoch.description) mid-pass, so its results were discarded. Not a budget refusal — the queue has already been rebuilt.")
        case .providerFailed(let message):
            note(.warning, "Provider failed", "\(message). Affected passages stay queued for retry.")
        }
        await refresh()
    }

    /// Applies a batch from the peer device that is *genuinely concurrent* with
    /// local writes — the only case the version-vector design exists for.
    ///
    /// Both sides are built from the history the two devices last shared, then
    /// advanced independently: this device edits, and the peer (which never saw
    /// that edit) deletes one document and edits another. Neither version vector
    /// dominates the other, so `Reconciler` reaches its `.concurrent` branch.
    ///
    /// Building the peer's records from the *current* local version instead
    /// would make them strictly newer, and last-writer-wins would handle them
    /// correctly too — the demo would prove nothing.
    public func syncFromPeer() async {
        isWorking = true
        defer { isWorking = false }

        let live = await coordinator.exportManifest().filter { !$0.state.isTombstone }
        guard let plan = OfflineEditScenario.make(from: live, peer: configuration.peerDevice) else {
            note(.warning, "Nothing to sync", "The local manifest has no live documents.")
            return
        }

        // This device's own offline edits, applied first so the peer's batch is
        // concurrent with them rather than newer than them.
        for edit in plan.localEdits {
            await coordinator.upsert(document: edit.id, passages: edit.passages, hash: edit.hash)
        }

        let report = await coordinator.applyRemote(records: plan.peerRecords, passages: plan.peerPassages)
        note(
            .success,
            "Synced from \(configuration.peerDevice.raw)",
            "concurrent merges resolved \(report.concurrentResolved) · tombstones upheld \(report.tombstonesUpheld) · superseded \(report.supersededByRemote) · stale ignored \(report.staleRemotesIgnored) · re-queued \(report.invalidated.count)"
        )
        await refresh()
    }

    // MARK: Private

    private func drainFully(budget: WorkBudget, label: String) async {
        // Bounded rather than `while true`: an unbounded drain loop in a view
        // model is a hang waiting for a provider bug to trigger it.
        let maximumPasses = 64
        for _ in 0..<maximumPasses {
            let outcome = await coordinator.drainMigration(budget: budget, conditions: DeviceConditions())
            switch outcome {
            case .progressed, .alreadyDraining:
                continue
            case .idle, .finished:
                return
            case .deferred(let rejection):
                note(.warning, "\(label) deferred", rejection.description)
                return
            case .abandoned(let epoch):
                note(.warning, "\(label) abandoned", "Superseded by \(epoch.description).")
                return
            case .providerFailed(let message):
                note(.warning, "\(label) failed", message)
                return
            }
        }
        note(.warning, "\(label) stopped early", "Hit the \(maximumPasses)-pass ceiling.")
    }

    private func refresh() async {
        result = await coordinator.search(query, limit: 6)
        progress = await coordinator.migrationProgress
        activeEpochLabel = await coordinator.currentEpoch.description
        let counts = await coordinator.vectorCountsByEpoch
        epochCounts = counts
            .map { (epoch: $0.key.description, count: $0.value) }
            .sorted { $0.epoch < $1.epoch }
    }

    private func chunkSummary() async -> String {
        let count = await coordinator.liveChunks.count
        return "\(count) passages queued for embedding."
    }

    private func note(_ kind: ActivityEntry.Kind, _ title: String, _ detail: String) {
        log.insert(ActivityEntry(kind: kind, title: title, detail: detail), at: 0)
        // Bounded so a long session cannot grow the log without limit.
        if log.count > 40 { log.removeLast(log.count - 40) }
    }
}
