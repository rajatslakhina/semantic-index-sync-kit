#if canImport(SwiftUI)
import Foundation
import Observation
import SemanticIndexSync

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
    public private(set) var hasUpgraded = false

    public var thermalState: ThermalState = .nominal
    public var isLowPowerModeEnabled = false

    private let configuration: WorkbenchConfiguration
    private let coordinator: IndexCoordinator

    public init(configuration: WorkbenchConfiguration) {
        self.configuration = configuration
        self.query = configuration.initialQuery
        self.coordinator = IndexCoordinator(
            device: configuration.localDevice,
            provider: DeterministicEmbeddingProvider(epoch: configuration.baselineEpoch)
        )
    }

    public func setQuery(_ text: String) { query = text }

    /// Seeds the corpus, embeds all of it, and runs the opening query, so the
    /// first thing on screen is a working search over a fully covered index
    /// rather than an empty state waiting for a tap.
    public func start() async {
        guard log.isEmpty else { return }
        isWorking = true
        defer { isWorking = false }

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

    /// Simulates the OS shipping a new revision of the on-device model.
    public func shipModelUpdate() async {
        guard !hasUpgraded else { return }
        isWorking = true
        defer { isWorking = false }

        let updated = DeterministicEmbeddingProvider(epoch: configuration.upgradedEpoch)
        let progress = await coordinator.adopt(provider: updated)
        hasUpgraded = true
        note(
            .warning,
            "Embedding model bumped to \(configuration.upgradedEpoch.description)",
            "\(progress.remaining) passages now sit in the previous space and are excluded from semantic scoring until re-embedded. Existing vectors are kept, not deleted."
        )
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
        case .progressed(let value):
            note(.info, "Background pass ran", "\(value.completed) of \(value.total) re-embedded; \(value.remaining) remaining.")
        case .finished(let value):
            note(.success, "Migration complete", "All \(value.total) passages are now in \(value.targetEpoch.description).")
        case .providerFailed(let message):
            note(.warning, "Provider failed", "\(message). Affected passages stay queued for retry.")
        }
        await refresh()
    }

    /// Applies a batch from the peer device containing exactly the two cases the
    /// design exists for: a delete that a clock-based merge would resurrect, and
    /// a genuinely concurrent edit.
    public func syncFromPeer() async {
        isWorking = true
        defer { isWorking = false }

        let local = await coordinator.exportManifest()
        guard let first = local.first(where: { !$0.state.isTombstone }) else {
            note(.warning, "Nothing to sync", "The local manifest has no live documents.")
            return
        }

        // The peer deleted this document while offline. Its version vector shows
        // it saw the same history we did and then moved past it.
        let peerDelete = first.deleted(by: configuration.peerDevice)

        // And the peer edited a second document concurrently with us.
        var batch = [peerDelete]
        var passages: [DocumentID: [String]] = [:]
        if let second = local.dropFirst().first(where: { !$0.state.isTombstone }) {
            let concurrentEdit = DocumentRecord(
                id: second.id,
                state: .live(ContentHash("\(second.id.raw)-peer-edit")),
                version: second.version.incrementing(configuration.peerDevice),
                lastWriter: configuration.peerDevice
            )
            batch.append(concurrentEdit)
            passages[second.id] = [
                "Revised on the peer device: thermal throttling now pauses the re-index queue instead of dropping it."
            ]
        }

        let report = await coordinator.applyRemote(records: batch, passages: passages)
        note(
            .success,
            "Synced from \(configuration.peerDevice.raw)",
            "applied \(report.supersededByRemote) · stale ignored \(report.staleRemotesIgnored) · concurrent resolved \(report.concurrentResolved) · tombstones upheld \(report.tombstonesUpheld) · re-queued \(report.invalidated.count)"
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
            case .progressed:
                continue
            case .idle, .finished:
                return
            case .deferred(let rejection):
                note(.warning, "\(label) deferred", rejection.description)
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
#endif
