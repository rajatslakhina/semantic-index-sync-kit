import Foundation

/// Progress of a re-embedding migration toward a new epoch.
public struct MigrationProgress: Sendable, Equatable {
    public let targetEpoch: EmbeddingEpoch
    public let completed: Int
    public let remaining: Int

    public init(targetEpoch: EmbeddingEpoch, completed: Int, remaining: Int) {
        self.targetEpoch = targetEpoch
        self.completed = max(0, completed)
        self.remaining = max(0, remaining)
    }

    public var total: Int { Saturating.add(completed, remaining) }

    public var fraction: Double { Saturating.ratio(completed, total, whenEmpty: 1.0) }

    public var isFinished: Bool { remaining == 0 }
}

/// What one drain pass did.
public enum DrainOutcome: Sendable, Equatable {
    /// Nothing queued.
    case idle
    /// The budget refused this pass. The queue is untouched and resumable.
    case deferred(BudgetRejection)
    /// Work was done and more remains.
    case progressed(MigrationProgress)
    /// The queue drained; the index is wholly in the target epoch.
    case finished(MigrationProgress)
    /// The provider failed. Treated as a deferral, not a loss: the affected
    /// chunks stay queued so a later pass retries them.
    case providerFailed(String)

    public var progress: MigrationProgress? {
        switch self {
        case .progressed(let value), .finished(let value): return value
        case .idle, .deferred, .providerFailed: return nil
        }
    }
}
