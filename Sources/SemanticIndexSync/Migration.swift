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
    /// Another pass already holds every queued chunk. Not an error and not a
    /// budget refusal — reported distinctly so a caller polling in a loop can
    /// tell "someone else is on it" from "nothing to do".
    case alreadyDraining(MigrationProgress)
    /// Work was done and more remains.
    case progressed(MigrationProgress)
    /// The queue drained; the index is wholly in the target epoch.
    case finished(MigrationProgress)
    /// The provider failed. Treated as a deferral, not a loss: the affected
    /// chunks stay queued so a later pass retries them.
    case providerFailed(String)
    /// The active epoch changed while this pass was embedding, so its results
    /// describe a space nobody queries any more and were discarded.
    ///
    /// Distinct from ``deferred(_:)`` on purpose: the budget admitted this pass,
    /// and the queue was rebuilt underneath it rather than left untouched. Both
    /// halves of a budget refusal's contract are false here, so borrowing one
    /// would put a wrong sentence in the log.
    case abandoned(supersededBy: EmbeddingEpoch)

    public var progress: MigrationProgress? {
        switch self {
        case .progressed(let value), .finished(let value), .alreadyDraining(let value):
            return value
        case .idle, .deferred, .providerFailed, .abandoned: return nil
        }
    }
}
