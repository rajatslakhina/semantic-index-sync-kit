import Foundation
@testable import SemanticIndexSync

/// A rendezvous with no sleeps.
///
/// Concurrency tests that wait with `Task.sleep` are timing-dependent by
/// construction and flake on a loaded CI runner. This gate suspends until the
/// other side has genuinely arrived, so the interleaving under test is the one
/// the test asked for, every run.
actor Gate {
    private var hasEntered = false
    private var isReleased = false
    private var entryWaiters: [CheckedContinuation<Void, Never>] = []
    private var releaseWaiters: [CheckedContinuation<Void, Never>] = []

    func markEntered() {
        hasEntered = true
        let waiters = entryWaiters
        entryWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
    }

    func waitUntilEntered() async {
        if hasEntered { return }
        await withCheckedContinuation { entryWaiters.append($0) }
    }

    func release() {
        isReleased = true
        let waiters = releaseWaiters
        releaseWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
    }

    func waitForRelease() async {
        if isReleased { return }
        await withCheckedContinuation { releaseWaiters.append($0) }
    }
}

/// A provider whose `embed` parks inside the suspension point, so a test can
/// mutate the index while a drain is genuinely mid-flight.
struct GatedProvider: EmbeddingProvider {
    let epoch: EmbeddingEpoch
    let isAvailable: Bool
    let gate: Gate

    init(epoch: EmbeddingEpoch, isAvailable: Bool = true, gate: Gate) {
        self.epoch = epoch
        self.isAvailable = isAvailable
        self.gate = gate
    }

    func embed(_ texts: [String]) async throws -> [[Double]] {
        await gate.markEntered()
        await gate.waitForRelease()
        let base = DeterministicEmbeddingProvider(epoch: epoch)
        return texts.map { base.vector(for: $0) }
    }
}

/// Counts calls and decides which of them fail. An actor rather than a lock,
/// because `NSLock` is unavailable from an async context under strict concurrency
/// and reaching for `@unchecked Sendable` to dodge that is how a test suite grows
/// its own data race.
actor FailureBudget {
    private var remainingFailures: Int
    private var calls = 0

    init(failures: Int) { remainingFailures = max(0, failures) }

    var callCount: Int { calls }

    func recordCallAndDecide() -> Bool {
        calls += 1
        guard remainingFailures > 0 else { return false }
        remainingFailures -= 1
        return true
    }
}

/// Fails a configurable number of times before succeeding.
struct FlakyProvider: EmbeddingProvider {
    let epoch: EmbeddingEpoch
    let isAvailable: Bool
    let budget: FailureBudget

    init(epoch: EmbeddingEpoch, failures: Int, isAvailable: Bool = true) {
        self.epoch = epoch
        self.isAvailable = isAvailable
        self.budget = FailureBudget(failures: failures)
    }

    func embed(_ texts: [String]) async throws -> [[Double]] {
        let shouldFail = await budget.recordCallAndDecide()
        guard isAvailable else { throw EmbeddingError.providerUnavailable }
        if shouldFail { throw EmbeddingError.providerUnavailable }
        let base = DeterministicEmbeddingProvider(epoch: epoch)
        return texts.map { base.vector(for: $0) }
    }
}

/// Returns the wrong number of vectors, to prove the coordinator checks.
struct MiscountingProvider: EmbeddingProvider {
    let epoch: EmbeddingEpoch
    var isAvailable: Bool { true }

    func embed(_ texts: [String]) async throws -> [[Double]] {
        let base = DeterministicEmbeddingProvider(epoch: epoch)
        return texts.dropLast().map { base.vector(for: $0) }
    }
}
