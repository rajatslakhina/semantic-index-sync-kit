import Foundation

/// Which side of a merge won.
public enum MergeWinner: String, Sendable, Equatable {
    case local, remote
}

/// Why a merge resolved the way it did. Every decision is explainable; nothing
/// in this layer resolves silently.
public enum MergeOutcome: Sendable, Equatable {
    /// Both sides are the same point in history. No work.
    case identical
    /// The remote write strictly supersedes the local one.
    case tookRemote
    /// The local write strictly supersedes the remote one; the remote is stale.
    case ignoredStaleRemote
    /// Neither side saw the other. Resolved by an explicit, deterministic rule.
    case resolvedConcurrent(winner: MergeWinner, rule: ConcurrentRule)
}

/// The rule that decided a concurrent merge.
public enum ConcurrentRule: String, Sendable, Equatable, CustomStringConvertible {
    /// A delete beat a concurrent live write.
    case deleteWins
    /// Two concurrent live writes, ordered by content hash.
    case higherContentHash
    /// Two concurrent live writes with the *same* content hash — the edits agree,
    /// so the writer identity breaks the tie and both replicas pick the same one.
    case writerIdentity

    public var description: String { rawValue }
}

/// What one `apply` pass did.
public struct ReconciliationReport: Sendable, Equatable {
    public var inserted: Int = 0
    public var supersededByRemote: Int = 0
    public var staleRemotesIgnored: Int = 0
    public var identical: Int = 0
    public var concurrentResolved: Int = 0
    /// Concurrent merges specifically resolved in favour of a delete. Broken out
    /// because it is the case a reviewer will ask about.
    public var tombstonesUpheld: Int = 0
    /// Documents whose effective content changed, so their chunks' vectors are
    /// no longer valid and must be re-embedded.
    public var invalidated: [DocumentID] = []

    public init() {}

    public var totalConsidered: Int {
        var sum = 0
        for value in [inserted, supersededByRemote, staleRemotesIgnored, identical, concurrentResolved] {
            sum = Saturating.add(sum, value)
        }
        return sum
    }
}

/// Merges remote manifest rows into a local manifest.
///
/// Pure and synchronous on purpose: the ordering rules are the part of this
/// system most likely to be wrong, so they are kept free of actors, I/O and
/// time, where they can be property-tested exhaustively.
public struct Reconciler: Sendable {

    public init() {}

    /// Decide the winner of one document, and say why.
    ///
    /// ## The concurrent case
    ///
    /// When neither write saw the other, *something* has to break the tie, and
    /// the rule must be a pure function of the two records so that every replica
    /// reaches the same answer without talking to anyone. Three rules, in order:
    ///
    /// 1. **Delete wins.** A concurrent edit-vs-delete resolves to the delete.
    ///    This loses an edit, which is a real cost. The alternative loses a
    ///    delete, and in a *searchable index of personal content* that means
    ///    material the user removed becomes findable again on another device.
    ///    Between "your edit needs redoing" and "your deleted content came back",
    ///    only the first is recoverable by the user.
    /// 2. **Higher content hash wins.** Arbitrary but total and stable, so both
    ///    replicas converge. Deliberately *not* "longer content" or "more recent
    ///    timestamp", both of which look smarter and are not deterministic across
    ///    replicas.
    /// 3. **Writer identity.** Only reachable when two devices independently
    ///    produced byte-identical content, where no data is lost either way.
    public func resolve(local: DocumentRecord, remote: DocumentRecord) -> (record: DocumentRecord, outcome: MergeOutcome) {
        switch VersionVector.order(local.version, remote.version) {
        case .identical:
            return (local, .identical)

        case .descendant:
            return (local, .ignoredStaleRemote)

        case .ancestor:
            return (remote, .tookRemote)

        case .concurrent:
            // The merged history has seen both writes, whichever payload wins.
            let joined = local.version.merging(remote.version)

            if local.state.isTombstone != remote.state.isTombstone {
                let winnerIsLocal = local.state.isTombstone
                let chosen = winnerIsLocal ? local : remote
                let merged = DocumentRecord(
                    id: chosen.id,
                    state: .tombstone,
                    version: joined,
                    lastWriter: chosen.lastWriter
                )
                return (merged, .resolvedConcurrent(winner: winnerIsLocal ? .local : .remote, rule: .deleteWins))
            }

            if local.state.isTombstone {
                // Two concurrent deletes: same outcome either way, pick stably.
                let winnerIsLocal = local.lastWriter <= remote.lastWriter
                let chosen = winnerIsLocal ? local : remote
                let merged = DocumentRecord(
                    id: chosen.id,
                    state: .tombstone,
                    version: joined,
                    lastWriter: chosen.lastWriter
                )
                return (merged, .resolvedConcurrent(winner: winnerIsLocal ? .local : .remote, rule: .writerIdentity))
            }

            let localHash = local.state.contentHash
            let remoteHash = remote.state.contentHash
            // Both are `.live`, so both hashes are present. Rather than assert
            // that with a force-unwrap, fall through to a total comparison:
            // a missing hash sorts below any present one.
            if localHash == remoteHash {
                let winnerIsLocal = local.lastWriter <= remote.lastWriter
                let chosen = winnerIsLocal ? local : remote
                let merged = DocumentRecord(
                    id: chosen.id,
                    state: chosen.state,
                    version: joined,
                    lastWriter: chosen.lastWriter
                )
                return (merged, .resolvedConcurrent(winner: winnerIsLocal ? .local : .remote, rule: .writerIdentity))
            }

            let winnerIsLocal = Self.orderedAfter(localHash, remoteHash)
            let chosen = winnerIsLocal ? local : remote
            let merged = DocumentRecord(
                id: chosen.id,
                state: chosen.state,
                version: joined,
                lastWriter: chosen.lastWriter
            )
            return (merged, .resolvedConcurrent(winner: winnerIsLocal ? .local : .remote, rule: .higherContentHash))
        }
    }

    /// Total order over optional hashes: present beats absent, then lexicographic.
    private static func orderedAfter(_ lhs: ContentHash?, _ rhs: ContentHash?) -> Bool {
        switch (lhs, rhs) {
        case (.some(let l), .some(let r)): return l > r
        case (.some, .none): return true
        case (.none, .some): return false
        case (.none, .none): return false
        }
    }

    /// Merge a batch of remote rows into a local manifest, reporting what changed.
    public func apply(
        remote: [DocumentRecord],
        into manifest: inout [DocumentID: DocumentRecord]
    ) -> ReconciliationReport {
        var report = ReconciliationReport()

        for incoming in remote {
            guard let existing = manifest[incoming.id] else {
                manifest[incoming.id] = incoming
                report.inserted = Saturating.add(report.inserted, 1)
                if incoming.state.contentHash != nil {
                    report.invalidated.append(incoming.id)
                }
                continue
            }

            let previousHash = existing.state.contentHash
            let (winner, outcome) = resolve(local: existing, remote: incoming)
            manifest[incoming.id] = winner

            switch outcome {
            case .identical:
                report.identical = Saturating.add(report.identical, 1)
            case .tookRemote:
                report.supersededByRemote = Saturating.add(report.supersededByRemote, 1)
            case .ignoredStaleRemote:
                report.staleRemotesIgnored = Saturating.add(report.staleRemotesIgnored, 1)
            case .resolvedConcurrent(_, let rule):
                report.concurrentResolved = Saturating.add(report.concurrentResolved, 1)
                if rule == .deleteWins {
                    report.tombstonesUpheld = Saturating.add(report.tombstonesUpheld, 1)
                }
            }

            if winner.state.contentHash != previousHash {
                report.invalidated.append(incoming.id)
            }
        }

        report.invalidated.sort()
        return report
    }
}
