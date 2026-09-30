import Foundation

/// The single owner of index state: manifest, chunks, vectors and the migration
/// queue.
///
/// ## Why one actor rather than a lock per table
///
/// Every interesting invariant here spans two tables at once — "no vector
/// survives its document's tombstone", "the re-index queue holds exactly the
/// chunks that are stale". Splitting the state means those invariants hold only
/// between lock acquisitions, and the bugs that produces are the ones that
/// reproduce once a week on a user's device. One actor makes them hold at every
/// suspension point, and the cost is that embedding — the genuinely slow part —
/// has to be handled carefully, which is what ``drainMigration(budget:conditions:)``
/// does below.
///
/// ## Reentrancy
///
/// `drainMigration` awaits the provider, and the actor is reentrant across that
/// `await`: a sync, a delete or a second drain can interleave. Rather than
/// assume otherwise, the method re-validates every embedded chunk against
/// current state *after* the await and discards results whose chunk has since
/// been deleted, edited, or overtaken by another epoch bump. Nothing is written
/// back on the strength of a pre-await snapshot.
public actor IndexCoordinator {

    /// Identity of a stored vector: which chunk, in which embedding space.
    struct VectorKey: Hashable, Sendable {
        let chunk: ChunkID
        let epoch: EmbeddingEpoch
    }

    // MARK: Stored state

    private let device: DeviceID
    private let reconciler = Reconciler()
    private let lexical: LexicalIndex

    private var manifest: [DocumentID: DocumentRecord] = [:]
    private var chunks: [ChunkID: Chunk] = [:]

    /// Keyed by chunk **and** epoch, not by chunk alone.
    ///
    /// Keying by chunk alone would make a new vector overwrite the previous
    /// epoch's, which quietly turns the retention policy documented on
    /// ``adopt(provider:)`` into a lie: a rollback would find nothing and
    /// trigger a second full re-index. The extra key component is what makes
    /// "keep the old space until the new one has settled" an actual behaviour
    /// rather than a comment.
    private var vectors: [VectorKey: StoredVector] = [:]

    private var provider: any EmbeddingProvider
    /// The epoch every *new* vector is produced in and every query is embedded in.
    private var activeEpoch: EmbeddingEpoch

    /// Chunks known to lack a fresh vector in ``activeEpoch``. Insertion-ordered
    /// so a migration makes visible progress front-to-back instead of thrashing
    /// a random subset — and so progress is reproducible in tests.
    private var pending: [ChunkID] = []
    private var pendingSet: Set<ChunkID> = []

    /// Chunks a drain pass has handed to the provider and is awaiting, mapped to
    /// the token of the pass that owns each claim.
    ///
    /// Without this, two concurrent drains take the same prefix of `pending`,
    /// embed the same chunks twice, and both burn the battery budget
    /// ``WorkBudget`` exists to conserve.
    ///
    /// The *token* matters as much as the set. Releasing by chunk id alone lets a
    /// resuming pass free a claim it no longer owns: an edit mid-flight drops the
    /// first pass's claim and re-queues the chunk, a second pass legitimately
    /// claims it, and then the first pass wakes up and deletes the second's
    /// claim — after which a third pass takes the same chunk and embeds it again.
    /// A pass may only release what it still holds.
    private var inFlight: [ChunkID: UInt64] = [:]

    /// Monotonic id for drain passes. Saturates rather than wrapping, so two
    /// passes can never share a token.
    private var nextPassToken: UInt64 = 0

    // MARK: Init

    public init(device: DeviceID, provider: any EmbeddingProvider, lexical: LexicalIndex = LexicalIndex()) {
        self.device = device
        self.provider = provider
        self.activeEpoch = provider.epoch
        self.lexical = lexical
    }

    // MARK: Introspection

    public var currentEpoch: EmbeddingEpoch { activeEpoch }

    public var records: [DocumentRecord] {
        manifest.values.sorted { $0.id < $1.id }
    }

    public var liveChunks: [Chunk] {
        chunks.values
            .filter { isLive($0.id.document) }
            .sorted { $0.id < $1.id }
    }

    /// Progress toward having the whole live corpus in the active epoch.
    ///
    /// **Derived, not accumulated.** A running counter of "chunks embedded since
    /// the last `adopt`" drifts away from the index the moment a chunk is
    /// deleted or re-written after being embedded: the count keeps the work, the
    /// corpus loses the chunk, and `total` grows past the number of passages
    /// that exist — which is exactly the fabricated denominator a progress bar
    /// would then render. Computing coverage from current state instead makes
    /// `completed + remaining == liveChunks.count` true by construction, and
    /// removes a whole class of accounting bug rather than guarding against it.
    public var migrationProgress: MigrationProgress {
        var covered = 0
        for chunk in liveChunks {
            let key = VectorKey(chunk: chunk.id, epoch: activeEpoch)
            if let vector = vectors[key], vector.isFresh(for: chunk, in: activeEpoch) {
                covered = Saturating.add(covered, 1)
            }
        }
        return MigrationProgress(
            targetEpoch: activeEpoch,
            completed: covered,
            remaining: pending.count
        )
    }

    /// Vectors currently held, grouped by the epoch that produced them. This is
    /// the view that makes a mixed-epoch index visible in the demo.
    public var vectorCountsByEpoch: [EmbeddingEpoch: Int] {
        var counts: [EmbeddingEpoch: Int] = [:]
        for vector in vectors.values {
            counts[vector.epoch] = Saturating.add(counts[vector.epoch] ?? 0, 1)
        }
        return counts
    }

    private func isLive(_ document: DocumentID) -> Bool {
        guard let record = manifest[document] else { return false }
        return !record.state.isTombstone
    }

    /// Cross-table invariant audit.
    ///
    /// Internal rather than public: it is a statement of what this actor promises
    /// about its own state, checked by the concurrency suite after arbitrary
    /// interleavings. Returning the violations rather than asserting keeps it
    /// usable from a test without `fatalError` in shipping code.
    internal func invariantViolations() -> [String] {
        var problems: [String] = []

        for key in vectors.keys {
            let id = key.chunk
            if chunks[id] == nil { problems.append("vector without chunk: \(id)") }
            else if !isLive(id.document) { problems.append("vector survived tombstone: \(id)") }
        }
        // The invariant the one-actor design is justified by: the queue holds
        // exactly the live chunks that lack a usable vector in the active epoch.
        // Checking only the other direction would miss a stranded chunk — one
        // with no vector and no queue entry, permanently unsearchable.
        for chunk in chunks.values where isLive(chunk.id.document) {
            let key = VectorKey(chunk: chunk.id, epoch: activeEpoch)
            let covered = vectors[key]?.isFresh(for: chunk, in: activeEpoch) ?? false
            if !covered && !pendingSet.contains(chunk.id) {
                problems.append("live chunk is neither covered nor queued: \(chunk.id)")
            }
            if covered && pendingSet.contains(chunk.id) {
                problems.append("covered chunk is still queued: \(chunk.id)")
            }
        }
        if !Set(inFlight.keys).isSubset(of: pendingSet) {
            problems.append("in-flight chunks are not all queued")
        }
        for id in pending {
            if chunks[id] == nil { problems.append("queued chunk no longer exists: \(id)") }
            else if !isLive(id.document) { problems.append("queued chunk is tombstoned: \(id)") }
        }
        if Set(pending).count != pending.count { problems.append("queue contains duplicates") }
        if Set(pending) != pendingSet { problems.append("queue and membership set diverged") }
        for chunk in chunks.values where !isLive(chunk.id.document) {
            problems.append("chunk survived tombstone: \(chunk.id)")
        }
        return problems
    }

    // MARK: Local authoring

    /// Write a document locally: replaces its chunks and queues them for embedding.
    @discardableResult
    public func upsert(
        document: DocumentID,
        passages: [String],
        hash: ContentHash
    ) -> DocumentRecord {
        let existing = manifest[document]
        let record: DocumentRecord
        if let existing, !existing.state.isTombstone {
            record = existing.edited(to: hash, by: device)
        } else if let existing {
            // Re-creating a deleted document. The new write must dominate the
            // tombstone, so it builds on the tombstone's version rather than
            // starting fresh — otherwise it would look concurrent with its own
            // delete and lose to the delete-wins rule.
            record = DocumentRecord(
                id: document,
                state: .live(hash),
                version: existing.version.incrementing(device),
                lastWriter: device
            )
        } else {
            record = DocumentRecord.live(document, hash: hash, by: device)
        }
        manifest[document] = record
        replaceChunks(of: document, passages: passages, hash: hash)
        return record
    }

    /// Delete a document locally: tombstone the record and drop its vectors.
    @discardableResult
    public func delete(document: DocumentID) -> DocumentRecord? {
        guard let existing = manifest[document] else { return nil }
        let tombstone = existing.deleted(by: device)
        manifest[document] = tombstone
        purgeChunks(of: document)
        return tombstone
    }

    private func replaceChunks(of document: DocumentID, passages: [String], hash: ContentHash) {
        purgeChunks(of: document)
        for (ordinal, text) in passages.enumerated() {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            let id = ChunkID(document: document, ordinal: ordinal)
            chunks[id] = Chunk(id: id, text: trimmed, sourceHash: hash)
            enqueue(id)
        }
    }

    /// Removes a document's chunks, vectors and queue entries together.
    ///
    /// These three always move as a unit. A vector left behind after a tombstone
    /// is the concrete shape of "deleted content is still searchable", so the
    /// only place that removes a chunk is here.
    private func purgeChunks(of document: DocumentID) {
        let owned = chunks.keys.filter { $0.document == document }
        for id in owned {
            chunks.removeValue(forKey: id)
            // Every epoch's vector for this chunk, not just the active one.
            for key in vectors.keys where key.chunk == id {
                vectors.removeValue(forKey: key)
            }
            pendingSet.remove(id)
            inFlight.removeValue(forKey: id)
        }
        if !owned.isEmpty {
            let removed = Set(owned)
            pending.removeAll { removed.contains($0) }
        }
    }

    private func enqueue(_ id: ChunkID) {
        guard !pendingSet.contains(id) else { return }
        pendingSet.insert(id)
        pending.append(id)
    }

    // MARK: Sync

    /// Merge a peer's manifest rows and re-queue anything whose content changed.
    ///
    /// The peer sends `passages` for documents whose content this replica does
    /// not yet have. Vectors are **not** accepted from peers — see
    /// ``IndexCoordinator`` docs and the README's "What does not sync" section.
    @discardableResult
    public func applyRemote(
        records incoming: [DocumentRecord],
        passages: [DocumentID: [String]] = [:]
    ) -> ReconciliationReport {
        let report = reconciler.apply(remote: incoming, into: &manifest)

        for document in report.invalidated {
            guard let record = manifest[document] else { continue }
            guard case .live(let hash) = record.state else {
                purgeChunks(of: document)
                continue
            }
            if let text = passages[document] {
                replaceChunks(of: document, passages: text, hash: hash)
            } else {
                // Content not shipped with this batch: the old chunks are known
                // stale, so they are dropped rather than left to serve results
                // from superseded text.
                purgeChunks(of: document)
            }
        }

        // A remote tombstone that won outside the `invalidated` set still has to
        // take its vectors with it.
        for record in incoming {
            if manifest[record.id]?.state.isTombstone == true {
                purgeChunks(of: record.id)
            }
        }

        return report
    }

    /// The rows this replica would send to a peer.
    public func exportManifest() -> [DocumentRecord] { records }

    // MARK: Epoch migration

    /// Point the index at a new embedding space.
    ///
    /// Vectors from the previous epoch are **kept**, not deleted. Deleting them
    /// would be simpler and is the wrong call: it would drop the index to zero
    /// semantic coverage the instant an OS update lands, for however long the
    /// re-embed takes. Keeping them costs disk and buys nothing for *scoring* —
    /// they are never compared across epochs — but it means a rollback to the
    /// previous model (a staged OS rollout being pulled, the user disabling and
    /// re-enabling the feature) restores full coverage instantly instead of
    /// triggering a second full re-index.
    ///
    /// The retention cost is bounded by ``discardVectors(outside:)``, which the
    /// app calls once a migration has finished and the new epoch has proven stable.
    @discardableResult
    public func adopt(provider newProvider: any EmbeddingProvider) -> MigrationProgress {
        provider = newProvider
        let newEpoch = newProvider.epoch
        guard newEpoch != activeEpoch else { return migrationProgress }

        activeEpoch = newEpoch
        pending.removeAll(keepingCapacity: true)
        pendingSet.removeAll(keepingCapacity: true)
        // Any pass still suspended in the provider is now working for a space
        // nobody queries; it will discard its results on resume. Releasing its
        // claim here keeps `inFlight` a subset of the queue — otherwise a later
        // pass would skip a chunk that nothing is actually working on and report
        // `.alreadyDraining` about a pass that no longer exists.
        inFlight.removeAll(keepingCapacity: true)

        for chunk in liveChunks {
            let key = VectorKey(chunk: chunk.id, epoch: newEpoch)
            let covered = vectors[key]?.isFresh(for: chunk, in: newEpoch) ?? false
            if !covered { enqueue(chunk.id) }
        }
        return migrationProgress
    }

    /// Drop vectors that belong to no listed epoch. Called after a migration has
    /// settled, to reclaim the disk that ``adopt(provider:)`` deliberately spends.
    ///
    /// Any live chunk left without a usable vector in the active epoch is
    /// re-queued here. Skipping that step would strand the chunk: unsearchable
    /// by meaning, absent from the re-index queue, and therefore never
    /// recoverable — an index permanently stuck below full coverage while every
    /// internal check still reports it healthy.
    @discardableResult
    public func discardVectors(outside keep: Set<EmbeddingEpoch>) -> Int {
        let doomed = vectors.keys.filter { !keep.contains($0.epoch) }
        for key in doomed { vectors.removeValue(forKey: key) }
        requeueUncoveredChunks()
        return doomed.count
    }

    /// Re-queues every live chunk that lacks a fresh vector in the active epoch.
    private func requeueUncoveredChunks() {
        for chunk in liveChunks {
            let key = VectorKey(chunk: chunk.id, epoch: activeEpoch)
            if let vector = vectors[key], vector.isFresh(for: chunk, in: activeEpoch) { continue }
            enqueue(chunk.id)
        }
    }

    /// Do at most one budgeted pass of re-embedding.
    public func drainMigration(
        budget: WorkBudget,
        conditions: DeviceConditions = DeviceConditions()
    ) async -> DrainOutcome {
        guard !pending.isEmpty else { return .idle }
        if let rejection = budget.admits(conditions) { return .deferred(rejection) }
        guard provider.isAvailable else { return .providerFailed("provider unavailable") }

        let epochAtStart = activeEpoch
        guard budget.maxChunksPerPass > 0 else { return .deferred(.zeroAllowance) }

        // Skip chunks another pass is already embedding. Taking the same prefix
        // twice is the concurrency bug this guard exists to prevent.
        let batch = availableBatch(limit: budget.maxChunksPerPass)
        guard !batch.isEmpty else { return .alreadyDraining(migrationProgress) }

        // Snapshot the exact text each vector will describe, so a post-await
        // comparison can tell "still the same text" from "edited underneath us".
        var snapshot: [(id: ChunkID, chunk: Chunk)] = []
        snapshot.reserveCapacity(batch.count)
        for id in batch {
            guard let chunk = chunks[id], isLive(id.document) else { continue }
            snapshot.append((id, chunk))
        }
        guard !snapshot.isEmpty else {
            // Every queued chunk in this slice is gone. Drop them and report
            // progress rather than spinning on a queue of ghosts.
            dequeue(batch)
            return pending.isEmpty
                ? .finished(migrationProgress)
                : .progressed(migrationProgress)
        }

        // Claim them before suspending, under this pass's own token.
        let passToken = nextPassToken
        nextPassToken = nextPassToken == UInt64.max ? UInt64.max : nextPassToken &+ 1
        for entry in snapshot { inFlight[entry.id] = passToken }

        let embedded: [[Double]]
        do {
            embedded = try await provider.embed(snapshot.map(\.chunk.text))
        } catch {
            // Released and left queued on purpose: a provider failure is
            // transient, and silently dropping the chunks would leave them
            // permanently unsearchable with no record of why.
            releaseClaims(of: snapshot.map(\.id), heldBy: passToken)
            return .providerFailed(String(describing: error))
        }

        // ---- Everything below re-reads current state. The snapshot above is
        // ---- only used to detect what changed across the suspension point.
        releaseClaims(of: snapshot.map(\.id), heldBy: passToken)

        guard activeEpoch == epochAtStart else {
            // A second epoch bump landed while we were embedding. These vectors
            // describe a space nobody queries any more, so they are discarded —
            // but this is emphatically NOT a budget refusal. Reporting it as one
            // would put "budget allows zero chunks" in a log line about a pass
            // the budget admitted, and tell the reader the queue was untouched
            // when `adopt` has just rebuilt it. It gets its own case.
            return .abandoned(supersededBy: activeEpoch)
        }

        guard embedded.count == snapshot.count else {
            return .providerFailed(
                "provider returned \(embedded.count) vectors for \(snapshot.count) inputs"
            )
        }

        var satisfied: Set<ChunkID> = []
        for (offset, entry) in snapshot.enumerated() {
            guard embedded.indices.contains(offset) else { continue }
            let values = embedded[offset]
            guard values.count == epochAtStart.dimension else { continue }
            // Re-validate against *current* state, not the snapshot.
            guard let current = chunks[entry.id], isLive(entry.id.document) else { continue }
            guard current.sourceHash == entry.chunk.sourceHash else {
                // Edited across the await. The vector describes text that is no
                // longer there; leave the chunk queued for the next pass.
                continue
            }
            guard pendingSet.contains(entry.id) else {
                // Already satisfied by another pass while we were suspended.
                continue
            }
            vectors[VectorKey(chunk: entry.id, epoch: epochAtStart)] = StoredVector(
                chunk: entry.id,
                epoch: epochAtStart,
                sourceHash: current.sourceHash,
                values: values
            )
            pendingSet.remove(entry.id)
            satisfied.insert(entry.id)
        }

        // One pass over the queue instead of one per embedded chunk: the inner
        // `removeAll` made a budgeted pass O(batch x queue), which at the corpus
        // size this package targets is 128 x 50_000 array scans — in the one
        // code path whose entire justification is spending as little of the
        // user's device as possible.
        if !satisfied.isEmpty {
            pending.removeAll { satisfied.contains($0) }
        }

        // Chunks from this slice that vanished entirely (deleted mid-flight) are
        // dropped from the queue; ones that were merely edited stay queued.
        let survivors = Set(snapshot.map(\.id))
        let vanished = batch.filter { !survivors.contains($0) }
        dequeue(vanished)

        return pending.isEmpty ? .finished(migrationProgress) : .progressed(migrationProgress)
    }

    /// The queued chunks no other pass is already embedding.
    ///
    /// Factored out of ``drainMigration(budget:conditions:)`` so the guard is
    /// directly assertable while a pass is parked in the provider. Testing it
    /// only through a second `drainMigration` call would mean that removing the
    /// guard makes the test *hang* rather than fail — and a test that hangs
    /// under a regression is not a test that catches it.
    internal func availableBatch(limit: Int) -> [ChunkID] {
        guard limit > 0 else { return [] }
        var batch: [ChunkID] = []
        batch.reserveCapacity(min(limit, pending.count))
        for id in pending where inFlight[id] == nil {
            batch.append(id)
            if batch.count == limit { break }
        }
        return batch
    }

    /// Releases only the claims this pass still owns.
    private func releaseClaims(of ids: [ChunkID], heldBy token: UInt64) {
        for id in ids where inFlight[id] == token {
            inFlight.removeValue(forKey: id)
        }
    }

    private func dequeue(_ ids: [ChunkID]) {
        guard !ids.isEmpty else { return }
        let set = Set(ids)
        for id in set {
            pendingSet.remove(id)
            inFlight.removeValue(forKey: id)
        }
        pending.removeAll { set.contains($0) }
    }

    // MARK: Query

    /// Search the index, reporting how much of it the answer actually covers.
    public func search(_ text: String, limit: Int = 10) async -> QueryResult {
        let corpus = liveChunks
        let cappedLimit = max(0, min(limit, corpus.count))

        guard provider.isAvailable else {
            let lexicalScores = lexical.scores(for: text, over: corpus)
            let completeness = Completeness(
                liveChunks: corpus.count,
                semanticallyCovered: 0,
                awaitingReindex: corpus.count,
                isLexicalOnly: true
            )
            let hits = Self.rank(
                corpus: corpus,
                semantic: [:],
                lexical: lexicalScores,
                limit: cappedLimit
            )
            return QueryResult(hits: hits, completeness: completeness, epoch: nil)
        }

        let epoch = activeEpoch
        let queryVector: [Double]
        do {
            let embedded = try await provider.embed([text])
            guard let first = embedded.first, first.count == epoch.dimension else {
                throw EmbeddingError.dimensionMismatch(
                    expected: epoch.dimension,
                    actual: embedded.first?.count ?? 0
                )
            }
            queryVector = first
        } catch {
            // Re-read: the corpus may have moved while the query was embedding,
            // so the scores must be computed over what is live *now*.
            let fallbackCorpus = liveChunks
            let completeness = Completeness(
                liveChunks: fallbackCorpus.count,
                semanticallyCovered: 0,
                awaitingReindex: fallbackCorpus.count,
                isLexicalOnly: true
            )
            let hits = Self.rank(
                corpus: fallbackCorpus,
                semantic: [:],
                lexical: lexical.scores(for: text, over: fallbackCorpus),
                limit: max(0, min(limit, fallbackCorpus.count))
            )
            return QueryResult(hits: hits, completeness: completeness, epoch: nil)
        }

        // Re-read the corpus after the await: the set of live chunks may have
        // moved while the query was being embedded.
        let currentCorpus = liveChunks
        var semanticScores: [ChunkID: Double] = [:]
        var covered = 0

        for chunk in currentCorpus {
            let key = VectorKey(chunk: chunk.id, epoch: epoch)
            guard let vector = vectors[key], vector.isFresh(for: chunk, in: epoch) else {
                // The epoch gate. A vector from another space is *not* scored at
                // a lower weight or scaled — it is not comparable at all, and
                // scoring it would return a confident, meaningless number.
                continue
            }
            covered = Saturating.add(covered, 1)
            let similarity = VectorMath.cosineSimilarity(queryVector, vector.values)
            if similarity > 0 { semanticScores[chunk.id] = similarity }
        }

        // Always recomputed over the corpus as it is *now*. Reusing scores
        // computed before the suspension point on the strength of an unchanged
        // element count would be wrong: a delete and an upsert that interleave
        // leave the count identical and the membership different, and the
        // min-max normalisation would then be taken over a range that partly
        // belongs to chunks which are no longer there.
        let currentLexical = lexical.scores(for: text, over: currentCorpus)

        let completeness = Completeness(
            liveChunks: currentCorpus.count,
            semanticallyCovered: covered,
            awaitingReindex: Saturating.subtract(currentCorpus.count, covered),
            isLexicalOnly: false
        )
        let hits = Self.rank(
            corpus: currentCorpus,
            semantic: semanticScores,
            lexical: currentLexical,
            limit: max(0, min(limit, currentCorpus.count))
        )
        return QueryResult(hits: hits, completeness: completeness, epoch: epoch)
    }

    /// Fuse the two score spaces and take the top `limit`.
    ///
    /// BM25 is unbounded and cosine is in `[-1, 1]`, so the two are min-max
    /// normalised within this query before being combined — comparing raw values
    /// would let one long document's BM25 score dominate every semantic hit.
    /// Normalising per query (rather than against a global maximum) keeps the
    /// fusion stable as the corpus changes underneath it.
    static func rank(
        corpus: [Chunk],
        semantic: [ChunkID: Double],
        lexical: [ChunkID: Double],
        limit: Int
    ) -> [ScoredChunk] {
        guard limit > 0, !corpus.isEmpty else { return [] }

        let semanticWeight = 0.65
        let lexicalWeight = 0.35
        let normalizedLexical = normalize(lexical)
        let normalizedSemantic = normalize(semantic)

        var scored: [ScoredChunk] = []
        scored.reserveCapacity(corpus.count)

        for chunk in corpus {
            let semanticValue = normalizedSemantic[chunk.id]
            let lexicalValue = normalizedLexical[chunk.id]
            guard semanticValue != nil || lexicalValue != nil else { continue }

            let source: ScoreSource
            switch (semanticValue, lexicalValue) {
            case (.some, .some): source = .hybrid
            case (.some, .none): source = .semantic
            default: source = .lexicalFallback
            }

            let combined = semanticWeight * (semanticValue ?? 0) + lexicalWeight * (lexicalValue ?? 0)
            scored.append(ScoredChunk(chunk: chunk, score: combined, source: source))
        }

        // Ties broken by chunk id so ranking is a total order and snapshot tests
        // are not flaky.
        scored.sort {
            if $0.score != $1.score { return $0.score > $1.score }
            return $0.chunk.id < $1.chunk.id
        }
        return Array(scored.prefix(limit))
    }

    private static func normalize(_ scores: [ChunkID: Double]) -> [ChunkID: Double] {
        guard !scores.isEmpty else { return [:] }
        let values = scores.values.filter { $0.isFinite }
        guard let lowest = values.min(), let highest = values.max() else { return [:] }
        let span = highest - lowest
        guard span > 0, span.isFinite else {
            // Every score identical: they are all equally relevant, not all zero.
            return scores.mapValues { _ in 1.0 }
        }
        return scores.mapValues { value in
            guard value.isFinite else { return 0 }
            return (value - lowest) / span
        }
    }
}
