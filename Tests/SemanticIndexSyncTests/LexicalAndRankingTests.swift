import XCTest
@testable import SemanticIndexSync

final class LexicalIndexTests: XCTestCase {

    private func chunk(_ ordinal: Int, _ text: String) -> Chunk {
        Chunk(
            id: ChunkID(document: DocumentID("d"), ordinal: ordinal),
            text: text,
            sourceHash: ContentHash("h")
        )
    }

    func testEmptyQueryAndEmptyCorpusReturnNothing() {
        let index = LexicalIndex()
        XCTAssertTrue(index.scores(for: "", over: [chunk(0, "alpha")]).isEmpty)
        XCTAssertTrue(index.scores(for: "alpha", over: []).isEmpty)
        XCTAssertTrue(index.scores(for: "!!! ???", over: [chunk(0, "alpha")]).isEmpty)
    }

    /// An unsmoothed Okapi IDF goes negative for a term present in every
    /// document, which silently inverts ranking.
    ///
    /// The unsmoothed formula is computed inline first and asserted to be
    /// negative, so the test demonstrates that the trap is real before asserting
    /// the implementation avoids it. `LexicalIndex` only stores entries scoring
    /// above zero, so an unsmoothed implementation would return an *empty*
    /// result here — which is exactly what the non-empty assertion catches.
    func testUnsmoothedIDFWouldInvertRankingAndTheSmoothedOneDoesNot() {
        let count = 5
        let corpus = (0..<count).map { chunk($0, "budget passage number \($0)") }

        // The bug, demonstrated: df == N, so the unsmoothed log argument is
        // below 1 and the IDF is negative.
        let documentFrequency = Double(count)
        let unsmoothedArgument = (Double(count) - documentFrequency + 0.5) / (documentFrequency + 0.5)
        XCTAssertLessThan(unsmoothedArgument, 1.0)
        XCTAssertLessThan(
            Foundation.log(unsmoothedArgument), 0.0,
            "The unsmoothed IDF must be negative, or this test does not demonstrate the trap."
        )

        let scores = LexicalIndex().scores(for: "budget", over: corpus)
        XCTAssertEqual(
            scores.count, count,
            "Every chunk contains the term; a negative IDF would drop them all."
        )
        for (id, score) in scores {
            XCTAssertGreaterThan(score, 0, "Non-positive score for \(id)")
            XCTAssertTrue(score.isFinite)
        }
    }

    func testRareTermOutranksCommonTerm() {
        let corpus = [
            chunk(0, "common common common"),
            chunk(1, "common common common"),
            chunk(2, "common rare"),
        ]
        let rareScores = LexicalIndex().scores(for: "rare", over: corpus)
        let commonScores = LexicalIndex().scores(for: "common", over: corpus)
        guard let rare = rareScores[ChunkID(document: DocumentID("d"), ordinal: 2)],
              let common = commonScores[ChunkID(document: DocumentID("d"), ordinal: 2)] else {
            return XCTFail("Expected both terms to score.")
        }
        XCTAssertGreaterThan(rare, common)
    }

    func testChunksWithoutAnyQueryTermAreOmittedNotZeroed() {
        let corpus = [chunk(0, "alpha beta"), chunk(1, "gamma delta")]
        let scores = LexicalIndex().scores(for: "alpha", over: corpus)
        XCTAssertEqual(scores.count, 1)
        XCTAssertNotNil(scores[ChunkID(document: DocumentID("d"), ordinal: 0)])
    }

    func testDegenerateParametersAreClamped() {
        let index = LexicalIndex(k1: .nan, b: 12)
        let scores = index.scores(for: "alpha", over: [chunk(0, "alpha beta")])
        XCTAssertEqual(scores.count, 1)
        for score in scores.values { XCTAssertTrue(score.isFinite) }
    }
}

final class RankFusionTests: XCTestCase {

    private func chunk(_ ordinal: Int) -> Chunk {
        Chunk(
            id: ChunkID(document: DocumentID("d"), ordinal: ordinal),
            text: "passage \(ordinal)",
            sourceHash: ContentHash("h")
        )
    }

    func testEmptyLimitOrCorpusReturnsNoHits() {
        XCTAssertTrue(IndexCoordinator.rank(corpus: [chunk(0)], semantic: [:], lexical: [:], limit: 0).isEmpty)
        XCTAssertTrue(IndexCoordinator.rank(corpus: [], semantic: [:], lexical: [:], limit: 5).isEmpty)
    }

    func testChunksWithNoScoreInEitherSpaceAreDropped() {
        let corpus = [chunk(0), chunk(1)]
        let hits = IndexCoordinator.rank(
            corpus: corpus,
            semantic: [corpus[0].id: 0.9],
            lexical: [:],
            limit: 5
        )
        XCTAssertEqual(hits.count, 1)
        XCTAssertEqual(hits.first?.source, .semantic)
    }

    func testSourceLabellingDistinguishesTheThreeCases() {
        let corpus = [chunk(0), chunk(1), chunk(2)]
        let hits = IndexCoordinator.rank(
            corpus: corpus,
            semantic: [corpus[0].id: 0.9, corpus[2].id: 0.4],
            lexical: [corpus[1].id: 2.0, corpus[2].id: 1.0],
            limit: 5
        )
        let sources = Dictionary(uniqueKeysWithValues: hits.map { ($0.chunk.id, $0.source) })
        XCTAssertEqual(sources[corpus[0].id], .semantic)
        XCTAssertEqual(sources[corpus[1].id], .lexicalFallback)
        XCTAssertEqual(sources[corpus[2].id], .hybrid)
    }

    /// BM25 is unbounded and cosine is capped at 1. Without normalisation a
    /// single long document's keyword score swamps every semantic hit; this
    /// asserts it does not.
    func testUnboundedLexicalScoreDoesNotSwampSemanticRanking() {
        let corpus = [chunk(0), chunk(1)]
        let hits = IndexCoordinator.rank(
            corpus: corpus,
            semantic: [corpus[0].id: 0.95, corpus[1].id: 0.10],
            lexical: [corpus[0].id: 1.0, corpus[1].id: 900.0],
            limit: 5
        )
        // Semantic carries 0.65 of the weight, lexical 0.35, both normalised to
        // [0,1] within the query: chunk 0 scores 0.65, chunk 1 scores 0.35.
        XCTAssertEqual(hits.first?.chunk.id, corpus[0].id)
        XCTAssertEqual(hits.first?.score ?? 0, 0.65, accuracy: 1e-9)
    }

    func testIdenticalScoresAreTreatedAsEquallyRelevantNotZero() {
        let corpus = [chunk(0), chunk(1)]
        let hits = IndexCoordinator.rank(
            corpus: corpus,
            semantic: [:],
            lexical: [corpus[0].id: 3.0, corpus[1].id: 3.0],
            limit: 5
        )
        XCTAssertEqual(hits.count, 2)
        for hit in hits { XCTAssertEqual(hit.score, 0.35, accuracy: 1e-9) }
        // Ties break on chunk id so ranking is a total order, not luck of the draw.
        XCTAssertEqual(hits.map(\.chunk.id), [corpus[0].id, corpus[1].id])
    }

    /// A NaN score is treated as *no* score: the chunk it belonged to is
    /// dropped, and the chunk that scored legitimately is unaffected.
    ///
    /// Asserting only `isFinite` on the results would pass against an
    /// implementation that returned nothing at all, so exact membership, count
    /// and value are asserted instead.
    func testNonFiniteScoreIsTreatedAsNoScoreRatherThanPropagating() {
        let corpus = [chunk(0), chunk(1)]
        let hits = IndexCoordinator.rank(
            corpus: corpus,
            semantic: [corpus[0].id: Double.nan],
            lexical: [corpus[1].id: 1.0],
            limit: 5
        )

        // The NaN-only chunk contributes nothing usable, so it is omitted —
        // not returned at some arbitrary finite score.
        XCTAssertEqual(hits.count, 1)
        XCTAssertEqual(hits.first?.chunk.id, corpus[1].id)
        XCTAssertEqual(hits.first?.source, .lexicalFallback)
        XCTAssertEqual(hits.first?.score ?? 0, 0.35, accuracy: 1e-12)
        for hit in hits { XCTAssertTrue(hit.score.isFinite) }

        // And the surviving chunk's score is exactly what it would have been
        // without the NaN present at all — the poison does not leak through
        // the min-max normalisation.
        let clean = IndexCoordinator.rank(
            corpus: corpus, semantic: [:], lexical: [corpus[1].id: 1.0], limit: 5
        )
        XCTAssertEqual(clean.first?.score ?? -1, hits.first?.score ?? -2, accuracy: 1e-12)
    }

    /// Cardinality alone would pass against a `rank` that returned any three
    /// chunks in any order, so the identity and order of the top slice are
    /// asserted against deterministic scores.
    func testLimitTakesTheHighestScoringSliceInOrder() {
        let corpus = (0..<10).map { chunk($0) }
        // Descending, so the expected top three are chunks 0, 1, 2.
        let lexical = Dictionary(
            uniqueKeysWithValues: corpus.enumerated().map { ($0.element.id, Double(10 - $0.offset)) }
        )
        let hits = IndexCoordinator.rank(corpus: corpus, semantic: [:], lexical: lexical, limit: 3)

        XCTAssertEqual(hits.count, 3)
        XCTAssertEqual(hits.map(\.chunk.id), [corpus[0].id, corpus[1].id, corpus[2].id])
        XCTAssertGreaterThan(hits[0].score, hits[1].score)
        XCTAssertGreaterThan(hits[1].score, hits[2].score)
    }
}

final class CompletenessTests: XCTestCase {

    func testSummaryTellsTheThreeStoriesApart() {
        let complete = Completeness(liveChunks: 10, semanticallyCovered: 10, awaitingReindex: 0, isLexicalOnly: false)
        XCTAssertTrue(complete.isComplete)
        XCTAssertTrue(complete.summary.contains("all 10"))

        let degraded = Completeness(liveChunks: 10, semanticallyCovered: 6, awaitingReindex: 4, isLexicalOnly: false)
        XCTAssertFalse(degraded.isComplete)
        XCTAssertTrue(degraded.summary.contains("60%"), "Got: \(degraded.summary)")
        XCTAssertTrue(degraded.summary.contains("4 still re-indexing"))

        let lexical = Completeness(liveChunks: 10, semanticallyCovered: 0, awaitingReindex: 10, isLexicalOnly: true)
        XCTAssertFalse(lexical.isComplete)
        XCTAssertTrue(lexical.summary.contains("Keyword search only"))
    }

    func testNegativeInputsAreClampedAtConstruction() {
        let value = Completeness(liveChunks: -3, semanticallyCovered: -1, awaitingReindex: -2, isLexicalOnly: false)
        XCTAssertEqual(value.liveChunks, 0)
        XCTAssertEqual(value.semanticCoverage, 1.0)
    }

    func testMigrationProgressFractionIsDefinedForAnEmptyQueue() {
        let epoch = EmbeddingEpoch(modelIdentifier: "m", revision: 1, dimension: 8)
        let empty = MigrationProgress(targetEpoch: epoch, completed: 0, remaining: 0)
        XCTAssertEqual(empty.fraction, 1.0)
        XCTAssertTrue(empty.isFinished)

        let half = MigrationProgress(targetEpoch: epoch, completed: 5, remaining: 5)
        XCTAssertEqual(half.fraction, 0.5)
        XCTAssertFalse(half.isFinished)
    }
}
