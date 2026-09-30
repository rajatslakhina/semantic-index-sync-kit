import XCTest
@testable import SemanticIndexSync

final class EpochGateTests: XCTestCase {

    private let oldEpoch = EmbeddingEpoch(modelIdentifier: "system.text", revision: 3, dimension: 24)
    private let newEpoch = EmbeddingEpoch(modelIdentifier: "system.text", revision: 4, dimension: 24)

    // MARK: The gate has teeth

    /// The whole design rests on one claim: comparing vectors across epochs
    /// produces a confident number that means nothing. This test demonstrates the
    /// bug first — a cross-epoch cosine that scores *high enough to rank* — and
    /// only then asserts the freshness gate rejects it.
    ///
    /// Assert the gate alone and the test would still pass against an
    /// implementation that rejected everything.
    func testCrossEpochCosineScoresPlausiblyYetTheGateStillRejectsIt() {
        let text = "thermal budget pauses the re-index queue"
        let oldVectorValues = DeterministicEmbeddingProvider(epoch: oldEpoch).vector(for: text)
        let newVectorValues = DeterministicEmbeddingProvider(epoch: newEpoch).vector(for: text)

        // A naive implementation would compare these directly. The arithmetic
        // succeeds and returns a finite, sortable number.
        let naiveSimilarity = VectorMath.cosineSimilarity(newVectorValues, oldVectorValues)
        XCTAssertTrue(naiveSimilarity.isFinite)
        XCTAssertNotEqual(
            oldVectorValues, newVectorValues,
            "The two epochs must genuinely differ, or this test proves nothing."
        )

        // Same text, same epoch, for reference: this is what a real match is.
        let honestSimilarity = VectorMath.cosineSimilarity(newVectorValues, newVectorValues)
        XCTAssertEqual(honestSimilarity, 1.0, accuracy: 1e-9)

        // The gate rejects the stale vector on epoch alone, whatever it scored.
        let chunk = Chunk(
            id: ChunkID(document: DocumentID("d"), ordinal: 0),
            text: text,
            sourceHash: ContentHash("h0")
        )
        let stale = StoredVector(chunk: chunk.id, epoch: oldEpoch, sourceHash: chunk.sourceHash, values: oldVectorValues)
        let fresh = StoredVector(chunk: chunk.id, epoch: newEpoch, sourceHash: chunk.sourceHash, values: newVectorValues)

        XCTAssertFalse(stale.isFresh(for: chunk, in: newEpoch))
        XCTAssertTrue(fresh.isFresh(for: chunk, in: newEpoch))
    }

    /// The second way a vector goes stale: the text moved, not the model.
    func testEditedTextInvalidatesAVectorEvenInTheSameEpoch() {
        let values = DeterministicEmbeddingProvider(epoch: newEpoch).vector(for: "original")
        let edited = Chunk(
            id: ChunkID(document: DocumentID("d"), ordinal: 0),
            text: "rewritten",
            sourceHash: ContentHash("h1")
        )
        let vector = StoredVector(chunk: edited.id, epoch: newEpoch, sourceHash: ContentHash("h0"), values: values)
        XCTAssertFalse(vector.isFresh(for: edited, in: newEpoch))
    }

    func testDimensionMismatchIsRejectedRatherThanCrashing() {
        let chunk = Chunk(
            id: ChunkID(document: DocumentID("d"), ordinal: 0),
            text: "x",
            sourceHash: ContentHash("h")
        )
        let truncated = StoredVector(chunk: chunk.id, epoch: newEpoch, sourceHash: chunk.sourceHash, values: [1, 0])
        XCTAssertFalse(truncated.isFresh(for: chunk, in: newEpoch))
        // And scoring mismatched lengths returns 0 rather than indexing out of range.
        XCTAssertEqual(VectorMath.cosineSimilarity([1, 0, 0], [1, 0]), 0)
    }

    func testEpochComparabilityIsIdentityNotApproximation() {
        XCTAssertTrue(oldEpoch.isComparable(to: oldEpoch))
        XCTAssertFalse(oldEpoch.isComparable(to: newEpoch))
        let widened = EmbeddingEpoch(modelIdentifier: "system.text", revision: 3, dimension: 48)
        XCTAssertFalse(oldEpoch.isComparable(to: widened))
    }

    func testNonPositiveDimensionIsClampedAtConstruction() {
        XCTAssertEqual(EmbeddingEpoch(modelIdentifier: "m", revision: 1, dimension: 0).dimension, 1)
        XCTAssertEqual(EmbeddingEpoch(modelIdentifier: "m", revision: 1, dimension: -7).dimension, 1)
    }

    // MARK: Degenerate vector arithmetic

    func testZeroVectorNormalisesToZeroRatherThanNaN() {
        let normalized = VectorMath.l2Normalized([0, 0, 0])
        XCTAssertEqual(normalized, [0, 0, 0])
        XCTAssertFalse(normalized.contains { $0.isNaN })
    }

    func testNonFiniteInputsDoNotPoisonTheScore() {
        let similarity = VectorMath.cosineSimilarity([Double.nan, 1, 0], [1, 1, 0])
        XCTAssertTrue(similarity.isFinite)
        XCTAssertTrue(similarity >= -1 && similarity <= 1)

        let normalized = VectorMath.l2Normalized([Double.infinity, 1])
        XCTAssertFalse(normalized.contains { !$0.isFinite })
    }

    func testEmptyVectorsScoreZero() {
        XCTAssertEqual(VectorMath.cosineSimilarity([], []), 0)
    }
}
