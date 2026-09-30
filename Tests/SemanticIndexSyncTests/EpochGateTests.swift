import XCTest
@testable import SemanticIndexSync

final class EpochGateTests: XCTestCase {

    private let oldEpoch = EmbeddingEpoch(modelIdentifier: "system.text", revision: 3, dimension: 24)
    private let newEpoch = EmbeddingEpoch(modelIdentifier: "system.text", revision: 4, dimension: 24)

    // MARK: The gate has teeth

    /// The whole design rests on one claim: a vector from another embedding
    /// space can score *arbitrarily well* and still mean nothing.
    ///
    /// The demonstration uses explicitly constructed vectors rather than the
    /// package's own embedder, because that embedder salts tokens with the
    /// revision and therefore makes the two spaces orthogonal — a cross-epoch
    /// cosine of exactly 0, which a naive implementation would *not* rank and
    /// which would prove nothing. Real embedding models are not orthogonal
    /// across revisions: retrained weights produce vectors that are numerically
    /// close and semantically incomparable, which is precisely why the failure
    /// is invisible. So the test builds that case directly: a stale vector that
    /// scores 1.0, and a gate that rejects it anyway.
    func testAStaleVectorScoringNearPerfectlyIsStillRejected() {
        // What the *previous* model produced for this passage, still on disk.
        let storedInOldEpoch = VectorMath.l2Normalized((0..<24).map { Double(($0 % 7) + 1) })
        // What the *new* model produces for the query. A different vector — as
        // it must be, or there would be nothing to demonstrate — but a nearby
        // one, which is what a retrained revision of the same model actually
        // yields. Asserting `cos(v, v) == 1` instead would be a property of the
        // cosine function, true against every implementation, and would prove
        // nothing about this package.
        let queryInNewEpoch = VectorMath.l2Normalized((0..<24).map { Double(($0 % 7) + 1) + 0.15 })
        XCTAssertNotEqual(queryInNewEpoch, storedInOldEpoch)

        // A naive implementation compares them directly. The arithmetic
        // succeeds and returns a score high enough to rank at the very top.
        let naiveSimilarity = VectorMath.cosineSimilarity(queryInNewEpoch, storedInOldEpoch)
        XCTAssertGreaterThan(
            naiveSimilarity, 0.9,
            "The stale vector must score near the top, or this test does not demonstrate the bug."
        )
        let values = storedInOldEpoch

        let chunk = Chunk(
            id: ChunkID(document: DocumentID("d"), ordinal: 0),
            text: "thermal budget pauses the re-index queue",
            sourceHash: ContentHash("h0")
        )
        let stale = StoredVector(chunk: chunk.id, epoch: oldEpoch, sourceHash: chunk.sourceHash, values: values)
        let fresh = StoredVector(chunk: chunk.id, epoch: newEpoch, sourceHash: chunk.sourceHash, values: values)

        // Same numbers, same perfect score, opposite verdicts — epoch alone decides.
        XCTAssertFalse(stale.isFresh(for: chunk, in: newEpoch))
        XCTAssertTrue(fresh.isFresh(for: chunk, in: newEpoch))
    }

    /// And the gate holds end to end: a chunk whose only vector is in the
    /// previous epoch contributes nothing to semantic coverage, however well it
    /// would have scored.
    func testStaleEpochChunkIsExcludedFromSemanticCoverageEndToEnd() async {
        let coordinator = IndexCoordinator(
            device: DeviceID("phone"),
            provider: DeterministicEmbeddingProvider(epoch: oldEpoch)
        )
        await coordinator.upsert(
            document: DocumentID("d"),
            passages: ["thermal budget pauses the re-index queue"],
            hash: ContentHash("h0")
        )
        _ = await coordinator.drainMigration(budget: .foregroundInteractive, conditions: DeviceConditions())

        let before = await coordinator.search("thermal budget", limit: 5)
        XCTAssertEqual(before.completeness.semanticallyCovered, 1)

        await coordinator.adopt(provider: DeterministicEmbeddingProvider(epoch: newEpoch))
        let after = await coordinator.search("thermal budget", limit: 5)
        XCTAssertEqual(after.completeness.semanticallyCovered, 0)
        XCTAssertEqual(after.completeness.awaitingReindex, 1)
        XCTAssertEqual(after.hits.first?.source, .lexicalFallback)
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

    /// A non-finite component is skipped, not propagated. Asserting only
    /// `isFinite` would pass against an implementation that returned a constant,
    /// so the exact expected value is asserted instead: with the NaN pair
    /// dropped, `[_, 1, 0]` against `[_, 1, 0]` is a perfect match on what
    /// remains.
    func testNonFiniteInputsAreSkippedNotPropagated() {
        XCTAssertEqual(VectorMath.cosineSimilarity([Double.nan, 1, 0], [1, 1, 0]), 1.0, accuracy: 1e-12)
        XCTAssertEqual(VectorMath.cosineSimilarity([Double.nan, 3, 4], [1, 3, 4]), 1.0, accuracy: 1e-12)
        // And a genuinely partial match still scores partially, so the skip is
        // not silently zeroing the whole vector.
        let partial = VectorMath.cosineSimilarity([Double.nan, 1, 1], [1, 1, 0])
        XCTAssertEqual(partial, 1.0 / 2.0.squareRoot(), accuracy: 1e-12)

        // Infinity is dropped before normalisation rather than producing NaN.
        XCTAssertEqual(VectorMath.l2Normalized([Double.infinity, 1]), [0, 1])
    }

    func testEmptyVectorsScoreZero() {
        XCTAssertEqual(VectorMath.cosineSimilarity([], []), 0)
    }
}
