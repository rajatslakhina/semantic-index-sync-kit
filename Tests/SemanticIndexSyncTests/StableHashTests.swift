import XCTest
@testable import SemanticIndexSync

/// These assert against **hardcoded** FNV-1a reference values.
///
/// The tempting version of this test — hash a string twice in one process and
/// assert the two agree — passes for `Hasher` too, which is precisely the
/// function that must not be used here because its seed is randomised per
/// process. A test that cannot fail against the bug it exists to catch is worse
/// than no test. Pinning the constants is what gives this one teeth: swap in
/// `Hasher`, or change the prime, and it fails immediately.
final class StableHashTests: XCTestCase {

    func testMatchesReferenceFNV1aVectors() {
        XCTAssertEqual(StableHash.fnv1a(""), 14_695_981_039_346_656_037)
        XCTAssertEqual(StableHash.fnv1a("a"), 12_638_187_200_555_641_996)
        XCTAssertEqual(StableHash.fnv1a("semantic-index-sync"), 17_151_552_246_426_296_080)
        XCTAssertEqual(StableHash.fnv1a("thermal"), 5_790_970_284_486_884_672)
    }

    /// The embedder must produce the same vector in every process, forever,
    /// because a vector written to disk today is compared against a query vector
    /// produced by a different launch tomorrow.
    ///
    /// Asserting that two instances *in this process* agree would pass for
    /// `Hasher` too, so the expectation is pinned to the exact values instead —
    /// computed independently from the FNV-1a definition, not from this code.
    func testEmbedderOutputIsPinnedAcrossProcesses() {
        let epoch = EmbeddingEpoch(modelIdentifier: "test", revision: 1, dimension: 64)
        let produced = DeterministicEmbeddingProvider(epoch: epoch).vector(for: "battery thermal budget")

        XCTAssertEqual(produced.count, 64)

        // Three tokens, three distinct buckets, unit length: each |value| is 1/√3.
        let unit = 1.0 / 3.0.squareRoot()
        var expected = [Double](repeating: 0, count: 64)
        expected[15] = -unit
        expected[45] = -unit
        expected[47] = unit

        for index in produced.indices {
            XCTAssertEqual(produced[index], expected[index], accuracy: 1e-12, "bucket \(index)")
        }
    }

    /// A revision bump must genuinely change the space. If it did not, every
    /// epoch test in this suite would pass for the wrong reason.
    func testRevisionBumpChangesTheVectorSpace() {
        let text = "battery thermal budget"
        let old = DeterministicEmbeddingProvider(
            epoch: EmbeddingEpoch(modelIdentifier: "test", revision: 1, dimension: 32)
        ).vector(for: text)
        let new = DeterministicEmbeddingProvider(
            epoch: EmbeddingEpoch(modelIdentifier: "test", revision: 2, dimension: 32)
        ).vector(for: text)
        XCTAssertNotEqual(old, new)
    }
}
