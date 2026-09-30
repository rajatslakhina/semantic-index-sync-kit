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

    func testDistinctInputsProduceDistinctBuckets() {
        let inputs = ["battery", "thermal", "migration", "epoch", "tombstone"]
        let hashes = Set(inputs.map { StableHash.fnv1a($0) })
        XCTAssertEqual(hashes.count, inputs.count)
    }

    /// The embedder must be a pure function of (epoch, text) with no hidden
    /// per-instance state, because a vector written today is compared against a
    /// query vector produced by a different instance tomorrow.
    func testEmbedderIsStableAcrossInstances() {
        let epoch = EmbeddingEpoch(modelIdentifier: "test", revision: 1, dimension: 16)
        let first = DeterministicEmbeddingProvider(epoch: epoch).vector(for: "battery thermal budget")
        let second = DeterministicEmbeddingProvider(epoch: epoch).vector(for: "battery thermal budget")
        XCTAssertEqual(first, second)
        XCTAssertEqual(first.count, 16)
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
