import XCTest
@testable import SemanticIndexSync

final class VersionVectorTests: XCTestCase {

    private let alpha = DeviceID("alpha")
    private let bravo = DeviceID("bravo")

    func testAbsenceEqualsZeroSoEqualHistoriesCompareEqual() {
        var explicit = VersionVector([alpha: 1, bravo: 0])
        let implicit = VersionVector([alpha: 1])
        XCTAssertEqual(explicit, implicit)
        explicit.merge(VersionVector([bravo: 0]))
        XCTAssertEqual(explicit, implicit)
        XCTAssertEqual(explicit.counter(for: bravo), 0)
        XCTAssertEqual(explicit.writerCount, 1)
    }

    func testOrderCoversAllFourRelations() {
        let base = VersionVector([alpha: 1])
        XCTAssertEqual(VersionVector.order(base, base), .identical)

        let later = base.incrementing(alpha)
        XCTAssertEqual(VersionVector.order(later, base), .descendant)
        XCTAssertEqual(VersionVector.order(base, later), .ancestor)

        let sibling = base.incrementing(bravo)
        XCTAssertEqual(VersionVector.order(later, sibling), .concurrent)
        XCTAssertEqual(VersionVector.order(sibling, later), .concurrent)
    }

    func testMergeIsThePointwiseMaximumAndIsCommutative() {
        let left = VersionVector([alpha: 3, bravo: 1])
        let right = VersionVector([alpha: 1, bravo: 5])
        let merged = left.merging(right)
        XCTAssertEqual(merged, right.merging(left))
        XCTAssertEqual(merged.counter(for: alpha), 3)
        XCTAssertEqual(merged.counter(for: bravo), 5)
        XCTAssertTrue(merged.dominatesOrEquals(left))
        XCTAssertTrue(merged.dominatesOrEquals(right))
    }

    func testIncrementSaturatesRatherThanWrapping() {
        var vector = VersionVector([alpha: UInt64.max])
        vector.increment(alpha)
        // Wrapping would send this to 0 and silently invert every subsequent
        // comparison; saturating freezes history, which is at least detectable.
        XCTAssertEqual(vector.counter(for: alpha), UInt64.max)
    }

    func testEmptyVectorIsAnAncestorOfEverything() {
        let empty = VersionVector()
        let written = VersionVector([alpha: 1])
        XCTAssertEqual(VersionVector.order(empty, written), .ancestor)
        XCTAssertEqual(VersionVector.order(empty, empty), .identical)
    }

    func testDescriptionIsStablyOrdered() {
        let vector = VersionVector([bravo: 2, alpha: 1])
        XCTAssertEqual(vector.description, "{alpha:1,bravo:2}")
        XCTAssertEqual(VersionVector().description, "{}")
    }
}
