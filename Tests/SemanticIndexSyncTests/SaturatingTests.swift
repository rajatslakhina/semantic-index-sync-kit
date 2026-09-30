import XCTest
@testable import SemanticIndexSync

/// Every case here is an operation Swift traps on. A trap is a crash, and a
/// crash in a background re-index task is a data-loss bug with no log line.
final class SaturatingTests: XCTestCase {

    func testAdditionClampsInsteadOfTrapping() {
        XCTAssertEqual(Saturating.add(Int.max, 1), Int.max)
        XCTAssertEqual(Saturating.add(Int.min, -1), Int.min)
        XCTAssertEqual(Saturating.add(Int.max, Int.max), Int.max)
        XCTAssertEqual(Saturating.add(3, 4), 7)
    }

    func testSubtractionClampsInsteadOfTrapping() {
        XCTAssertEqual(Saturating.subtract(Int.min, 1), Int.min)
        XCTAssertEqual(Saturating.subtract(Int.max, -1), Int.max)
        XCTAssertEqual(Saturating.subtract(10, 4), 6)
    }

    func testMultiplicationClampsToTheCorrectRail() {
        XCTAssertEqual(Saturating.multiply(Int.max, 2), Int.max)
        XCTAssertEqual(Saturating.multiply(Int.max, -2), Int.min)
        XCTAssertEqual(Saturating.multiply(Int.min, -1), Int.max)
        XCTAssertEqual(Saturating.multiply(0, Int.max), 0)
        XCTAssertEqual(Saturating.multiply(6, 7), 42)
    }

    func testDivisionHandlesTheTwoTrappingCases() {
        XCTAssertEqual(Saturating.divide(10, 0, fallback: -1), -1)
        XCTAssertEqual(Saturating.divide(Int.min, -1), Int.max)
        XCTAssertEqual(Saturating.divide(9, 2), 4)
    }

    func testRemainderHandlesTheTwoTrappingCases() {
        XCTAssertEqual(Saturating.remainder(10, 0, fallback: 7), 7)
        XCTAssertEqual(Saturating.remainder(Int.min, -1), 0)
        XCTAssertEqual(Saturating.remainder(9, 4), 1)
    }

    /// `Int(Double)` traps three different ways. All three are covered, and the
    /// bounds are derived from `Int.max`, so this test is also correct on a
    /// 32-bit `Int` platform.
    func testDoubleToIntConversionIsTotal() {
        XCTAssertEqual(Saturating.int(clamping: Double.nan), 0)
        XCTAssertEqual(Saturating.int(clamping: Double.infinity), Int.max)
        XCTAssertEqual(Saturating.int(clamping: -Double.infinity), Int.min)
        XCTAssertEqual(Saturating.int(clamping: Double(Int.max)), Int.max)
        XCTAssertEqual(Saturating.int(clamping: Double(Int.min)), Int.min)
        XCTAssertEqual(Saturating.int(clamping: 1e300), Int.max)
        XCTAssertEqual(Saturating.int(clamping: -1e300), Int.min)
        XCTAssertEqual(Saturating.int(clamping: 3.99), 3)
        XCTAssertEqual(Saturating.int(clamping: -3.99), -3)
    }

    func testRatioDefinesTheEmptyAndOutOfRangeCases() {
        XCTAssertEqual(Saturating.ratio(0, 0), 1.0)
        XCTAssertEqual(Saturating.ratio(5, 0, whenEmpty: 0.0), 0.0)
        XCTAssertEqual(Saturating.ratio(3, -2), 1.0)
        XCTAssertEqual(Saturating.ratio(1, 4), 0.25)
        // Numerator above the denominator is clamped rather than reporting >100%.
        XCTAssertEqual(Saturating.ratio(9, 4), 1.0)
        XCTAssertEqual(Saturating.ratio(-3, 4), 0.0)
    }
}
