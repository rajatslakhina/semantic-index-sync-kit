import Foundation

/// Trap-free arithmetic for the few places this library converts between
/// `Double` scores and `Int` counts, or accumulates unbounded totals.
///
/// Every operation Swift can trap on is funnelled through here rather than
/// guarded ad hoc at each call site. Scattered guards are how one gets missed;
/// a single audited helper is how the audit stays cheap.
///
/// All ceilings are derived from `Int.max` rather than a 64-bit literal, because
/// `Int` is 32-bit on watchOS and a hardcoded `9_223_372_036_854_775_807` would
/// be a compile error there and a silent wrong answer if written as a `Double`.
public enum Saturating {

    /// `a + b`, clamped to the representable range instead of trapping.
    public static func add(_ a: Int, _ b: Int) -> Int {
        let (result, overflow) = a.addingReportingOverflow(b)
        guard overflow else { return result }
        return b > 0 ? Int.max : Int.min
    }

    /// `a - b`, clamped to the representable range instead of trapping.
    public static func subtract(_ a: Int, _ b: Int) -> Int {
        let (result, overflow) = a.subtractingReportingOverflow(b)
        guard overflow else { return result }
        return b < 0 ? Int.max : Int.min
    }

    /// `a * b`, clamped to the representable range instead of trapping.
    public static func multiply(_ a: Int, _ b: Int) -> Int {
        let (result, overflow) = a.multipliedReportingOverflow(by: b)
        guard overflow else { return result }
        // Sign of the true product decides which rail we clamp to. Zero cannot
        // overflow, so neither operand is zero here.
        let negative = (a < 0) != (b < 0)
        return negative ? Int.min : Int.max
    }

    /// `a / b`. Returns `fallback` for `b == 0` and for `Int.min / -1`, the two
    /// integer divisions that trap in Swift.
    public static func divide(_ a: Int, _ b: Int, fallback: Int = 0) -> Int {
        guard b != 0 else { return fallback }
        guard !(a == Int.min && b == -1) else { return Int.max }
        return a / b
    }

    /// `a % b`. Returns `fallback` for `b == 0` and for `Int.min % -1`.
    public static func remainder(_ a: Int, _ b: Int, fallback: Int = 0) -> Int {
        guard b != 0 else { return fallback }
        guard !(a == Int.min && b == -1) else { return 0 }
        return a % b
    }

    /// `Int(value)` without the three traps that conversion carries: NaN,
    /// infinity, and a magnitude outside `Int`'s range.
    public static func int(clamping value: Double) -> Int {
        guard !value.isNaN else { return 0 }
        // `Double(Int.max)` rounds *up* to exactly 2^63 on 64-bit, so `>=` is the
        // correct comparison: anything at or above it is out of range.
        // `Double(Int.min)` is exactly representable, so `<=` is correct there.
        if value >= Double(Int.max) { return Int.max }
        if value <= Double(Int.min) { return Int.min }
        return Int(value)
    }

    /// `numerator / denominator` as a fraction, with a defined answer when the
    /// denominator is zero or negative (both of which mean "no population").
    public static func ratio(_ numerator: Int, _ denominator: Int, whenEmpty: Double = 1.0) -> Double {
        guard denominator > 0 else { return whenEmpty }
        let clamped = min(max(numerator, 0), denominator)
        return Double(clamped) / Double(denominator)
    }
}
