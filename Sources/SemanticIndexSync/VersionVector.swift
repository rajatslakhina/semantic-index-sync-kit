import Foundation

/// How two versions of the same document relate in causal order.
public enum CausalOrder: String, Sendable, Equatable, CustomStringConvertible {
    /// The two versions are the same point in history.
    case identical
    /// The left version happened strictly before the right one.
    case ancestor
    /// The left version happened strictly after the right one.
    case descendant
    /// Neither version saw the other. This is the case last-writer-wins hides.
    case concurrent

    public var description: String { rawValue }
}

/// A per-device counter map giving a partial order over document revisions.
///
/// ## Why not a timestamp
///
/// The obvious cheap design is last-writer-wins on a wall-clock timestamp. It is
/// wrong for this system in a way that is not merely theoretical: device clocks
/// disagree, users change them, and an offline device rejoining after two days
/// carries writes stamped in the past. Under LWW, a *delete* performed on the
/// phone can be silently undone by an older edit arriving later from the laptop,
/// which for a searchable index of personal content means deleted material
/// becomes findable again. A version vector cannot express that mistake: it can
/// tell "this write never saw the delete" apart from "this write supersedes it",
/// which is exactly the distinction LWW throws away.
///
/// ## Cost accepted
///
/// The vector is `O(devices)` per document rather than `O(1)`. For a personal
/// index — single-digit devices per account — that is a few dozen bytes, and it
/// buys a correctness property the cheaper encoding cannot have. It would be the
/// wrong trade at fleet scale with unbounded writers; that is a documented limit,
/// not an oversight. See ``VersionVector/counters``.
public struct VersionVector: Sendable, Equatable, Codable, CustomStringConvertible {

    /// Invariant: no entry is ever stored with value `0`. Absence *is* zero, so
    /// that two vectors describing the same history compare equal regardless of
    /// which devices happen to appear in the dictionary.
    private var storage: [DeviceID: UInt64]

    public init() { storage = [:] }

    public init(_ counters: [DeviceID: UInt64]) {
        storage = counters.filter { $0.value > 0 }
    }

    /// The non-zero counters, exposed for persistence and diagnostics.
    public var counters: [DeviceID: UInt64] { storage }

    /// Number of devices that have ever written this document.
    public var writerCount: Int { storage.count }

    public func counter(for device: DeviceID) -> UInt64 {
        storage[device] ?? 0
    }

    /// Records one local write by `device`.
    ///
    /// Saturates at `UInt64.max` rather than wrapping. Wrapping would silently
    /// reorder history; saturating freezes it, which is detectable. At one write
    /// per nanosecond this is unreachable for ~584 years, so the branch exists to
    /// keep the operation total, not because it is expected to fire.
    public mutating func increment(_ device: DeviceID) {
        let current = storage[device] ?? 0
        storage[device] = current == UInt64.max ? UInt64.max : current &+ 1
    }

    public func incrementing(_ device: DeviceID) -> VersionVector {
        var copy = self
        copy.increment(device)
        return copy
    }

    /// Pointwise maximum: the causal join of two histories.
    public mutating func merge(_ other: VersionVector) {
        for (device, value) in other.storage where value > (storage[device] ?? 0) {
            storage[device] = value
        }
    }

    public func merging(_ other: VersionVector) -> VersionVector {
        var copy = self
        copy.merge(other)
        return copy
    }

    /// True when this vector has seen everything `other` has seen.
    public func dominatesOrEquals(_ other: VersionVector) -> Bool {
        for (device, value) in other.storage where (storage[device] ?? 0) < value {
            return false
        }
        return true
    }

    /// The partial order between two vectors.
    public static func order(_ lhs: VersionVector, _ rhs: VersionVector) -> CausalOrder {
        let lhsSeesRhs = lhs.dominatesOrEquals(rhs)
        let rhsSeesLhs = rhs.dominatesOrEquals(lhs)
        switch (lhsSeesRhs, rhsSeesLhs) {
        case (true, true): return .identical
        case (true, false): return .descendant
        case (false, true): return .ancestor
        case (false, false): return .concurrent
        }
    }

    public var description: String {
        guard !storage.isEmpty else { return "{}" }
        let body = storage
            .sorted { $0.key < $1.key }
            .map { "\($0.key.raw):\($0.value)" }
            .joined(separator: ",")
        return "{\(body)}"
    }
}
