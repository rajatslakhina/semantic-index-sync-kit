import Foundation

/// The identity of an embedding space.
///
/// ## The invariant this type exists to enforce
///
/// A cosine similarity between two vectors is only meaningful if both were
/// produced by the same model at the same revision into the same dimensionality.
/// Mix epochs and the arithmetic still succeeds — it returns a number in
/// `[-1, 1]`, it sorts, the UI renders — and every result is noise. This is the
/// nastiest failure mode in an on-device retrieval stack precisely because
/// *nothing throws*.
///
/// So epoch is not metadata attached to a vector for bookkeeping. It is a
/// comparability token, and ``IndexCoordinator`` refuses to score across it.
public struct EmbeddingEpoch: Hashable, Sendable, Codable, CustomStringConvertible {

    /// Which model produced the space (e.g. a system model's bundle identifier).
    public let modelIdentifier: String

    /// Monotonic revision of that model. An OS update that reships the same
    /// model identifier with retrained weights *must* bump this, otherwise the
    /// index silently mixes two spaces under one name.
    public let revision: Int

    /// Vector length. Carried in the identity because a dimension change is the
    /// one epoch difference that would otherwise crash rather than degrade.
    public let dimension: Int

    public init(modelIdentifier: String, revision: Int, dimension: Int) {
        self.modelIdentifier = modelIdentifier
        self.revision = revision
        // A non-positive dimension cannot describe a real space, and would make
        // every similarity a division by zero downstream. Clamp at construction
        // so the invalid value never reaches the scorer.
        self.dimension = max(1, dimension)
    }

    /// Whether vectors from the two epochs may be compared to each other.
    ///
    /// Deliberately identity, not compatibility-with-fallback: there is no
    /// defensible rule for "close enough" between two embedding spaces.
    public func isComparable(to other: EmbeddingEpoch) -> Bool { self == other }

    public var description: String { "\(modelIdentifier)@r\(revision)/d\(dimension)" }
}
