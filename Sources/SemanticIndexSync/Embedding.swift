import Foundation

/// A chunk of text the index can embed and search.
public struct Chunk: Sendable, Equatable, Identifiable {
    public let id: ChunkID
    public let text: String
    /// Fingerprint of the *document revision* this chunk was cut from. A vector
    /// is only valid for the hash it was produced under.
    public let sourceHash: ContentHash

    public init(id: ChunkID, text: String, sourceHash: ContentHash) {
        self.id = id
        self.text = text
        self.sourceHash = sourceHash
    }
}

/// A vector, stamped with the two facts that decide whether it may be used.
public struct StoredVector: Sendable, Equatable {
    public let chunk: ChunkID
    public let epoch: EmbeddingEpoch
    public let sourceHash: ContentHash
    public let values: [Double]

    public init(chunk: ChunkID, epoch: EmbeddingEpoch, sourceHash: ContentHash, values: [Double]) {
        self.chunk = chunk
        self.epoch = epoch
        self.sourceHash = sourceHash
        self.values = values
    }

    /// Usable for a query in `epoch` against the current text of `chunk`.
    ///
    /// Two independent ways to go stale, and both must be checked:
    /// the model moved (epoch), or the text moved (hash).
    public func isFresh(for chunk: Chunk, in epoch: EmbeddingEpoch) -> Bool {
        self.chunk == chunk.id
            && self.epoch.isComparable(to: epoch)
            && self.sourceHash == chunk.sourceHash
            && self.values.count == epoch.dimension
    }
}

public enum EmbeddingError: Error, Sendable, Equatable {
    case providerUnavailable
    case dimensionMismatch(expected: Int, actual: Int)
}

/// Source of embeddings. Abstracted so the index can be exercised end to end on
/// Linux CI with no model weights present — the retrieval system's correctness
/// does not depend on the quality of the vectors, only on their epoch discipline.
public protocol EmbeddingProvider: Sendable {
    var epoch: EmbeddingEpoch { get }
    var isAvailable: Bool { get }
    func embed(_ texts: [String]) async throws -> [[Double]]
}

/// A seeded, process-stable hash. Deliberately **not** Swift's `Hasher`, whose
/// seed is randomised per process: a vector persisted on Monday must compare
/// identically on Tuesday, and any hash that changes between launches would make
/// the whole on-disk index silently meaningless after a relaunch.
public enum StableHash {
    private static let offsetBasis: UInt64 = 0xcbf2_9ce4_8422_2325
    private static let prime: UInt64 = 0x0000_0100_0000_01B3

    /// FNV-1a over the string's UTF-8 bytes.
    public static func fnv1a(_ string: String) -> UInt64 {
        var hash = offsetBasis
        for byte in string.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* prime
        }
        return hash
    }
}

/// A deterministic feature-hashing embedder.
///
/// Not a language model and not pretending to be one: it hashes tokens into
/// buckets and L2-normalises, so documents sharing vocabulary land near each
/// other. That is enough to make the *system* behaviour — epoch gating,
/// migration, coverage reporting — observable and testable with zero ML
/// dependencies, which is the point. A production app injects the real provider.
public struct DeterministicEmbeddingProvider: EmbeddingProvider {
    public let epoch: EmbeddingEpoch
    public let isAvailable: Bool

    public init(epoch: EmbeddingEpoch, isAvailable: Bool = true) {
        self.epoch = epoch
        self.isAvailable = isAvailable
    }

    public func embed(_ texts: [String]) async throws -> [[Double]] {
        guard isAvailable else { throw EmbeddingError.providerUnavailable }
        return texts.map { vector(for: $0) }
    }

    /// Exposed synchronously so tests can assert stability without concurrency.
    public func vector(for text: String) -> [Double] {
        let dimension = epoch.dimension
        var accumulator = [Double](repeating: 0, count: dimension)

        for token in Tokenizer.tokens(in: text) {
            // The revision is folded into the token before hashing, so a model
            // revision bump genuinely produces a different space rather than the
            // same numbers under a new label. Without this, the epoch tests
            // would pass for the wrong reason.
            let salted = "\(epoch.modelIdentifier)|\(epoch.revision)|\(token)"
            let hash = StableHash.fnv1a(salted)
            let bucket = Int(hash % UInt64(max(1, dimension)))
            guard accumulator.indices.contains(bucket) else { continue }
            // Sign bucket keeps unrelated tokens from all pushing the same way.
            let sign: Double = (hash >> 63) == 1 ? -1 : 1
            accumulator[bucket] += sign
        }

        return VectorMath.l2Normalized(accumulator)
    }
}

/// Vector arithmetic with every degenerate case defined.
public enum VectorMath {

    /// Unit-length copy. A zero vector (a chunk with no tokens) normalises to
    /// zero rather than NaN, and a zero vector scores 0 against everything,
    /// which is the honest answer.
    public static func l2Normalized(_ values: [Double]) -> [Double] {
        var sumOfSquares = 0.0
        for value in values where value.isFinite {
            sumOfSquares += value * value
        }
        let magnitude = sumOfSquares.squareRoot()
        guard magnitude > 0, magnitude.isFinite else {
            return [Double](repeating: 0, count: values.count)
        }
        return values.map { $0.isFinite ? $0 / magnitude : 0 }
    }

    /// Cosine similarity of two vectors.
    ///
    /// Returns `0` for mismatched lengths instead of crashing on an index out of
    /// range. A length mismatch means an epoch bug upstream; the scorer's job is
    /// to not turn that bug into a crash in front of the user.
    public static func cosineSimilarity(_ lhs: [Double], _ rhs: [Double]) -> Double {
        guard lhs.count == rhs.count, !lhs.isEmpty else { return 0 }
        var dot = 0.0
        var lhsSquares = 0.0
        var rhsSquares = 0.0
        for index in lhs.indices {
            let l = lhs[index]
            let r = rhs[index]
            guard l.isFinite, r.isFinite else { continue }
            dot += l * r
            lhsSquares += l * l
            rhsSquares += r * r
        }
        let denominator = (lhsSquares * rhsSquares).squareRoot()
        guard denominator > 0, denominator.isFinite else { return 0 }
        let similarity = dot / denominator
        guard similarity.isFinite else { return 0 }
        return min(max(similarity, -1), 1)
    }
}

/// Lowercasing word tokenizer shared by the vector and lexical paths, so the two
/// scorers never disagree about what a token is.
public enum Tokenizer {
    public static func tokens(in text: String) -> [String] {
        text.lowercased()
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .map(String.init)
            .filter { !$0.isEmpty }
    }
}
