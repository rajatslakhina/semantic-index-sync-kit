import Foundation

/// Stable identity of one device participating in the index's sync mesh.
///
/// Device identity is load-bearing here, not cosmetic: it is the key space of
/// ``VersionVector``, which is how this library orders concurrent edits without
/// trusting wall-clock time.
public struct DeviceID: Hashable, Sendable, Codable, CustomStringConvertible, Comparable {
    public let raw: String

    public init(_ raw: String) { self.raw = raw }

    public var description: String { raw }

    public static func < (lhs: DeviceID, rhs: DeviceID) -> Bool { lhs.raw < rhs.raw }
}

/// Identity of a user-visible document (a note, a message thread, a saved page).
public struct DocumentID: Hashable, Sendable, Codable, CustomStringConvertible, Comparable {
    public let raw: String

    public init(_ raw: String) { self.raw = raw }

    public var description: String { raw }

    public static func < (lhs: DocumentID, rhs: DocumentID) -> Bool { lhs.raw < rhs.raw }
}

/// Identity of one embeddable slice of a document.
///
/// Chunk identity is `(document, ordinal)` rather than a content hash on purpose:
/// re-chunking an edited document must *reuse* chunk identities so the migration
/// queue can tell "this slot needs a fresh vector" from "this is a new slot".
public struct ChunkID: Hashable, Sendable, Codable, CustomStringConvertible, Comparable {
    public let document: DocumentID
    public let ordinal: Int

    public init(document: DocumentID, ordinal: Int) {
        self.document = document
        // A negative ordinal would still be a usable dictionary key, but it can
        // never be produced by chunking and would silently corrupt ordering, so
        // it is clamped rather than trusted.
        self.ordinal = max(0, ordinal)
    }

    public var description: String { "\(document.raw)#\(ordinal)" }

    public static func < (lhs: ChunkID, rhs: ChunkID) -> Bool {
        if lhs.document != rhs.document { return lhs.document < rhs.document }
        return lhs.ordinal < rhs.ordinal
    }
}

/// A caller-supplied content fingerprint for a document revision.
///
/// The library deliberately does not compute this: hashing strategy (and whether
/// it is cryptographic) is an application decision, and forcing one here would
/// make the package depend on a crypto module it does not otherwise need.
/// What the library *does* guarantee is that it never compares a stored vector
/// against a chunk whose hash has moved on — see ``StoredVector/isFresh(for:in:)``.
public struct ContentHash: Hashable, Sendable, Codable, CustomStringConvertible, Comparable {
    public let raw: String

    public init(_ raw: String) { self.raw = raw }

    public var description: String { raw }

    public static func < (lhs: ContentHash, rhs: ContentHash) -> Bool { lhs.raw < rhs.raw }
}
