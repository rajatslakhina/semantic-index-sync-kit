import Foundation

/// The lifecycle state of a document in the manifest.
public enum RecordState: Sendable, Equatable, Codable {
    /// The document exists, with the given content fingerprint.
    case live(ContentHash)
    /// The document was deleted. The tombstone is retained rather than the row
    /// being dropped, because "absent" and "deleted" are different facts to a
    /// peer that has not seen the delete yet.
    case tombstone

    public var isTombstone: Bool {
        if case .tombstone = self { return true }
        return false
    }

    public var contentHash: ContentHash? {
        if case .live(let hash) = self { return hash }
        return nil
    }
}

/// One row of the replicated manifest: what a device believes about a document.
///
/// This is the only thing that syncs between devices. Vectors deliberately do
/// not travel — see ``IndexCoordinator`` for why.
public struct DocumentRecord: Sendable, Equatable, Codable, Identifiable {
    public let id: DocumentID
    public var state: RecordState
    public var version: VersionVector
    /// Which device produced the most recent local mutation. Used only as the
    /// final, deterministic tie-break between otherwise indistinguishable
    /// concurrent edits — never as an ordering signal.
    public var lastWriter: DeviceID

    public init(id: DocumentID, state: RecordState, version: VersionVector, lastWriter: DeviceID) {
        self.id = id
        self.state = state
        self.version = version
        self.lastWriter = lastWriter
    }

    /// A freshly authored live document.
    public static func live(
        _ id: DocumentID,
        hash: ContentHash,
        by device: DeviceID,
        from base: VersionVector = VersionVector()
    ) -> DocumentRecord {
        DocumentRecord(
            id: id,
            state: .live(hash),
            version: base.incrementing(device),
            lastWriter: device
        )
    }

    /// The tombstone that supersedes this record.
    public func deleted(by device: DeviceID) -> DocumentRecord {
        DocumentRecord(
            id: id,
            state: .tombstone,
            version: version.incrementing(device),
            lastWriter: device
        )
    }

    /// An edit to this record's content.
    public func edited(to hash: ContentHash, by device: DeviceID) -> DocumentRecord {
        DocumentRecord(
            id: id,
            state: .live(hash),
            version: version.incrementing(device),
            lastWriter: device
        )
    }
}
