#if canImport(SwiftUI)
import Foundation
import SemanticIndexSync

/// A document the workbench seeds its index with.
public struct SeedDocument: Sendable, Identifiable, Equatable {
    public let id: DocumentID
    public let title: String
    public let passages: [String]
    public let hash: ContentHash

    public init(id: String, title: String, passages: [String], revision: String) {
        self.id = DocumentID(id)
        self.title = title
        self.passages = passages
        self.hash = ContentHash("\(id)-\(revision)")
    }
}

/// Everything the workbench needs that is *not* the library's business to decide.
///
/// The host app owns this: which corpus to seed, which model identifiers stand in
/// for the on-device model before and after an OS update, and what background
/// budget the app has chosen. Keeping it out of the library is the point — a
/// package that hardcodes a battery floor has made a product decision on the
/// app's behalf.
public struct WorkbenchConfiguration: Sendable {
    public let localDevice: DeviceID
    public let peerDevice: DeviceID
    public let baselineEpoch: EmbeddingEpoch
    public let upgradedEpoch: EmbeddingEpoch
    public let seeds: [SeedDocument]
    public let backgroundBudget: WorkBudget
    public let interactiveBudget: WorkBudget
    public let initialQuery: String

    public init(
        localDevice: DeviceID,
        peerDevice: DeviceID,
        baselineEpoch: EmbeddingEpoch,
        upgradedEpoch: EmbeddingEpoch,
        seeds: [SeedDocument],
        backgroundBudget: WorkBudget,
        interactiveBudget: WorkBudget,
        initialQuery: String
    ) {
        self.localDevice = localDevice
        self.peerDevice = peerDevice
        self.baselineEpoch = baselineEpoch
        self.upgradedEpoch = upgradedEpoch
        self.seeds = seeds
        self.backgroundBudget = backgroundBudget
        self.interactiveBudget = interactiveBudget
        self.initialQuery = initialQuery
    }
}
#endif
