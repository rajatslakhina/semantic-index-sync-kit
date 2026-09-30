import Foundation

/// Where a hit's score came from.
public enum ScoreSource: String, Sendable, Equatable, CustomStringConvertible {
    /// Scored in the query's own embedding space.
    case semantic
    /// Scored on keywords only, because no comparable vector was available.
    case lexicalFallback
    /// Scored by both paths and fused.
    case hybrid

    public var description: String { rawValue }
}

public struct ScoredChunk: Sendable, Equatable, Identifiable {
    public let chunk: Chunk
    public let score: Double
    public let source: ScoreSource

    public var id: ChunkID { chunk.id }

    public init(chunk: Chunk, score: Double, source: ScoreSource) {
        self.chunk = chunk
        self.score = score.isFinite ? score : 0
        self.source = source
    }
}

/// An honest account of how much of the corpus the query could actually see.
///
/// ## Why this is part of the return type, not a log line
///
/// During an epoch migration the index legitimately cannot answer over the whole
/// corpus in the query's embedding space. The tempting move is to return the
/// results you *can* compute and say nothing. That produces a search box that
/// quietly went shallow — the user asks a question they answered successfully
/// last week and gets nothing back, with no signal that the reason is temporary.
///
/// Making completeness a required part of every result forces the caller to
/// decide what to show. The demo app renders it as a coverage bar. The important
/// property is that silence is not the default.
public struct Completeness: Sendable, Equatable {
    /// Live (non-tombstoned) chunks in the index at query time.
    public let liveChunks: Int
    /// Chunks that had a fresh vector in the query's epoch.
    public let semanticallyCovered: Int
    /// Chunks skipped by the vector path because their vector is from an older
    /// epoch, is missing, or was produced from superseded text.
    public let awaitingReindex: Int
    /// True when no embedding provider was available and the answer is keyword-only.
    public let isLexicalOnly: Bool

    public init(liveChunks: Int, semanticallyCovered: Int, awaitingReindex: Int, isLexicalOnly: Bool) {
        self.liveChunks = max(0, liveChunks)
        self.semanticallyCovered = max(0, semanticallyCovered)
        self.awaitingReindex = max(0, awaitingReindex)
        self.isLexicalOnly = isLexicalOnly
    }

    /// Fraction of the live corpus searchable by meaning. An empty index is
    /// fully covered by definition — there is nothing missing from it.
    public var semanticCoverage: Double {
        Saturating.ratio(semanticallyCovered, liveChunks, whenEmpty: 1.0)
    }

    /// True only when every live chunk was searchable in the query's own space.
    public var isComplete: Bool {
        !isLexicalOnly && awaitingReindex == 0
    }

    /// A line fit to put in front of a user.
    public var summary: String {
        if isLexicalOnly {
            return "Keyword search only — the on-device model is unavailable."
        }
        if isComplete {
            return "Searched all \(liveChunks) indexed passages."
        }
        let percent = Saturating.int(clamping: (semanticCoverage * 100).rounded())
        return "Searched \(percent)% of \(liveChunks) passages by meaning; \(awaitingReindex) still re-indexing."
    }
}

public struct QueryResult: Sendable, Equatable {
    public let hits: [ScoredChunk]
    public let completeness: Completeness
    /// The epoch the query itself was embedded in, or `nil` in lexical-only mode.
    public let epoch: EmbeddingEpoch?

    public init(hits: [ScoredChunk], completeness: Completeness, epoch: EmbeddingEpoch?) {
        self.hits = hits
        self.completeness = completeness
        self.epoch = epoch
    }
}
