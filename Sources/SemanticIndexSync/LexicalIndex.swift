import Foundation

/// A small BM25 scorer over the live chunk set.
///
/// Its role here is not to compete with the vector path — it is the floor the
/// system stands on when the vector path cannot answer: no model installed, the
/// user turned on-device intelligence off, or a migration has not yet re-embedded
/// a chunk. Without it, an epoch bump would make the search box return nothing,
/// which is a far worse user-visible outcome than slightly weaker ranking.
public struct LexicalIndex: Sendable {

    private let k1: Double
    private let b: Double

    public init(k1: Double = 1.2, b: Double = 0.75) {
        self.k1 = k1.isFinite ? max(0, k1) : 1.2
        self.b = b.isFinite ? min(max(b, 0), 1) : 0.75
    }

    /// Score every chunk against the query. Chunks with no query term present
    /// are omitted rather than returned at zero.
    public func scores(for query: String, over chunks: [Chunk]) -> [ChunkID: Double] {
        let queryTerms = Set(Tokenizer.tokens(in: query))
        guard !queryTerms.isEmpty, !chunks.isEmpty else { return [:] }

        var termFrequencies: [ChunkID: [String: Int]] = [:]
        var lengths: [ChunkID: Int] = [:]
        var documentFrequency: [String: Int] = [:]
        var totalLength = 0

        for chunk in chunks {
            let tokens = Tokenizer.tokens(in: chunk.text)
            lengths[chunk.id] = tokens.count
            totalLength = Saturating.add(totalLength, tokens.count)

            var frequencies: [String: Int] = [:]
            for token in tokens {
                frequencies[token] = Saturating.add(frequencies[token] ?? 0, 1)
            }
            termFrequencies[chunk.id] = frequencies

            for term in queryTerms where frequencies[term] != nil {
                documentFrequency[term] = Saturating.add(documentFrequency[term] ?? 0, 1)
            }
        }

        let count = chunks.count
        // `count` is > 0 here and both operands are `Double`, so this cannot
        // trap. The `max(1,)` is belt-and-braces against a future refactor that
        // reaches this line with an empty corpus.
        let averageLength = Double(totalLength) / Double(max(1, count))

        var results: [ChunkID: Double] = [:]
        for chunk in chunks {
            let frequencies = termFrequencies[chunk.id] ?? [:]
            let length = Double(lengths[chunk.id] ?? 0)
            var score = 0.0

            for term in queryTerms {
                let frequency = Double(frequencies[term] ?? 0)
                guard frequency > 0 else { continue }
                let containing = Double(documentFrequency[term] ?? 0)
                // Okapi IDF with the +1 smoothing that keeps the log argument
                // above 1, so a term present in every chunk scores 0 rather than
                // negative — an unsmoothed IDF can go negative and invert ranking.
                let idfArgument = (Double(count) - containing + 0.5) / (containing + 0.5) + 1.0
                guard idfArgument > 0 else { continue }
                let idf = Foundation.log(idfArgument)
                let denominator = frequency + k1 * (1 - b + b * (length / max(averageLength, 1e-9)))
                guard denominator > 0, denominator.isFinite else { continue }
                let contribution = idf * (frequency * (k1 + 1)) / denominator
                guard contribution.isFinite else { continue }
                score += contribution
            }

            if score > 0 { results[chunk.id] = score }
        }
        return results
    }
}
