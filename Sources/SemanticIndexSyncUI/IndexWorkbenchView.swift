#if canImport(SwiftUI)
import SwiftUI
import SemanticIndexSync

/// The demo surface: a semantic index you can watch go partially blind during an
/// embedding-model migration, and recover one budgeted background pass at a time.
public struct IndexWorkbenchView: View {

    @State private var model: IndexWorkbenchModel
    @State private var queryText: String

    public init(configuration: WorkbenchConfiguration) {
        _model = State(initialValue: IndexWorkbenchModel(configuration: configuration))
        _queryText = State(initialValue: configuration.initialQuery)
    }

    public var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    coverageCard
                    searchCard
                    epochCard
                    syncCard
                    logCard
                }
                .padding(16)
            }
            .navigationTitle("Semantic Index Sync")
            .background(Color.secondaryBackground)
        }
        .task { await model.start() }
    }

    // MARK: Coverage

    private var coverageCard: some View {
        Card(title: "Search coverage") {
            let completeness = model.result?.completeness
            let coverage = completeness?.semanticCoverage ?? 1.0

            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .firstTextBaseline) {
                    Text(percentText(coverage))
                        .font(.system(size: 40, weight: .semibold, design: .rounded))
                        .monospacedDigit()
                    Spacer()
                    StatusPill(
                        text: completeness?.isComplete == true ? "Complete" : "Degraded",
                        tint: completeness?.isComplete == true ? .green : .orange
                    )
                }

                CoverageBar(fraction: coverage, isComplete: completeness?.isComplete ?? true)

                Text(completeness?.summary ?? "Preparing the index…")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                if let progress = model.progress, !progress.isFinished {
                    Divider()
                    HStack {
                        Text("Re-index queue")
                            .font(.subheadline.weight(.medium))
                        Spacer()
                        Text("\(progress.completed)/\(progress.total)")
                            .font(.subheadline.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                    ProgressView(value: progress.fraction)
                }
            }
        }
    }

    // MARK: Search

    private var searchCard: some View {
        Card(title: "Query") {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 8) {
                    TextField("Search your notes", text: $queryText)
                        .textFieldStyle(.roundedBorder)
                        .submitLabel(.search)
                        .onSubmit { runQuery() }
                    Button("Search", action: runQuery)
                        .buttonStyle(.borderedProminent)
                        .disabled(model.isWorking)
                }

                let hits = model.result?.hits ?? []
                if hits.isEmpty {
                    Text(model.isWorking ? "Working…" : "No passages matched.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(hits) { hit in
                        HitRow(hit: hit)
                    }
                }
            }
        }
    }

    private func runQuery() {
        model.setQuery(queryText)
        Task { await model.runQuery() }
    }

    // MARK: Epoch and migration

    private var epochCard: some View {
        Card(title: "Embedding epoch") {
            VStack(alignment: .leading, spacing: 12) {
                LabeledContent("Active space") {
                    Text(model.activeEpochLabel)
                        .font(.footnote.monospaced())
                        .foregroundStyle(.secondary)
                }

                if !model.epochCounts.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(model.epochCounts, id: \.epoch) { entry in
                            HStack {
                                Circle()
                                    .fill(entry.epoch == model.activeEpochLabel ? Color.accentColor : Color.secondary)
                                    .frame(width: 7, height: 7)
                                Text(entry.epoch)
                                    .font(.caption.monospaced())
                                Spacer()
                                Text("\(entry.count) vectors")
                                    .font(.caption.monospacedDigit())
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    .padding(10)
                    .background(Color.tertiaryBackground, in: RoundedRectangle(cornerRadius: 8))
                }

                Button {
                    Task { await model.shipModelUpdate() }
                } label: {
                    Label("Ship an OS model revision", systemImage: "arrow.up.circle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(model.hasUpgraded || model.isWorking)

                Divider()

                Text("Background pass conditions")
                    .font(.subheadline.weight(.medium))

                Picker("Thermal state", selection: Binding(
                    get: { model.thermalState },
                    set: { model.thermalState = $0 }
                )) {
                    ForEach(ThermalState.allCases, id: \.self) { state in
                        Text(state.description.capitalized).tag(state)
                    }
                }
                .pickerStyle(.segmented)

                Toggle("Low Power Mode", isOn: Binding(
                    get: { model.isLowPowerModeEnabled },
                    set: { model.isLowPowerModeEnabled = $0 }
                ))
                .font(.subheadline)

                Button {
                    Task { await model.runBackgroundPass() }
                } label: {
                    Label("Run one background pass", systemImage: "play.circle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.isWorking)
            }
        }
    }

    // MARK: Sync

    private var syncCard: some View {
        Card(title: "Peer sync") {
            VStack(alignment: .leading, spacing: 10) {
                Text("Applies a batch containing a delete performed offline on the peer and a concurrent edit. A clock-based merge resurrects the delete; the version-vector merge does not.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                Button {
                    Task { await model.syncFromPeer() }
                } label: {
                    Label("Sync from peer device", systemImage: "arrow.triangle.2.circlepath")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(model.isWorking)
            }
        }
    }

    // MARK: Log

    private var logCard: some View {
        Card(title: "Activity") {
            if model.log.isEmpty {
                Text("Nothing yet.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else {
                VStack(alignment: .leading, spacing: 10) {
                    ForEach(model.log) { entry in
                        HStack(alignment: .top, spacing: 8) {
                            Circle()
                                .fill(tint(for: entry.kind))
                                .frame(width: 7, height: 7)
                                .padding(.top, 5)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(entry.title).font(.subheadline.weight(.medium))
                                Text(entry.detail)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }
                    }
                }
            }
        }
    }

    private func tint(for kind: ActivityEntry.Kind) -> Color {
        switch kind {
        case .info: return .accentColor
        case .success: return .green
        case .warning: return .orange
        }
    }

    private func percentText(_ fraction: Double) -> String {
        let clamped = fraction.isFinite ? min(max(fraction, 0), 1) : 0
        let percent = Int((clamped * 100).rounded())
        return "\(percent)%"
    }
}

// MARK: - Components

private struct Card<Content: View>: View {
    let title: String
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.headline)
            content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(Color.cardBackground, in: RoundedRectangle(cornerRadius: 14))
    }
}

private struct CoverageBar: View {
    let fraction: Double
    let isComplete: Bool

    var body: some View {
        GeometryReader { proxy in
            let clamped = fraction.isFinite ? min(max(fraction, 0), 1) : 0
            let width = max(0, proxy.size.width) * clamped
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 5)
                    .fill(Color.tertiaryBackground)
                RoundedRectangle(cornerRadius: 5)
                    .fill(isComplete ? Color.green : Color.orange)
                    .frame(width: width)
            }
        }
        .frame(height: 10)
        .animation(.easeOut(duration: 0.25), value: fraction)
    }
}

private struct StatusPill: View {
    let text: String
    let tint: Color

    var body: some View {
        Text(text)
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(tint.opacity(0.16), in: Capsule())
            .foregroundStyle(tint)
    }
}

private struct HitRow: View {
    let hit: ScoredChunk

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text(hit.chunk.id.description)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                Spacer()
                Text(sourceLabel)
                    .font(.caption2.weight(.semibold))
                    .padding(.horizontal, 7)
                    .padding(.vertical, 2)
                    .background(sourceTint.opacity(0.16), in: Capsule())
                    .foregroundStyle(sourceTint)
                Text(String(format: "%.3f", hit.score))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Text(hit.chunk.text)
                .font(.footnote)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(Color.tertiaryBackground, in: RoundedRectangle(cornerRadius: 9))
    }

    private var sourceLabel: String {
        switch hit.source {
        case .semantic: return "MEANING"
        case .hybrid: return "HYBRID"
        case .lexicalFallback: return "KEYWORD"
        }
    }

    private var sourceTint: Color {
        switch hit.source {
        case .semantic: return .accentColor
        case .hybrid: return .purple
        case .lexicalFallback: return .orange
        }
    }
}

// MARK: - Platform colours

extension Color {
    /// Platform-appropriate greys, resolved per platform so the demo renders
    /// correctly in both light and dark appearance on iOS and macOS.
    static var secondaryBackground: Color {
        #if os(iOS)
        Color(uiColor: .secondarySystemBackground)
        #else
        Color(nsColor: .windowBackgroundColor)
        #endif
    }

    static var cardBackground: Color {
        #if os(iOS)
        Color(uiColor: .systemBackground)
        #else
        Color(nsColor: .controlBackgroundColor)
        #endif
    }

    static var tertiaryBackground: Color {
        #if os(iOS)
        Color(uiColor: .tertiarySystemBackground)
        #else
        Color(nsColor: .underPageBackgroundColor)
        #endif
    }
}
#endif
