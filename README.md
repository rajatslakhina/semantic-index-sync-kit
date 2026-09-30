# SemanticIndexSync

**Your on-device semantic search does not break when the embedding model is updated. It gets *quietly worse*, returns confident answers, and throws nothing.**

A cosine similarity between a query vector from model revision 4 and a stored vector from revision 3 is a perfectly well-formed number. It is finite. It sorts. It renders in a list. It is also noise — and there is no exception, no log line and no crash report to tell you it happened. The user just finds that the search box which answered their question last week now returns the wrong three notes, and they conclude the feature is bad.

`SemanticIndexSync` is the layer that makes that failure impossible to express: an incremental, offline-first index where **an embedding space has an identity**, vectors are only ever compared within it, and the price of a model update is a *visible, budgeted, resumable* re-index rather than a silent loss of quality.

It also answers the second half of the problem, which shows up the moment the user has more than one device: how a delete performed offline on the phone is not undone by an edit that arrives later from the laptop.

- **Library repo:** you are here.
- **Demo app:** [**semantic-index-sync-kit-demo**](https://github.com/rajatslakhina/semantic-index-sync-kit-demo) — a separate Xcode project that consumes this package over the network at a released version, exactly as a real client would.

---

## Why this matters

iOS 27's **Foundation Models** framework — whose public `LanguageModel` / `LanguageModelExecutor` provider protocols let any conforming package back a `LanguageModelSession`, with Core AI models runnable *through* such a session — made "search and answer over the user's own data, on the phone" a realistic feature to ship. (Those protocols supply text *generation*; the embeddings this library indexes come from whatever provider the app injects.) The model is the easy part — it is two lines and Apple maintains it. The system underneath it is the part that gets a team paged, and it has four properties that a tutorial never mentions:

1. **The embedding space is a versioned dependency you do not control.** An OS update reships the model. The user toggles Apple Intelligence off and on. A provider package changes. Every one of those invalidates every vector on disk, and none of them raise an error.
2. **Re-embedding a corpus is expensive and the device is not yours.** Fifty thousand chunks cannot be re-embedded in a foreground burst. It has to happen across BGTask windows, under thermal and battery budgets, resumably, without the user noticing anything except that search briefly says less.
3. **Deletes are load-bearing.** This is an index of personal content. A merge rule that occasionally resurrects a deleted document has not lost an edit — it has made something the user removed findable again.
4. **Degradation has to be legible.** A search that can only see 60% of the corpus must say so. Silence is the failure mode that makes users stop trusting the feature permanently.

This package is the data plane for all four. It contains **no model, no weights and no ML dependency at all** — which is the design decision the rest of the correctness story rests on. See *Rejected alternatives*.

---

## The core idea: epoch as a comparability token

```swift
public struct EmbeddingEpoch: Hashable, Sendable, Codable {
    public let modelIdentifier: String   // which model
    public let revision: Int             // which weights
    public let dimension: Int            // which shape
}
```

Every `StoredVector` carries the epoch it was produced in *and* the content hash of the text it describes. It is usable only if both still hold:

```swift
public func isFresh(for chunk: Chunk, in epoch: EmbeddingEpoch) -> Bool {
    self.chunk == chunk.id
        && self.epoch.isComparable(to: epoch)   // the model moved
        && self.sourceHash == chunk.sourceHash  // the text moved
        && self.values.count == epoch.dimension
}
```

`IndexCoordinator.search` skips anything that fails this check. Not down-weights it, not scales it — **skips it**, because there is no defensible rule for "close enough" between two embedding spaces. A skipped chunk is still reachable on the keyword path, and it is counted:

```swift
public struct Completeness {
    public let liveChunks: Int
    public let semanticallyCovered: Int
    public let awaitingReindex: Int
    public let isLexicalOnly: Bool

    public var semanticCoverage: Double   // guarded; an empty index is 1.0, not NaN
    public var isComplete: Bool
    public var summary: String            // "Searched 60% of 10 passages by meaning; 4 still re-indexing."
}
```

`Completeness` is a **required field of every `QueryResult`**, not an optional diagnostic. That is deliberate: it makes "the index went shallow and said nothing" un-writable, because the caller has to receive the number whether it looks at it or not.

---

## What is in it

| Type | Responsibility |
|---|---|
| `EmbeddingEpoch` | Identity of an embedding space. Comparability is identity, never approximation. |
| `StoredVector` | A vector plus the two facts that decide whether it may still be used. |
| `VersionVector` / `CausalOrder` | Partial order over document revisions. Distinguishes *stale* from *concurrent* — the distinction wall-clock ordering throws away. |
| `Reconciler` | Pure, synchronous merge. Three total tie-break rules, every decision typed and explained. |
| `WorkBudget` / `DeviceConditions` | Thermal, battery, low-power and external-power admission control. Refusals are **typed reasons**, not a bare `false`. |
| `IndexCoordinator` | The actor that owns manifest, chunks, vectors and the migration queue together. |
| `LexicalIndex` | Smoothed BM25. The floor the system stands on when the vector path cannot answer. |
| `Completeness` / `QueryResult` | The honesty contract described above. |
| `Saturating` | Every trapping `Int` operation in the package, funnelled through one audited helper. |

The package also ships a second product, **`SemanticIndexSyncUI`**: `IndexWorkbenchView` (the SwiftUI surface), `IndexWorkbenchModel` (a deliberately thin view model that sequences calls into the coordinator and owns none of the rules), `WorkbenchConfiguration` (what the *host app* decides — corpus, budgets, which model identifiers stand in for the OS model) and `OfflineEditScenario` (a pure builder for a genuinely concurrent pair of histories). The companion demo app is the host.

---

## Design decisions, and what was rejected

### 1. Version vectors, not last-writer-wins

**Decision.** Each document carries a per-device counter map. Merging compares them as a partial order: `identical`, `ancestor`, `descendant`, or `concurrent`.

**Rejected: timestamp LWW.** It is `O(1)` instead of `O(devices)` and one line to implement. It is also unable to represent the case that matters. Device clocks disagree, users change them, and an offline device rejoining after two days carries writes stamped in the past. The package's own test suite demonstrates the bug rather than asserting around it — `testLastWriterWinsResurrectsADeleteAndVersionVectorsDoNot` implements the naive reconciler inline, **asserts it produces the resurrection**, and only then asserts the real one does not. A test that cannot fail against the bug it exists to catch proves nothing.

**Cost accepted.** The vector is `O(devices)` per document. For a personal index — single-digit devices — that is a few dozen bytes. It would be the wrong trade at fleet scale with unbounded writers, and that is documented as a limit rather than left for a reader to discover.

### 2. Delete wins a concurrent merge

**Decision.** When an edit and a delete are genuinely concurrent, the delete wins.

**Rejected: edit wins / "most content wins".** This loses a user's edit, which is a real cost, openly stated. The alternative loses a *delete*. In a searchable index of personal content that means material the user removed becomes findable again on another device. Between "your edit needs redoing" and "your deleted note came back", only the first is recoverable by the user — so the rule is asymmetric on purpose.

Concurrent *live* edits fall through to higher content hash, then writer identity. Both are arbitrary; both are **pure functions of the two records**, which is the actual requirement — every replica has to reach the same answer without talking to anyone. `testReplicasConvergeUnderEveryDeliveryOrder` exercises all 24 orderings of a four-write history and asserts a single converged state.

### 3. Old vectors are retained across a model bump, not deleted

**Decision.** Vectors are keyed by **chunk *and* epoch**, so `adopt(provider:)` keeps the previous space on disk beside the new one. `discardVectors(outside:)` reclaims it later, on the app's schedule, and re-queues anything it orphaned.

**Rejected: keying by chunk alone.** One dictionary key shorter, and it silently makes this whole decision a dead letter — each new vector would overwrite its predecessor, so after a migration there would be nothing left to roll back to and nothing left to reclaim. Retention costs disk and buys nothing for *scoring* (cross-epoch vectors are never compared), but `testPreviousEpochVectorsAreRetainedSoRollbackCostsNothing` asserts what it does buy: rolling back queues **zero** work instead of triggering a second full re-index. The demo app's "roll the model revision back" button is that assertion made visible.

**Rejected: delete on bump.** Simpler still, and it drops the index to zero semantic coverage the instant a staged OS rollout lands, with no way back but a full re-embed.

### 4. Vectors never sync; content and manifest do

**Decision.** `applyRemote` accepts manifest rows and passages. It does not accept vectors.

**Rejected: shipping vectors with the data.** It looks like a pure win — embed once, use everywhere. It is wrong the moment two devices sit on different OS versions, which during a staged rollout is *most* of them: the receiving device would import vectors from an epoch it cannot query and either store them uselessly or, worse, compare them. Recomputing per device costs energy once per device and removes the entire class of bug.

### 5. One actor, not a lock per table

**Decision.** `IndexCoordinator` owns manifest, chunks, vectors and queue together.

**Rationale.** Every invariant here spans two tables — "no vector survives its document's tombstone", "the queue holds exactly the stale chunks". Split the state and those hold only between lock acquisitions. The cost is that the genuinely slow part (embedding) happens across an `await` inside a reentrant actor. Two separate things follow from that, and getting only the first is a trap worth naming:

1. `drainMigration` **re-validates every result against current state after the suspension point**, discarding anything whose chunk was since deleted, edited, or overtaken by a second epoch bump.
2. It also **claims its batch before suspending** (`inFlight`). Without that, a second pass entering while the first is parked takes the *same* prefix of the queue and embeds the same chunks again, burning exactly the battery budget `WorkBudget` exists to conserve. An overlapping pass gets `.alreadyDraining` and touches nothing.
3. And `MigrationProgress` is **derived from current state, never accumulated**. A running "chunks embedded since the last `adopt`" counter drifts away from the index the moment a chunk is deleted or rewritten after being embedded — the count keeps the work, the corpus loses the chunk, and `total` grows past the number of passages that exist, which is precisely the fabricated denominator a progress bar would then render. Computing coverage from the index makes `completed + remaining == liveChunks.count` true by construction; the concurrency suite asserts it after racing 24 upserts, 8 drains, 8 deletes and 5 remote merges.

There is a fourth, smaller lesson in the same method: a mid-flight epoch bump returns its own `.abandoned(supersededBy:)` case rather than borrowing `.deferred(.zeroAllowance)`. Borrowing was tempting and wrong — the budget *admitted* that pass, and the queue was *not* left untouched — and it would have put a sentence in the support log that is false in both halves.

Five tests park a provider mid-`embed` using a continuation rendezvous — there is no `Task.sleep` anywhere in the suite — and assert both properties. The overlapping-pass case is asserted against `availableBatch` *while* the first pass is parked, rather than by starting a second `drainMigration`: without the guard that second call would park on the same provider and the test would **hang** instead of failing, and a test that hangs under a regression is not a test that catches it.

### 6. No model, by design

**Decision.** `EmbeddingProvider` is a protocol; the package ships `DeterministicEmbeddingProvider`, a feature-hashing stand-in with zero ML dependencies.

**Rationale.** The correctness of this system does not depend on the *quality* of its vectors, only on their epoch discipline — so the entire thing is testable, and CI-runnable on Linux, with no weights present. The stand-in hashes with an explicit FNV-1a rather than Swift's `Hasher`, whose seed is randomised per process: a vector persisted today must compare identically tomorrow. `StableHashTests` pins **hardcoded reference constants** for that reason. The tempting version of that test — hash twice in one process and assert they agree — passes for `Hasher` too, which is precisely the bug it exists to catch.

---

## Crash-safety posture

This is a package that runs in a background task where a trap is a silent failure with no log line, so:

- **No force-unwraps.** Not "few" — the `!` operator does not appear as a postfix unwrap anywhere in `Sources/`.
- **Every trapping `Int` operation routes through `Saturating`** — in `Sources/` and in the SwiftUI layer alike: `+`/`-`/`*` overflow, `/` and `%` by zero, `Int.min / -1`, and `Int(Double)` for NaN, infinity and out-of-range. Bounds are derived from `Int.max`, never a 64-bit literal, so the package is correct where `Int` is 32-bit. (The one exception is deliberate and local: `VersionVector`'s per-device counter is a `UInt64` and saturates with `&+` at its own call site, since `Saturating` is an `Int` helper.)
- **Every collection access is bounds-checked**, including the embedder's hash-bucket write and the drain loop's index into the provider's response.
- **Degenerate float inputs are defined**: a zero vector normalises to zero rather than NaN; mismatched vector lengths score `0` rather than indexing out of range; non-finite inputs are filtered rather than propagated.
- **The unsmoothed-IDF trap is closed**: Okapi IDF goes negative for a term present in every document, silently inverting ranking. `testUnsmoothedIDFWouldInvertRankingAndTheSmoothedOneDoesNot` computes the unsmoothed formula inline, asserts it is negative, and only then asserts the shipped one is not.

---

## Testing

98 XCTest cases across two test targets. The suite is written against a specific standard: **a test that would still pass if the implementation were gutted is worse than no test**, because it reads like coverage. Concretely:

- For the two properties this README makes the loudest claims about, a **deliberately broken implementation is fed in and asserted to fail**:
  - `ReconcilerTests` implements `LastWriterWinsReconciler` inline and asserts it resurrects the tombstone before asserting the real reconciler does not.
  - `EpochGateTests` scores a *different* query vector — what a retrained revision of the same model would produce — against the stale one, asserts the result is **above 0.9** first (*i.e. that a naive implementation would rank it at the very top*), and only then asserts the gate rejects it anyway. Asserting `cos(v, v) == 1` instead would be a property of the cosine function, true against every implementation, and would prove nothing. (It builds those vectors explicitly rather than using this package's own embedder, which salts tokens with the revision and so makes the two spaces exactly orthogonal — a cross-epoch cosine of 0, which a naive implementation would *not* rank and which would therefore prove nothing. Real models are not orthogonal across revisions; that is why the failure is invisible.)
- **No self-fulfilling assertions.** Exact expected values, not bounds the implementation satisfies by construction — migration coverage is checked as `4 → 0 → 2 → 4`, not "greater than zero"; `discardVectors` is checked as "exactly 4 reclaimed and exactly 4 re-queued", not "≥ 0"; the embedder is pinned to its literal output vector rather than compared against a second call in the same process (which would pass for `Hasher` too, the very function that must not be used here).
- **Concurrency tests have real concurrent writers.** `ConcurrentWriterTests` races 24 upserts, 8 drains, 8 deletes and 5 remote merges against one coordinator, then runs a cross-table invariant audit. The reentrancy tests synchronise with a continuation rendezvous rather than a sleep, so the interleaving under test is the one requested on every run; the suite's one `Task.sleep` is a failure deadline, never a wait-and-assume.
- **Mutation-checked, reproducibly.** `Scripts/mutation-check.sh` applies three deliberate invariant breaks to a *copy* of the tree — the epoch gate stops gating, delete-wins becomes edit-wins, the IDF smoothing term is removed — and runs the suite. It reports **10 failures**. The exact patches are committed rather than described, because the count depends on how each break is written, and a number nobody can re-derive is an assertion rather than evidence. Run it yourself.

---

## Using it

```swift
.package(url: "https://github.com/rajatslakhina/semantic-index-sync-kit.git", from: "2.0.0")
```

```swift
import SemanticIndexSync

let coordinator = IndexCoordinator(
    device: DeviceID("iphone-15"),
    provider: myFoundationModelsProvider   // any EmbeddingProvider
)

await coordinator.upsert(
    document: DocumentID("note-42"),
    passages: chunker.chunks(of: note.body),
    hash: ContentHash(note.revisionFingerprint)
)

// Background task: one budgeted, resumable pass.
switch await coordinator.drainMigration(budget: .background, conditions: currentConditions) {
case .deferred(let why):   logger.info("re-index deferred: \(why)")   // typed reason
case .progressed(let p):   logger.info("\(p.completed)/\(p.total)")
case .finished:            scheduler.cancelReindexTask()
case .providerFailed, .idle: break
}

// Query: the coverage number comes back whether you look at it or not.
let result = await coordinator.search(userQuery, limit: 10)
searchView.render(result.hits, footnote: result.completeness.summary)
```

When the OS ships a new model revision:

```swift
let progress = await coordinator.adopt(provider: updatedProvider)
// progress.remaining passages are now outside the query's embedding space.
// Search keeps working on the keyword floor; coverage climbs per pass.
```

---

## Limits, stated plainly

- **Brute-force vector scan.** Every live chunk with a fresh vector is scored per query. Fine for a personal corpus in the low tens of thousands of chunks; an ANN index is the next layer and is deliberately out of scope here, because it does not interact with the epoch or merge problems this package is about.
- **Version vectors grow with device count.** Correct for a personal index, wrong for unbounded writers. No entry pruning is implemented.
- **Tombstones are never garbage-collected.** A real deployment needs a collection rule tied to a "every device has seen this" watermark.
- **The chunker is the caller's.** So is the content hash — hashing strategy is an application decision, and baking one in would add a crypto dependency the package does not otherwise need.

---

## Verification

Run it yourself:

```bash
swift build -Xswiftc -warnings-as-errors   # zero warnings, enforced
swift test                                  # 98 tests
./Scripts/mutation-check.sh                 # re-derive the mutation figure
```

**What was actually verified for this release**, stated exactly:

- **Clean build, zero warnings.** `rm -rf .build && swift build -Xswiftc -warnings-as-errors` with **Swift 6.1.2 on Linux (x86_64)** — a *clean* build, because an incremental one compiles nothing and still prints `Build complete!`.
- **98 of 98 tests passing** under `swift test` on that toolchain.
- **Mutation check:** `Scripts/mutation-check.sh` → 10 failures, run before this release and reproducible on any machine with the toolchain.
- **CI** runs the same clean build and test on Linux and on macOS, plus an iOS Simulator compile — and it passed on the `v2.0.0` commit, which is what establishes that `IndexWorkbenchView.swift` compiles at all (see below). Live status for every commit is on the [Actions tab](../../actions) — preferred over a run ID here, which goes stale on the next commit.

**What is *not* covered locally, stated rather than left to be discovered.** `IndexWorkbenchView.swift` is behind `#if canImport(SwiftUI)`, so the Linux build compiles it to nothing — it was never compiled on the machine that produced this release. The `ios-simulator` CI job is the only thing that compiles it, and on the `v2.0.0` commit that job passed. Nothing here has been *run*, though: no Simulator launch, no screenshots. Compiling is not launching, and the two are never treated as the same claim. Everything else in `SemanticIndexSyncUI` — the view model, the configuration and the scenario builder — is deliberately **not** behind that guard, so it is compiled and tested on Linux like the rest.

The [companion demo app's README](https://github.com/rajatslakhina/semantic-index-sync-kit-demo#verification--exactly-what-happened) states separately, and without conflation, whether that app was *built* for a Simulator and whether it was *run* on one. (It was built; it was not run.)

---

## License

MIT — see [LICENSE](LICENSE).
