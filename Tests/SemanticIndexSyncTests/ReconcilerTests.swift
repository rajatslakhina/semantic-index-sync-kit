import XCTest
@testable import SemanticIndexSync

/// A deliberately naive merge, written here so the real one can be measured
/// against it. This is the design the library exists to argue against: order
/// writes by wall-clock time and take the latest.
private struct LastWriterWinsReconciler {
    struct Stamped {
        let record: DocumentRecord
        let timestamp: Double
    }

    func resolve(local: Stamped, remote: Stamped) -> DocumentRecord {
        remote.timestamp >= local.timestamp ? remote.record : local.record
    }
}

final class ReconcilerTests: XCTestCase {

    private let phone = DeviceID("phone")
    private let laptop = DeviceID("laptop")
    private let note = DocumentID("note-1")

    // MARK: The headline property

    /// Feeds the same history through a deliberately broken implementation and
    /// asserts it produces the bug, then asserts the real implementation does
    /// not. Without the first half, the second half would pass against a merge
    /// that simply always returns a tombstone.
    func testLastWriterWinsResurrectsADeleteAndVersionVectorsDoNot() {
        // Both devices agree on the document.
        let shared = DocumentRecord.live(note, hash: ContentHash("v0"), by: phone)
        XCTAssertEqual(shared.version.counter(for: phone), 1)

        // The phone deletes it.
        let deletion = shared.deleted(by: phone)

        // The laptop, offline and unaware of the delete, edits the same document.
        let offlineEdit = shared.edited(to: ContentHash("v1"), by: laptop)

        // Neither write saw the other. This is the case LWW cannot represent.
        XCTAssertEqual(VersionVector.order(deletion.version, offlineEdit.version), .concurrent)

        // --- The broken implementation. The laptop reconnects later, so its
        // --- edit carries the later wall-clock stamp.
        let naive = LastWriterWinsReconciler()
        let naiveResult = naive.resolve(
            local: .init(record: deletion, timestamp: 100),
            remote: .init(record: offlineEdit, timestamp: 140)
        )
        XCTAssertFalse(
            naiveResult.state.isTombstone,
            "The naive merge is supposed to exhibit the bug; if it does not, this test proves nothing."
        )

        // --- The real implementation.
        let reconciler = Reconciler()
        let (resolved, outcome) = reconciler.resolve(local: deletion, remote: offlineEdit)
        XCTAssertTrue(resolved.state.isTombstone)
        XCTAssertEqual(outcome, .resolvedConcurrent(winner: .local, rule: .deleteWins))

        // And the merged version has seen both writes, so neither side replays.
        XCTAssertTrue(resolved.version.dominatesOrEquals(deletion.version))
        XCTAssertTrue(resolved.version.dominatesOrEquals(offlineEdit.version))
    }

    /// The delete must also win when it arrives as the *remote* side, or the
    /// rule is an accident of argument order rather than a rule.
    func testDeleteWinsRegardlessOfWhichSideItArrivesOn() {
        let shared = DocumentRecord.live(note, hash: ContentHash("v0"), by: phone)
        let deletion = shared.deleted(by: phone)
        let offlineEdit = shared.edited(to: ContentHash("v1"), by: laptop)

        let reconciler = Reconciler()
        let (asRemote, outcomeA) = reconciler.resolve(local: offlineEdit, remote: deletion)
        XCTAssertTrue(asRemote.state.isTombstone)
        XCTAssertEqual(outcomeA, .resolvedConcurrent(winner: .remote, rule: .deleteWins))

        let (asLocal, outcomeB) = reconciler.resolve(local: deletion, remote: offlineEdit)
        XCTAssertTrue(asLocal.state.isTombstone)
        XCTAssertEqual(outcomeB, .resolvedConcurrent(winner: .local, rule: .deleteWins))

        XCTAssertEqual(asRemote.state, asLocal.state)
        XCTAssertEqual(asRemote.version, asLocal.version)
    }

    // MARK: Ordering

    func testStaleRemoteIsIgnoredRatherThanApplied() {
        let first = DocumentRecord.live(note, hash: ContentHash("v0"), by: phone)
        let second = first.edited(to: ContentHash("v1"), by: phone)

        var manifest = [note: second]
        let report = Reconciler().apply(remote: [first], into: &manifest)

        XCTAssertEqual(report.staleRemotesIgnored, 1)
        XCTAssertEqual(report.supersededByRemote, 0)
        XCTAssertEqual(manifest[note]?.state, .live(ContentHash("v1")))
        XCTAssertTrue(report.invalidated.isEmpty, "A rejected write must not trigger a re-embed.")
    }

    func testNewerRemoteSupersedesAndInvalidates() {
        let first = DocumentRecord.live(note, hash: ContentHash("v0"), by: phone)
        let second = first.edited(to: ContentHash("v1"), by: laptop)

        var manifest = [note: first]
        let report = Reconciler().apply(remote: [second], into: &manifest)

        XCTAssertEqual(report.supersededByRemote, 1)
        XCTAssertEqual(report.invalidated, [note])
        XCTAssertEqual(manifest[note]?.state, .live(ContentHash("v1")))
    }

    func testIdenticalRemoteIsANoOpAndDoesNotRequeue() {
        let record = DocumentRecord.live(note, hash: ContentHash("v0"), by: phone)
        var manifest = [note: record]
        let report = Reconciler().apply(remote: [record], into: &manifest)

        XCTAssertEqual(report.identical, 1)
        XCTAssertTrue(report.invalidated.isEmpty)
    }

    func testUnknownDocumentIsInsertedAndQueuedForEmbedding() {
        let record = DocumentRecord.live(note, hash: ContentHash("v0"), by: laptop)
        var manifest: [DocumentID: DocumentRecord] = [:]
        let report = Reconciler().apply(remote: [record], into: &manifest)

        XCTAssertEqual(report.inserted, 1)
        XCTAssertEqual(report.invalidated, [note])
    }

    func testUnknownTombstoneIsInsertedWithoutQueueingWork() {
        let tombstone = DocumentRecord(
            id: note,
            state: .tombstone,
            version: VersionVector([laptop: 2]),
            lastWriter: laptop
        )
        var manifest: [DocumentID: DocumentRecord] = [:]
        let report = Reconciler().apply(remote: [tombstone], into: &manifest)

        XCTAssertEqual(report.inserted, 1)
        XCTAssertTrue(report.invalidated.isEmpty, "A tombstone has no content to embed.")
    }

    // MARK: Convergence

    /// Eventual consistency in one sentence: two replicas that have seen the same
    /// set of writes must hold the same state, regardless of the order the writes
    /// arrived in. Exercised over every permutation of a fixed write set, with no
    /// randomness, so a failure is reproducible.
    func testReplicasConvergeUnderEveryDeliveryOrder() {
        let base = DocumentRecord.live(note, hash: ContentHash("v0"), by: phone)
        let writes: [DocumentRecord] = [
            base.edited(to: ContentHash("v1"), by: phone),
            base.edited(to: ContentHash("v2"), by: laptop),
            base.deleted(by: DeviceID("tablet")),
            base.edited(to: ContentHash("v3"), by: DeviceID("watch")),
        ]

        let reconciler = Reconciler()
        var seen: [DocumentRecord] = []

        for permutation in Self.permutations(of: writes) {
            var manifest: [DocumentID: DocumentRecord] = [:]
            for write in permutation {
                _ = reconciler.apply(remote: [write], into: &manifest)
            }
            guard let final = manifest[note] else {
                XCTFail("Manifest lost the document entirely.")
                return
            }
            seen.append(final)
        }

        XCTAssertEqual(seen.count, 24, "4! orderings expected.")
        guard let reference = seen.first else { return XCTFail("No orderings ran.") }
        for (index, outcome) in seen.enumerated() {
            XCTAssertEqual(outcome.state, reference.state, "Ordering \(index) diverged in state.")
            XCTAssertEqual(outcome.version, reference.version, "Ordering \(index) diverged in version.")
        }
        // With a delete in the write set, delete-wins makes the converged state
        // a tombstone whatever the order.
        XCTAssertTrue(reference.state.isTombstone)
    }

    /// Concurrent live edits converge on a rule that is a pure function of the
    /// records — no timestamps, no arrival order.
    func testConcurrentLiveEditsConvergeByContentHash() {
        let base = DocumentRecord.live(note, hash: ContentHash("v0"), by: phone)
        let left = base.edited(to: ContentHash("aaa"), by: phone)
        let right = base.edited(to: ContentHash("zzz"), by: laptop)

        let reconciler = Reconciler()
        let (forward, forwardOutcome) = reconciler.resolve(local: left, remote: right)
        let (reverse, reverseOutcome) = reconciler.resolve(local: right, remote: left)

        XCTAssertEqual(forward.state, .live(ContentHash("zzz")))
        XCTAssertEqual(reverse.state, .live(ContentHash("zzz")))
        XCTAssertEqual(forward.version, reverse.version)
        XCTAssertEqual(forwardOutcome, .resolvedConcurrent(winner: .remote, rule: .higherContentHash))
        XCTAssertEqual(reverseOutcome, .resolvedConcurrent(winner: .local, rule: .higherContentHash))
    }

    func testConcurrentIdenticalEditsBreakTheTieOnWriterIdentity() {
        let base = DocumentRecord.live(note, hash: ContentHash("v0"), by: phone)
        let left = base.edited(to: ContentHash("same"), by: DeviceID("aaa-device"))
        let right = base.edited(to: ContentHash("same"), by: DeviceID("zzz-device"))

        let reconciler = Reconciler()
        let (forward, outcome) = reconciler.resolve(local: left, remote: right)
        let (reverse, _) = reconciler.resolve(local: right, remote: left)

        XCTAssertEqual(forward.lastWriter, DeviceID("aaa-device"))
        XCTAssertEqual(reverse.lastWriter, DeviceID("aaa-device"))
        XCTAssertEqual(outcome, .resolvedConcurrent(winner: .local, rule: .writerIdentity))
    }

    func testReportCountsAreConsistentWithInputSize() {
        let a = DocumentRecord.live(DocumentID("a"), hash: ContentHash("1"), by: phone)
        let b = DocumentRecord.live(DocumentID("b"), hash: ContentHash("1"), by: phone)
        var manifest = [a.id: a]
        let report = Reconciler().apply(remote: [a, b], into: &manifest)
        XCTAssertEqual(report.totalConsidered, 2)
        XCTAssertEqual(report.identical, 1)
        XCTAssertEqual(report.inserted, 1)
    }

    // MARK: Helpers

    private static func permutations<T>(of items: [T]) -> [[T]] {
        guard items.count > 1 else { return [items] }
        var output: [[T]] = []
        for index in items.indices {
            var rest = items
            let element = rest.remove(at: index)
            for tail in permutations(of: rest) {
                output.append([element] + tail)
            }
        }
        return output
    }
}
