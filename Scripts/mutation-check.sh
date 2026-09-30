#!/usr/bin/env bash
#
# Re-derive the mutation figure quoted in the README.
#
# The README claims a specific number of test failures from three deliberate
# invariant breaks. A number nobody can reproduce is an assertion, not evidence —
# and the count depends on exactly how each break is written — so the breaks are
# committed here rather than described in prose.
#
# Runs against a COPY; your working tree is never modified.
#
#   ./Scripts/mutation-check.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

cp -R "$ROOT" "$WORK/subject"
rm -rf "$WORK/subject/.build"
cd "$WORK/subject"

SRC="Sources/SemanticIndexSync"

# 1. The epoch gate stops gating: a vector from any embedding space is treated
#    as comparable. This is the failure the whole package exists to prevent.
perl -0pi -e 's/\&\& self\.epoch\.isComparable\(to: epoch\)/\&\& true \/* MUTANT 1 *\//' "$SRC/Embedding.swift"

# 2. Delete-wins becomes edit-wins: a concurrent edit resurrects a tombstone,
#    making deleted personal content searchable again on another device.
perl -0pi -e 's/let winnerIsLocal = local\.state\.isTombstone\n/let winnerIsLocal = !local.state.isTombstone \/* MUTANT 2 *\/\n/' "$SRC/Reconciler.swift"

# 3. The Okapi IDF smoothing term is removed, so a term present in every
#    document scores negative and silently inverts the ranking.
perl -0pi -e 's/\(containing \+ 0\.5\) \+ 1\.0/(containing + 0.5) \/* MUTANT 3 *\//' "$SRC/LexicalIndex.swift"

applied=$(grep -rc 'MUTANT' "$SRC" | grep -v ':0' | wc -l | tr -d ' ')
if [ "$applied" -ne 3 ]; then
  echo "Expected 3 mutated files, found $applied — the source has moved; update this script." >&2
  exit 1
fi
echo "Applied 3 mutations. Running the suite (all failures below are expected)..."
echo

set +e
swift test 2>&1 | tee "$WORK/out.log" | grep -E "Executed [0-9]+ tests"
set -e

echo
grep -oE "Executed [0-9]+ tests, with [0-9]+ failures" "$WORK/out.log" | tail -1
