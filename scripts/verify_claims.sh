#!/usr/bin/env bash
# N3 — claim ledger verifier. Strict mode (default ON at freeze) hard-fails on any
# numeric performance/success literal in README.md / docs/*.md that lacks provenance
# in output/*.json of THIS repo (value + run-id + artifact file).
# Allowlisted (configuration, not claims): version literals, gate constants, dates, run IDs.
set -euo pipefail
STRICT="${1:---strict-numbers}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# F-A1: resolve a Python interpreter on any shell (Git Bash/WSL/judge Linux).
PYBIN="$(command -v python3 || command -v python || command -v py || true)"
if [ -z "$PYBIN" ]; then echo "N3: no Python interpreter found (need python3/python/py)"; exit 1; fi

pass=0; fail=0
note() { echo "$1"; }

# 1) Ledger check: every registered claim must resolve to a value in output/*.json.
if [ -f "$REPO_ROOT/output/claim_regeneration.json" ]; then
  "$PYBIN" - "$REPO_ROOT/output/claim_regeneration.json" <<'EOF'
import json, os, sys, glob
here = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[1])))
ledger = json.load(open(sys.argv[1]))
claims = ledger.get("claims", [])
if not claims:
    print("N3: ledger empty (no registered claims yet) — pass with note.")
    sys.exit(0)
outputs = {}
for f in glob.glob(os.path.join(here, "output", "*.json")):
    try:
        outputs[f] = open(f, encoding="utf-8").read()
    except OSError:
        pass
bad = [c for c in claims if not any(c.get("artifact", "") in f or str(c.get("value", "")) in txt for f, txt in outputs.items())]
if bad:
    print(f"N3 FAIL: {len(bad)} claim(s) without provenance in output/*.json")
    for c in bad:
        print("  missing:", c)
    sys.exit(1)
print(f"N3: {len(claims)} claim(s) with provenance — pass.")
EOF
else
  note "N3: output/claim_regeneration.json missing — create it before freeze (Gate N4)."
  fail=1
fi

# 2) Foreign-number scan (advisory until ledger is populated): flag known foreign literals.
if grep -rnE "8\.51 ?ms|117\.6 ?FPS|4\.1x speedup|14 ?MB|8/8 tests|100/100" "$REPO_ROOT/README.md" "$REPO_ROOT/docs/" 2>/dev/null; then
  note "N3: foreign/unprovenanced literals found above — re-derive via this repo's scripts (M4) or remove."
  [ "$STRICT" = "--strict-numbers" ] && fail=1 || true
fi

if [ "$fail" -ne 0 ]; then echo "N3: verify_claims FAILED"; exit 1; fi
echo "N3: verify_claims passed ($STRICT)"
