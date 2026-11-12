"""Check declared comparability without changing historical experiments.

A matching declaration is necessary, not proof that a run used the declared data.
The private cohort/split hashes must also match when comparing new runs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def fingerprint(spec: Mapping) -> str:
    """Content identity, independent of JSON formatting."""
    return hashlib.sha256(
        json.dumps(dict(spec), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def compare_contracts(reference: Mapping, candidate: Mapping) -> list[str]:
    """Return every incompatible/missing field; only objective details may vary.

    Strict recursive key comparison prevents silently dropping the feature order,
    checkpoint rule, preprocessing or protocol settings from a declaration.
    Dataset and seed are part of a *cell* and must be equal for matched comparison.
    """
    differences = []

    def visit(a, b, path):
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            for key in sorted(set(a) | set(b)):
                here = f"{path}.{key}" if path else str(key)
                if not path and key in {"objective", "objective_config"}:
                    continue
                if key not in a or key not in b:
                    differences.append(f"{here}: missing or additional field")
                else:
                    visit(a[key], b[key], here)
        elif a != b:
            differences.append(f"{path}: {a!r} != {b!r}")

    visit(reference, candidate, "")
    return differences
