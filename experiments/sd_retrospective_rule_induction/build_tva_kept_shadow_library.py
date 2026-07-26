#!/usr/bin/env python3
"""Build an eval-only rule library from consequence-kept TVA shadow updates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def consequence_kept(rule: dict[str, Any], min_abs_delta: float) -> bool:
    gate = rule.get("consequence_gate") or {}
    if not bool(gate.get("consequence_supported", False)):
        return False
    before = float(rule.get("confidence", 0.0) or 0.0)
    after = float(rule.get("shadow_confidence", before) or before)
    return abs(after - before) >= min_abs_delta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-library", type=Path, required=True)
    parser.add_argument("--gated-shadow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-abs-delta", type=float, default=1e-9)
    args = parser.parse_args()

    source = load_json(args.source_library)
    gated = load_json(args.gated_shadow)

    shadow_by_id = {rule.get("id"): rule for rule in gated.get("rules", []) if rule.get("id")}
    changed_ids: list[str] = []
    output_rules: list[dict[str, Any]] = []

    for rule in source.get("rules", []):
        rid = rule.get("id")
        out = dict(rule)
        shadow = shadow_by_id.get(rid)
        if shadow and consequence_kept(shadow, args.min_abs_delta):
            before = float(out.get("confidence", 0.0) or 0.0)
            after = float(shadow.get("shadow_confidence", before) or before)
            out["confidence_before_tva_kept_shadow_eval"] = before
            out["confidence"] = after
            out["effective_prior_scale"] = after
            out["tva_kept_shadow_eval_applied"] = True
            out["tva_kept_shadow_delta"] = after - before
            out["tva_kept_shadow_reason"] = shadow.get("shadow_update_reason")
            out["tva_kept_shadow_consequence_gate"] = shadow.get("consequence_gate")
            changed_ids.append(str(rid))
        else:
            out["tva_kept_shadow_eval_applied"] = False
        output_rules.append(out)

    payload = dict(source)
    payload["rules"] = output_rules
    payload["tva_kept_shadow_eval_metadata"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_library": str(args.source_library),
        "source_gated_shadow": str(args.gated_shadow),
        "num_rules": len(output_rules),
        "num_applied": len(changed_ids),
        "applied_rule_ids": changed_ids,
        "min_abs_delta": float(args.min_abs_delta),
        "note": "Eval-only library. Original live rule library is not modified.",
    }
    write_json(args.output, payload)
    print(f"wrote TVA-kept shadow eval library: {args.output}")
    print(f"applied rules: {len(changed_ids)}")
    for rid in changed_ids:
        print(f"- {rid}")


if __name__ == "__main__":
    main()
