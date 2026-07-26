#!/usr/bin/env python3
"""Build a versioned offline rule library from v3 reliable and v4 substate rules."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import DEFAULT_RULES_V3, DEFAULT_RULE_LIBRARY, DEFAULT_SUBSTATE_RULES_V4, load_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reliable-rules", type=Path, default=DEFAULT_RULES_V3)
    parser.add_argument("--substate-rules", type=Path, default=DEFAULT_SUBSTATE_RULES_V4)
    parser.add_argument("--output", type=Path, default=DEFAULT_RULE_LIBRARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reliable = load_json(args.reliable_rules)
    substate = load_json(args.substate_rules)
    library = {
        "library_version": "offline_v4",
        "online_enabled": False,
        "status_meaning": {
            "reliable": "passed current offline consistency checks",
            "candidate": "substate candidate; requires replay and online validation",
            "cycle_candidate": "single-cycle near-threshold evidence; not eligible for online use",
            "cycle_supported": "passed single-cycle substate mining thresholds; requires cumulative validation",
            "needs_split": "coarse rule showed conflict or false-prefer risk",
        },
        "rules": [],
    }
    for rule in reliable.get("rules", []):
        out = dict(rule)
        out["status"] = "reliable"
        out["source"] = "v3"
        library["rules"].append(out)
    for rule in substate.get("rules", []):
        out = dict(rule)
        out["status"] = rule.get("status", "candidate")
        out["source"] = "v4_substate"
        library["rules"].append(out)
    write_json(args.output, library)
    print(f"wrote rule library: {args.output}")
    print(f"rules: {len(library['rules'])}")


if __name__ == "__main__":
    main()
