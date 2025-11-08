#!/usr/bin/env python3
"""Simple OpenSpec proposal generator.

Usage:
  python generate_proposal.py --title "Title" --prompt "Short prompt"

This script is intentionally dependency-free (stdlib only) so it can run on CI and dev machines
without installing extra packages.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime


def sanitize_title(title: str) -> str:
    # lower-case, replace non-alnum with hyphens, collapse hyphens
    s = title.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "proposal"


def next_id(specs_dir: str) -> str:
    # find numeric prefixes like 0001-...md
    if not os.path.isdir(specs_dir):
        os.makedirs(specs_dir, exist_ok=True)
        return "0001"
    ids = []
    for name in os.listdir(specs_dir):
        m = re.match(r"(0*\d+)-", name)
        if m:
            try:
                ids.append(int(m.group(1)))
            except Exception:
                pass
    if not ids:
        return "0001"
    return f"{max(ids) + 1:04d}"


def build_contents(id_: str, title: str, prompt: str, generated_by: str = "openspec-gen v0.1") -> str:
    now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return f"""---
id: {id_}
title: "{title}"
status: draft
generated_by: "{generated_by}"
generated_at: "{now}"
---

# Summary

{prompt}

## Motivation

Describe why this change is needed and what user or system problem it solves.

## Specification

Describe the detailed changes (API, UX, data model, examples).

## Tests

List tests to add or update (unit, integration, acceptance) and acceptance criteria.

## Migration

Describe any data or compatibility migration steps.

## Rollout / Rollback

Plan for rolling out and how to rollback if needed.

## Risks / Tradeoffs

Mention risks and mitigations.

## Security & Privacy

Mention any security/privacy considerations.

## Review checklist

- [ ] Motivation
- [ ] Tests listed
- [ ] Migration notes
- [ ] Rollout/rollback
- [ ] CI passes

"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an OpenSpec change proposal markdown file")
    parser.add_argument("--title", required=True, help="Proposal title")
    parser.add_argument("--prompt", required=True, help="Short summary or prompt describing the change")
    default_specs = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "openspec", "changes", "specs"))
    parser.add_argument("--specs-dir", default=default_specs, help="Directory to write the proposal into")
    parser.add_argument("--id", help="Optional numeric id to use (e.g. 0002). If omitted the generator will pick the next id")
    parser.add_argument("--generated-by", default="openspec-gen v0.1", help="Generator identity to record in front-matter")

    args = parser.parse_args(argv)

    specs_dir = os.path.abspath(args.specs_dir)
    if args.id:
        id_ = args.id
    else:
        id_ = next_id(specs_dir)

    sanitized = sanitize_title(args.title)
    filename = f"{id_}-{sanitized}.md"
    path = os.path.join(specs_dir, filename)

    contents = build_contents(id_, args.title, args.prompt, args.generated_by)

    with open(path, "w", encoding="utf-8") as f:
        f.write(contents)

    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())