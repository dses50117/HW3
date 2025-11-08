---
title: "AI-assisted change proposal generator"
id: 0001
status: draft
authors:
  - OpenSpec Agent
  - You
---

# Summary

Add an AI-assisted change proposal generator that helps contributors produce complete, reviewable OpenSpec change proposals from a short prompt or partial draft. The generator should produce a proposal stub with required headings, a motivation section, a suggested spec, test requirements, migration notes, and a reviewer checklist.

## Motivation

Creating a high-quality change proposal requires remembering required sections (motivation, spec, tests, migration, rollout). An AI-assisted generator reduces friction, increases consistency, and helps junior contributors produce proposals that meet repo standards.

## Goals

- Provide a command-line or editor-assisted tool (or an AI agent flow) that takes a short prompt and produces a complete OpenSpec proposal markdown file under `openspec/changes/specs/`.
- Generated proposals must include: summary, motivation, spec, tests, migration plan, rollout, risk, and review checklist.
- Mark generated proposals clearly (front-matter `generated_by` and `generated_at`) so reviewers can quickly see the origin.

## Non-Goals

- This proposal does not require automatic code generation or automatic merges. The generator only creates proposals and suggested diffs; humans must review and approve.

## Specification

1. CLI / Editor entrypoint (optional):
   - `openspec-gen --title "Add X" --prompt "Short description"`
   - Or via editor command palette: `OpenSpec: New Proposal from Prompt`

2. Output: a Markdown file at `openspec/changes/specs/0001-<sanitized-title>.md` with the following sections:
   - Title
   - Summary
   - Motivation
   - Specification (detailed description, API surface if applicable)
   - Tests (what to add or update)
   - Migration (DB/data steps, compatibility)
   - Rollout / Rollback plan
   - Risks/Tradeoffs
   - Security & Privacy considerations
   - Review checklist (e.g., unit tests, docs updated, CI green)

3. Front-matter metadata keys: `id`, `title`, `status` (draft), `generated_by` (string), `generated_at` (ISO8601)

4. The generator should include a short example PR description and suggested commit message.

## Design & Implementation Notes

- Implementation can start as a simple script (Node.js or Python) that uses a local prompt template. Optionally, integrate with an LLM or the repo's AI agent.
- The generator must never include secrets or call external APIs without explicit configuration.
- Ensure deterministic output for the same prompt + seed to make reviews reproducible.

### Files to add/modify

- Add a CLI script or editor extension integration under `tools/openspec-gen/` (optional for MVP).
- Add unit tests that validate generated proposal contains required headings and metadata.

## Tests

- Unit test: given a short prompt, generated file includes all required headings and front-matter.
- Integration test (CI): generate a proposal in a temp dir and validate lint/format.

## Migration

- No database migration. If generator scaffolds code or tests later, migrations must be listed in proposals.

## Rollout

- Roll out the generator as an opt-in tool. Initially place under `tools/` and document usage in `README.md`.

## Security & Privacy

- If using an external LLM, document which model/service and ensure no private repository secrets are sent. Default to local templates if no LLM is configured.

## Risks

- Generated proposals might be technically incorrect; require human review.

## Review Checklist

- [ ] Proposal includes Motivation
- [ ] Proposal includes Tests section with pass/fail criteria
- [ ] Proposal includes Rollout and Rollback plan
- [ ] Proposal metadata populated (`generated_by`, `generated_at`)
- [ ] No secrets included in the proposal

## Example

Example generated front-matter:

```yaml
---
id: 0001
title: "AI-assisted change proposal generator"
status: draft
generated_by: "openspec-gen v0.1"
generated_at: "2025-11-08T00:00:00Z"
---
```

And then the required sections described above.
