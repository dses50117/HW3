# Project Context

## Purpose
This repository contains OpenSpec artifacts and team-facing guidance for how to propose, review, and apply specification-driven changes to the project. The primary goals are:
- Capture change proposals as first-class, reviewable artifacts (OpenSpec change proposals).
- Let contributors and automated agents (AI assistants) create, refine, and implement changes in a consistent, auditable workflow.
- Provide clear project context, conventions, and minimal automation so proposals can be tested, validated, and merged safely.

## Tech Stack (assumptions & recommendations)
- Primary artifact format: Markdown for specs and proposals (`.md`).
- VCS: Git (hosted on GitHub/GitLab/Bitbucket depending on team).
- Editor/IDE: VS Code recommended (repo includes guidance for contributors).
- Optional toolchain (for local automation / linters / test runs): Node.js (16+), npm/yarn, and/or Python 3.10+ for scripting. These are optional; the OpenSpec files are editor-readable Markdown.
- CI: GitHub Actions (recommended) to run linting/tests and to validate proposals.

Note (assumptions): Where the repository doesn't include an explicit runtime, this project focuses on process and spec files. If you have an existing app (TS/Go/Python/etc.), list languages and frameworks here.

## Project Conventions

### Contract (what helpers / agents should assume)
- Inputs: Markdown OpenSpec proposals, small code patches, tests (if included).
- Outputs: Updated proposal files, suggested code diffs, test updates, and migration notes.
- Error modes: proposals missing motivation or tests should be marked WIP; code changes without CI validation are not safe to merge.

### Code Style
- Specs and proposals use plain Markdown with front-matter-like headings (title, summary, motivation, spec, tests, rollout).
- For code in the repo: prefer established linters/formatters (Prettier for JS/TS, Black for Python). Adopt `conventional commits` (type: scope: subject) for commit messages.

### Naming and Branching
- Branches: `feature/<short-desc>`, `fix/<short-desc>`, `chore/<short-desc>`, `spec/<id>-<short-desc>`.
- Change proposal files should live under `openspec/changes/specs/` and be prefixed with a numeric identifier (e.g., `0001-...md`) to maintain order.

### Architecture Patterns
- Spec-driven changes: proposals describe intent, API/UX impact, migration steps, and tests.
- Keep proposals small and focused: one proposal ≈ one feature or one non-trivial change.
- Agents and automation should operate by creating or updating spec files and producing minimal, reviewable code patches.

### Testing Strategy
- Each proposal that changes behavior must include at least one of: unit tests, integration tests, or an acceptance test plan.
- CI must run the test suite and any linters on pull requests that include code or spec changes.
- For spec-only changes (documentation/process), a lighter checklist is acceptable (reviewers, checklist items completed).

### Git Workflow and PRs
- Use feature branches with pull requests for review.
- PRs must reference the change proposal file (if one exists) and include a short testing summary.
- Use `Draft PR` for work-in-progress proposals.

## Domain Context
- This repository is focused on OpenSpec and change-proposal-driven development. Domain terms:
	- Proposal: an OpenSpec document describing a proposed change.
	- AGENT: an AI-assisted contributor following the instructions in `openspec/AGENTS.md`.
	- Reviewer: a human maintainer who verifies design, tests, and safety.

## Important Constraints
- Do not merge changes without passing CI unless explicitly approved by a maintainer.
- No secrets or credentials should be added to proposals or checked-in files.
- Prefer minimal, incremental changes to reduce review surface.

## External Dependencies
- None required to author specs. Recommended dev dependencies:
	- Node.js (for local tooling)
	- Prettier / ESLint / Black (formatters)
	- GitHub Actions (CI) or equivalent

If your project uses other external APIs or services, document them here so agents can account for network or permission constraints when proposing changes.
