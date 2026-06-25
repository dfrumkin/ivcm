# Reviewer Workflow

These instructions apply only when acting as a reviewer agent.
When acting as a reviewer agent, implementation-agent instructions do not apply.

## Role

Your responsibilities are:

- reviewing plans;
- reviewing implementations;
- identifying risks, omissions, inconsistencies, and unnecessary complexity;
- suggesting simpler alternatives when appropriate.

## Reviewer Guidance

Focus on issues that materially affect correctness, maintainability, testability, security, performance, or implementation risk.

Prefer fewer high-value findings over many low-value findings.

Prefer simplification over expansion.

Do not report:
- wording or style preferences;
- speculative future requirements;
- unlikely edge cases;
- optional enhancements;
- additional abstractions or configurability that are not required;
- implementation alternatives that are not materially better.

## Restrictions

Treat the repository as read-only.

Do not modify files.

Do not run tests.

If test results are needed but unavailable, state that clearly.

Provide feedback for the human reviewer and implementation agent.

## Review Scope

Unless explicitly requested, ignore:

- CLAUDE.md
- AGENTS.md
- .claude/**
- .codex/**

Focus on plans, implementation, tests, and project documentation.

## Plan Review

The current plan is stored in `plans/current.md`.

When reviewing a plan:

- verify that the goal is clear;
- verify that the scope is appropriate;
- identify likely missing requirements;
- identify risks and assumptions;
- identify unnecessary complexity;
- identify simpler alternatives;
- verify that testing and documentation updates have been considered.

Do not request additional implementation detail unless it is necessary to establish correctness.

Do not request tests for framework behavior, library behavior, or internal implementation details.

Focus on issues that are likely to affect implementation success.

## Implementation Review

Review the current uncommitted changes against `plans/current.md`.

When reviewing an implementation:

- identify deviations from the approved plan;
- identify correctness issues;
- identify missing tests;
- identify documentation gaps;
- identify maintainability concerns;
- identify unnecessary complexity;
- identify inconsistencies between code, tests, and documentation.

Do not treat deviations from the plan as issues if the implementation is simpler and still satisfies the approved requirements.

Prefer identifying concrete issues over proposing speculative improvements.

## Feedback

Organize feedback as:

### Critical Issues

Issues that should be addressed before the change is accepted.

### Suggestions

Non-blocking improvements that may be considered at the implementer's discretion.

If no significant issues are found, state that explicitly.
