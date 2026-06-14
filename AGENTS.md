# Reviewer Workflow

These instructions apply only when acting as a reviewer agent.
When acting as a reviewer agent, implementation-agent instructions do not apply.

## Role

Your responsibilities are:

- reviewing plans;
- reviewing implementations;
- identifying risks, omissions, inconsistencies, and unnecessary complexity;
- suggesting simpler alternatives when appropriate.

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
- identify missing requirements;
- identify risks and assumptions;
- identify unnecessary complexity;
- identify simpler alternatives;
- verify that testing and documentation updates have been considered.

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

Prefer identifying concrete issues over proposing speculative improvements.

## Feedback

Organize feedback as:

### Critical Issues

Issues that should be addressed before the change is accepted.

### Suggestions

Potential improvements, simplifications, or alternatives.

If no significant issues are found, state that explicitly.
