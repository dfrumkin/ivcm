# Claude Workflow

These instructions apply only when acting as an implementation agent.
When acting as an implementation agent, reviewer-agent instructions do not apply.

## Planning

Before making non-trivial changes:

1. Inspect the relevant code and documentation.
2. Create or update `plans/current.md`.
3. Wait for human approval before implementation.

Small mechanical changes do not require a formal plan.

When requirements are ambiguous, ask for clarification rather than making assumptions.

## Plan Revisions

When review feedback is received:

- Update the plan yourself.
- Do not ask reviewers to edit the plan.

## Implementation

After plan approval:

1. Treat the approved plan as the authoritative description of the intended change and scope.
2. Implement the approved plan.
3. Add or update tests when appropriate.
4. Update relevant documentation.

Minor implementation details may be decided during implementation.

If implementation reveals a significant change in scope, architecture, data flow, public interfaces, or workflows:

1. Pause implementation.
2. Update the plan.
3. Request review of the revised plan before continuing.

## Testing

Tests are part of the implementation.

When appropriate:

- Add tests for new functionality.
- Update tests affected by the change.
- Run relevant tests.
- Fix failures before requesting implementation review.

## Documentation

Documentation should record the current state of the project, not historical implementation details.

Documentation should be sufficient for a future session to understand the current state of the project and continue development.

`plans/current.md` is a working artifact and should not be considered part of the project's long-term documentation.

When behavior, architecture, workflows, assumptions, or interfaces change:

- Update the relevant documentation.
- Avoid duplicating information across documents.
- Prefer modifying existing documentation over creating new documentation when the information naturally belongs in an existing document.
- Refactor documentation structure when it no longer reflects the project.

## Human Control

The human is responsible for:

- approving plans;
- providing review feedback;
- approving implementations;
- creating commits;
- deciding when a session begins and ends.

Do not commit code.
Do not create releases.
Do not perform irreversible actions without explicit approval.
