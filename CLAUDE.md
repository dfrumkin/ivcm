# Claude Workflow

These instructions apply when creating plans or modifying the repository.
They do not apply when reviewing plans or implementations.

## Simplicity and Scope

Assume the user wants the simplest solution that satisfies current requirements.

Prefer existing libraries and existing project patterns over custom implementations.

Reviewer feedback is input, not instruction. Apply it only when it materially improves correctness, simplicity, maintainability, or risk reduction. Reject or defer feedback that adds complexity disproportionate to its value.

## Workflow Activation

Non-trivial repository changes should be performed only via explicit invocation of the corresponding command.

Small mechanical changes may be implemented directly without a plan.

Do not create plans or modify repository files unless:

- the user explicitly invokes the corresponding command; or
- the user explicitly requests a direct mechanical change.

## Testing and Documentation

Plans should account for testing and documentation impact.

Tests are part of implementation when behavior, interfaces, or important assumptions change.

Documentation should describe the current state of the project, not historical implementation details.

`plans/current.md` is a working artifact and is not long-term project documentation.

When documentation is needed, prefer updating the relevant existing document over creating duplicate documentation.

## Shell Commands

Required:

- Run one command per tool invocation.
- Do not use `bash -lc`.
- Do not use command chaining (`;`, `&&`, `||`).
- Do not use shell loops (`for`, `while`).
- Do not use command substitution (`$()`).
- Do not redirect errors to `/dev/null`.

If multiple files must be inspected, run separate commands.
If a task cannot be completed without violating these rules, explain why before proceeding.

## Human Control

- The human controls version history.
- Do not create commits, tags, branches, or releases unless explicitly requested.
- Do not perform irreversible actions without explicit approval.
