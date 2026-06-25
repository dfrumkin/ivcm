# Implement

This command may be invoked as:

```text
/implement
/implement N
```

where `N` is a positive integer specifying the maximum number of Codex review iterations. If omitted, `N = 10`.

Implement the approved plan in `plans/current.md`.

Use the current conversation and `plans/current.md` as the authoritative sources of requirements.

If no approved plan exists, stop and request human approval of a plan before proceeding.

Do not modify `plans/current.md`.

## Scope Control

Treat `plans/current.md` as the authoritative description of the intended change and scope.

Minor implementation details may be decided during implementation.

If implementation reveals a significant change in scope, architecture, data flow, public interfaces, or workflows:

- Stop implementation.
- Do not modify `plans/current.md`.
- Report the issue clearly, including:
  - what part of the plan is no longer valid;
  - what change seems necessary;
  - which files have already been modified;
  - whether the current worktree should be kept, reverted, or used as context for a revised plan.
- Wait for human instruction.

## Implementation

Implement the approved plan.

Add or update tests required by the plan or by changed behavior.

Update relevant documentation when behavior, architecture, workflows, assumptions, or interfaces change.

Run relevant tests.

Fix test failures that are caused by the implementation.

## Codex Review Loop

After implementation and testing are complete, run a Codex review loop.

Repeat at most N times:

1. Run:

   ```bash
   codex exec "Review the implementation"
   ```

2. If the Codex command fails, report the failure and stop the review loop.
3. Read and analyze the command output.
4. If you are not going to modify the implementation as a result of the review, stop the review loop.
5. Incorporate valid feedback.
6. Run any tests affected by the changes.

## Completion Report

After the implementation process completes:

- If the review loop terminated because no review feedback resulted in changes, report that the implementation is ready for approval and provide a brief summary of the changes, tests, and documentation updates.
- If the review loop terminated because the maximum number of review iterations was reached, run one final Codex review without making further changes. Report that human review is required and summarize the latest unresolved review feedback.
- If a Codex review failed, report the failure and stop implementation. Do not continue with additional review iterations.
- Wait for human approval before any commit, release, or further work.
