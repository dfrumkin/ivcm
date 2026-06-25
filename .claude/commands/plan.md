# Plan

This command may be invoked as:

```text
/plan
/plan N
```

where `N` is a positive integer specifying the maximum number of Codex review iterations. If omitted, `N = 10`.

Create or update `plans/current.md` based on the current conversation and repository state.

Use the current conversation as the primary source of requirements.

Do not create, modify, rename, or delete any repository file except `plans/current.md`.

## Preparation

Inspect relevant code and documentation.

Throughout the planning process, ask questions only if a material ambiguity prevents planning.

## Plan Content

The plan should:
- Describe the goal.
- Describe the scope.
- Describe the proposed implementation.
- Identify affected files and components.
- Describe testing and documentation changes, if any.

Prefer the simplest design that satisfies the requirements.

Do not add unrelated improvements, refactors, optimizations, or future work unless required by the request.

Avoid implementation details unless they materially affect design, scope, correctness, reproducibility, or user-visible behavior.

If planning follows a stopped implementation attempt, include a `Current Implementation State` section summarizing:
- what is already implemented;
- what is partially implemented;
- what is still pending;
- why the previous plan required revision.

## Testing Guidance

Test requirements, correctness properties, and user-visible behavior.

Do not add tests solely for:
- internal helpers;
- framework behavior;
- third-party library behavior;
- implementation details.

Prefer a small number of high-value tests.

Describe what must be tested, not how tests should be implemented.

## Codex Review Loop

After creating the initial version of `plans/current.md`, run a Codex review loop.

Repeat at most N times:

1. Run:

   ```bash
   codex exec "Review the plan"
   ```

2. If the Codex command fails, report the failure and stop the review loop.
3. Read and analyze the command output.
4. If you are not going to modify `plans/current.md` as a result of the review, stop the review loop.
5. Incorporate valid feedback into `plans/current.md`.

## Completion Report

After the planning process completes:

- If the review loop terminated because no review feedback resulted in changes, report that the plan is ready for approval and provide a brief summary of the plan.
- If the review loop terminated because the maximum number of review iterations was reached, run one final Codex review without making further changes. Summarize the plan and any unresolved review feedback. Report that human review is required before approval.
- If a Codex review failed, report the failure and stop planning. Do not continue with additional review iterations.
- Wait for human approval before any implementation work.
