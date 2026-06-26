# Coding standards

This project's source and tests are kept clean under **strict** static type checking
(Pyright, the engine behind the VS Code Pylance extension). This document records the typing
conventions and how to deal with type-checker diagnostics so the codebase stays at zero
warnings without scattering blanket suppressions.

## Type checking

- **Mode:** strict. Configured once in [`pyproject.toml`](../pyproject.toml) under
  `[tool.pyright]` (`typeCheckingMode = "strict"`, `venvPath`/`venv` pointing at `.venv`).
  Pylance reads this automatically, so the editor and the CLI agree.
- **Run it locally:** `uvx pyright` (from the repo root). It should report **0 errors**.
- **Interpreter:** select the project's `.venv` interpreter in your editor. A wrong
  interpreter makes Pyright unable to resolve installed packages and manufactures phantom
  "unknown type" / "missing import" errors that have nothing to do with the code.

Strict mode reports, among other things: missing parameter/return annotations, un-parameterised
generics, values whose type is `Unknown` (Pyright's name for an implicit `Any` coming from an
untyped source), unreachable code, and genuine type mismatches. The goal is to keep that signal
useful by not letting it accumulate.

## Annotation policy

Annotate **boundaries**, and let inference handle the rest.

- **Do** annotate every function/method's parameters and return type (module-level and class
  API). This is the single most effective habit: typing one function's return re-establishes
  types for everything downstream, which collapses most `Unknown`-type cascades at the source
  instead of one symptom at a time.
- **Don't** annotate obvious local variables — let Pyright infer them. Add a local annotation
  only when it is needed to break an `Unknown` cascade coming from an untyped library call.
- **Parameterise generics:** `Dataset[tuple[Any, int]]`, `DataLoader[Sample]`,
  `Mapping[str, torch.Tensor]` — never a bare `Dataset` / `dict` / `list` in an annotation.
- Keep `from __future__ import annotations` at the top of modules and use PEP 604 unions
  (`X | None`), matching the existing house style.
- Define a shared type alias when the same boundary type recurs (e.g. `Transform` in
  [`data/transforms.py`](../src/ivcm/data/transforms.py), `Batch` in
  [`models/base.py`](../src/ivcm/models/base.py), `Sample` in
  [`data/datamodules.py`](../src/ivcm/data/datamodules.py)) rather than repeating it.

### Containing untyped third-party boundaries

Several dependencies (torch, lightning, timm, **albumentations**, nibabel in places) return
loosely-typed or `Unknown` values. Re-establish types at *our* boundary instead of letting the
`Unknown` spread:

- Annotate the function that consumes the untyped value (e.g. `forward(x: Tensor) -> Tensor`),
  so callers see a concrete type.
- Model an untyped callable explicitly. Albumentations transforms have no stubs, so they are
  typed as `Transform = Callable[..., dict[str, Any]]`; the explicit `Any` is a *known* type,
  which stops Pyright's `Unknown` cascade while still being honest about the boundary.
- When a library function returns a base type that lacks the attribute you need, `cast` to the
  concrete type you know it is — see `_load_volume` casting `nib.load(...)` to `DataobjImage`.

## Suppressing diagnostics

Sometimes a diagnostic is caused by a third-party stub gap or a framework convention Pyright
can't model, not by our code. In those cases:

- Use a **targeted, rule-specific** ignore on the offending line:
  `# pyright: ignore[reportSomeRule]  # short reason`.
- **Never** use a bare `# pyright: ignore` (no rule) or a bare `# type: ignore`.
- **Never** disable a rule project-wide (or file-wide) to hide a single occurrence.
- Prefer fixing the code or annotating a boundary over suppressing. Reach for a suppression
  only when the cause is genuinely outside our control.
- `# noqa` (ruff) and `# pyright: ignore` (Pyright) are independent — a line may need both, and
  Pyright does **not** honour `# noqa`.

A suppression must sit on the exact line Pyright reports the diagnostic. Watch for the
formatter: if a long import/expression with a trailing ignore gets wrapped onto multiple lines,
the ignore can end up on the wrong line — keep the line short enough to stay intact (shorten
the comment) rather than splitting it.

The only suppressions currently in the tree, and why each is justified:

| Where | Rule(s) | Reason |
|-------|---------|--------|
| `data/transforms.py` imports | `reportMissingTypeStubs` | albumentations ships no type stubs |
| `data/datamodules.py` `_load_volume` | `reportUnknownMemberType` | `nib.load` has untyped `**kwargs` |
| `train.py` / `test_callbacks.py` `best_model_path` | `reportUnknownMemberType`, `reportUnknownArgumentType` | Lightning sets `ModelCheckpoint.best_model_path` internally without an annotation |
| `tests/conftest.py` autouse fixture | `reportUnusedFunction` | pytest invokes the fixture by collection, not by name |
| `tests/test_cv_config.py` imports | `reportPrivateUsage` | deliberate white-box unit tests of `cv.py`'s private helpers |
| `morphometry.py` `scikit-image`/`scipy.ndimage` imports | `reportUnknownVariableType`, `reportMissingTypeStubs` | both ship only partial type info; the CLI flags the callables as `reportUnknownVariableType`, Pylance also flags `scipy.ndimage` as `reportMissingTypeStubs` |

## Side-effect imports

An `import X` whose only purpose is to run module-level code (a side effect) reads as "unused"
to both ruff (`F401`) and Pyright (`reportUnusedImport`), but must not be deleted.

Prefer making the side effect **explicit** over suppressing it: expose a function and call it.
The custom `${len:...}` OmegaConf resolver is registered by `ivcm.register_resolvers()`, which
`ivcm/__init__.py` calls at import time (covering the `train`/`cross-validate` entry points) and
[`tests/conftest.py`](../tests/conftest.py) calls explicitly once for the whole test suite — so
individual test modules need no side-effect `import ivcm`. If a side-effect import is truly
unavoidable, keep it and stack a targeted ignore with a reason, e.g.
`import x  # noqa: F401  # pyright: ignore[reportUnusedImport]  # registers ...`.

## Tests

Tests are held to the same standard as `src/`. Annotate test-function parameters — fixtures
(`tmp_path: Path`, `monkeypatch: pytest.MonkeyPatch`) and `@pytest.mark.parametrize` arguments —
and helper/factory functions. Monkeypatch stubs replacing real functions should be small typed
`def`s (or typed callables) rather than untyped `lambda`s, which Pyright cannot infer.
