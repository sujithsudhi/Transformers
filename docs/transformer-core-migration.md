# Transformer Core Extraction Guide

This repository now consumes reusable transformer primitives from the separate
`transformer-core` repository:

- GitHub: `https://github.com/sujithsudhi/transformer-core.git`
- Local sibling path: `C:\Users\Sujith\Dev\Repos\transformer-core`

The original source for these blocks used to live under `models/core/` in this
repository. That code has now been extracted and should be maintained in the
shared repository instead.

## What Belongs In `transformer-core`

Move only generic, reusable building blocks:

- `PositionalEncoding`
- `TokenEmbedding`
- `MultiHeadSelfAttention`
- `ResidualBlock`
- `FeedForward`
- `TransformerEncoderLayer`
- `TransformerDecoderLayer`

Keep app-specific code in this repository:

- `ClassifierModel`
- training loops and configs
- data loaders
- evaluation and visualization utilities
- deployment helpers

## Recommended Repository Layout

Keep the shared repository beside this one:

```text
Repos/
  Transformers/
  transformer-core/
```

Inside `transformer-core/` use this structure:

```text
transformer-core/
  pyproject.toml
  README.md
  src/
  tests/
```

## One-Time Setup

1. Clone the shared repository beside this one.
2. Install it into your active environment:

```powershell
C:\DeepLearning\dl\Scripts\python.exe -m pip install -e C:\Users\Sujith\Dev\Repos\transformer-core
```

3. Verify the package import if needed:

```powershell
C:\DeepLearning\dl\Scripts\python.exe -c "from transformer_core import PositionalEncoding; print(PositionalEncoding)"
```

## Local Development Workflow

During active development, keep both repositories side by side and use the
editable install:

```powershell
cd C:\Users\Sujith\Dev\Repos\Transformers
python -m pip install -e ..\transformer-core
```

Editable installs mean:

- changes made in `transformer-core` are immediately visible here
- you can prototype a new reusable layer in the app repo, then move it into the
  shared repo once the design stabilizes
- this repository does not need a local copy of the shared source

## Release Workflow

When a new architecture introduces a reusable block:

1. Implement and test it in `transformer-core`.
2. Commit the change in `transformer-core`.
3. Tag a release:

```powershell
git tag v0.2.0
git push origin main --tags
```

4. Update this repository to that version if you move away from editable local installs:

```text
transformer-core @ git+https://github.com/<your-user>/transformer-core.git@v0.2.0
```

Use semantic versioning:

- `0.2.0` for new reusable features
- `0.2.1` for fixes
- `1.0.0` once the public API is stable

## Dependency Options

Use one of these patterns:

### Option 1: Editable local install

Best during active development across multiple local repositories.

```powershell
python -m pip install -e ..\transformer-core
```

### Option 2: Git-pinned dependency

Best for stable sharing without publishing to PyPI.

```text
transformer-core @ git+https://github.com/<your-user>/transformer-core.git@v0.2.0
```

### Option 3: Private package index

Best later if you want a more formal team workflow.

## Import Migration In This Repository

This repository already imports the shared package directly:

```python
from transformer_core import PositionalEncoding, TransformerDecoderLayer
from transformer_core import PositionalEncoding, TransformerEncoderLayer
```

## Practical Rule For New Architectures

Use this rule to decide where new code should go:

- if it is generic and likely reusable, add it to `transformer-core`
- if it is experimental or app-specific, keep it in the application repo first

Promote code into the shared package only after the abstraction is clear.
