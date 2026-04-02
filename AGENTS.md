# AGENTS.md — scKaroSpace

## Project overview

scKaroSpace generates self-contained HTML viewers for single-cell RNA-seq datasets (`.h5ad`).
The viewer is embedding-centric (UMAP, t-SNE, PCA) and runs entirely client-side — no server after export.

## Repository layout

```
sckaro/
  __init__.py        # Lazy public API (load_sc_data, export_to_html, …)
  cli.py             # argparse CLI entry point
  data_loader.py     # h5ad loading, obs filtering, ScDataset / EmbeddingView dataclasses
  exporter.py        # Core HTML generation (~3600 lines) — gene encoding, palettes, template
scripts/             # Example export scripts for real datasets
examples/            # Minimal smoke-test (pbmc3k.py)
pyproject.toml       # Package metadata and dependencies
```

## Core abstractions

- **`ScDataset`** — container: AnnData object, list of `EmbeddingView`s, selectable obs columns, gene names
- **`EmbeddingView`** — one panel: id, name, embedding key, x/y coordinates, optional cell indices
- Gene storage modes: `embedded` (all in HTML) or `sidecar` (HTML + JSON manifest + shards)

## Key entry points

```python
from sckaro import load_sc_data, export_to_html

dataset = load_sc_data("input.h5ad", embedding_keys=["X_umap"], split_by=None)
export_to_html(dataset, output_path="viewer.html", color="leiden")
```

CLI:

```bash
sckaro input.h5ad -o viewer.html --embeddings X_umap --color leiden
```

## Development guidelines

- **`exporter.py` is the heart of the project.** Almost all viewer logic lives there.
- The HTML template is a Python format string. Literal `{` and `}` inside JS/CSS must be doubled (`{{`, `}}`).
- Generated `.html`, `.karospace`, and local `.h5ad` files are gitignored — do not commit them.
- The viewer is vanilla HTML/CSS/JS (Canvas-based). Do not introduce external JS/CSS dependencies in the template.
- Obs column filtering (noise removal) happens in `data_loader.py` before export — keep loading and export concerns separate.
- Sparse gene encoding is chosen automatically when the zero-fraction exceeds `gene_sparse_threshold` (default 0.8).

## Running examples

```bash
pip install -e .
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/mpl \
  python examples/pbmc3k.py
```

## Dependencies

| Package | Purpose |
|---------|---------|
| scanpy / anndata | Read h5ad, compute HVGs, marker genes, DE |
| numpy / pandas / scipy | Numeric processing |
| tqdm | Progress bars |

No runtime server dependencies. All viewer assets are inlined into the output HTML.
