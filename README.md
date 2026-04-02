# scKaroSpace

`scKaroSpace` is a standalone HTML viewer generator for single-cell `h5ad` datasets.
It is embedding-centric: UMAP, t-SNE, PCA, and related 2D projections are the primary view,
with all data embedded directly into one output HTML file.

The viewer is pure client-side HTML, CSS, and vanilla JavaScript. No server is required after export.

## Features

- Multiple embedding panels from `adata.obsm` (UMAP, t-SNE, PCA, PHATE, scVI, …)
- Optional `split_by` panel generation — one panel per group value
- Cross-panel selection sync, zoom, pan, lasso, and hover sync
- Gene expression coloring with auto-selected highly variable genes
- Categorical and continuous metadata coloring from `adata.obs`
- Marker genes and pairwise differential expression in an Insights tab
- Dotplot, convex hulls, density contours, PAGA overlays, and velocity arrows when data is available
- Two export modes: single self-contained HTML, or HTML + gene shards (for large datasets)

## Install

```bash
pip install -e .
```

If your environment has issues with `scanpy` or `matplotlib` caches:

```bash
export NUMBA_CACHE_DIR=/tmp/numba-cache
export MPLCONFIGDIR=/tmp/mpl
```

## CLI usage

Basic export:

```bash
sckaro input.h5ad -o viewer.html --embeddings X_umap X_pca --color leiden
```

With split panels:

```bash
sckaro input.h5ad -o viewer.html --embeddings X_umap --split-by sample --color predicted.id
```

With analytics (marker genes + pairwise DE):

```bash
sckaro input.h5ad \
  -o viewer.html \
  --embeddings X_umap \
  --color seurat_clusters \
  --marker-genes-groupby seurat_clusters broadcelltypes predicted.id \
  --cluster-de-groupby seurat_clusters broadcelltypes
```

### CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `input` | — | Path to `.h5ad` file |
| `-o / --output` | `sckaro.html` | Output HTML path |
| `-c / --color` | `leiden` | Initial coloring: obs column or gene name |
| `--title` | `scKaroSpace` | Page title |
| `--embeddings KEY …` | all found | obsm keys to include as panels |
| `--split-by COLUMN` | — | obs column to split into per-group panels |
| `--theme` | `light` | `light` or `dark` |
| `--spot-size` | `3.0` | Default dot radius in screen pixels |
| `--downsample N` | — | Randomly downsample to N cells before export |
| `--genes GENE …` | — | Additional genes to embed for expression coloring |
| `--hvg-limit N` | `20` | Number of highly variable genes to auto-include |
| `--gene-sparse-threshold` | `0.8` | Zero-fraction above which sparse encoding is used |
| `--gene-storage` | `embedded` | `embedded` (single HTML) or `sidecar` (HTML + shards) |
| `--gene-aux-path PATH` | — | Output path for the sidecar manifest (when `--gene-storage sidecar`) |
| `--gene-sidecar-shard-size N` | `256` | Genes per shard in sidecar mode |
| `--gene-sidecar-format` | `json-v2` | `json-v2` or `binary-v1` (KSB1, smaller files) |
| `--gene-value-encoding` | `uint8` | Quantization for binary shards: `uint8` or `uint16` |
| `--marker-genes-groupby COL …` | — | obs columns for precomputed marker-gene analysis |
| `--cluster-de-groupby COL …` | — | obs columns for precomputed pairwise DE |

## Python API

```python
from sckaro import load_sc_data, export_to_html

dataset = load_sc_data(
    "input.h5ad",
    embedding_keys=["X_umap", "X_pca"],  # None = auto-detect all
    split_by=None,                        # obs column for split panels
    downsample=50_000,                    # optional cell cap
)

export_to_html(
    dataset,
    output_path="viewer.html",
    color="leiden",
    genes=["CD4", "MS4A1", "NKG7"],
    hvg_limit=20,
    marker_genes_groupby=["leiden"],
    cluster_de_groupby=["leiden"],
)
```

For large datasets, use sidecar storage so gene data is loaded on demand:

```python
# JSON shards (default)
export_to_html(dataset, output_path="viewer.html", gene_storage="sidecar")

# Binary shards (KSB1 — smaller, faster to load)
export_to_html(
    dataset,
    output_path="viewer.html",
    gene_storage="sidecar",
    gene_sidecar_format="binary-v1",
    gene_value_encoding="uint8",   # or "uint16" for higher precision
)

# Bundle everything into a single .karospace archive
export_to_html(dataset, output_path="viewer.karospace", gene_storage="sidecar",
               gene_sidecar_format="binary-v1")
```

## Examples

| Script | Dataset | Notes |
|--------|---------|-------|
| [`examples/pbmc3k.py`](./examples/pbmc3k.py) | PBMC 3k | Minimal smoke test |
| [`scripts/run_hs_gm_ctrl_multiomics_basic.py`](./scripts/run_hs_gm_ctrl_multiomics_basic.py) | HS_GM_Ctrl multiomics | Basic viewer |
| [`scripts/run_hs_gm_ctrl_multiomics_split.py`](./scripts/run_hs_gm_ctrl_multiomics_split.py) | HS_GM_Ctrl multiomics | Split-by-sample panels |
| [`scripts/run_hs_gm_ctrl_multiomics_analytics.py`](./scripts/run_hs_gm_ctrl_multiomics_analytics.py) | HS_GM_Ctrl multiomics | Marker genes + DE |

Run any script:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/mpl \
  python scripts/run_hs_gm_ctrl_multiomics_analytics.py
```

## Dataset expectations

The exporter expects:

- an `.h5ad` file readable by `scanpy`
- at least one 2D embedding in `adata.obsm`
- useful `obs` columns for categorical or continuous coloring

Obs columns are filtered automatically: noise columns (DoubletFinder artefacts),
columns with >90% missing values, near-constant numerics, and high-cardinality
categoricals (>200 unique values) are excluded from the coloring menu.

Optional data:

- `adata.uns["paga"]` — PAGA graph overlays
- `adata.obsm["velocity_umap"]` or any `velocity_*` key — RNA velocity arrows
- `adata.layers["normalized"]` / `["log1p"]` / `["lognorm"]` — preferred over `X` for gene coloring

## Development notes

- Most viewer logic lives in [`sckaro/exporter.py`](./sckaro/exporter.py)
- The HTML template is a Python format string; literal `{` and `}` inside JS/CSS must be doubled (`{{`, `}}`)
- Do not add external JS/CSS dependencies to the template — the output must be a single self-contained file
- Generated `.html`, `.karospace`, and local `.h5ad` files are gitignored
