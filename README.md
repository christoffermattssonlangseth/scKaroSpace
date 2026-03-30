# scKaroSpace

`scKaroSpace` is a standalone HTML viewer generator for single-cell `h5ad` datasets.
It is embedding-centric: UMAP, t-SNE, PCA, and related 2D projections are the primary view,
with all data embedded directly into one output HTML file.

The viewer is pure client-side HTML, CSS, and vanilla JavaScript. No server is required after export.

## Current features

- Multiple embedding panels from `adata.obsm`
- Optional `split_by` panel generation
- Cross-panel selection sync
- Modal zoom, pan, lasso, and hover sync
- Gene coloring from embedded expression values
- Categorical and continuous metadata coloring from `adata.obs`
- Marker genes and pairwise DE Insights tabs
- Dotplot, convex hulls, density contours, PAGA overlays, and velocity arrows when data is available
- Single-file HTML export with no sidecar assets

## Install

From the repo root:

```bash
pip install -e .
```

If your environment has issues with `scanpy` or `matplotlib` caches, these runtime env vars help:

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

With analytics:

```bash
sckaro input.h5ad \
  -o viewer.html \
  --embeddings X_umap \
  --color seurat_clusters \
  --marker-genes-groupby seurat_clusters broadcelltypes predicted.id \
  --cluster-de-groupby seurat_clusters broadcelltypes
```

## Python usage

```python
from sckaro import load_sc_data, export_to_html

dataset = load_sc_data(
    "input.h5ad",
    embedding_keys=["X_umap", "X_pca"],
    split_by=None,
)

export_to_html(
    dataset,
    output_path="viewer.html",
    color="leiden",
    genes=["CD4", "MS4A1", "NKG7"],
    marker_genes_groupby=["leiden"],
    cluster_de_groupby=["leiden"],
)
```

## Included examples

- [examples/pbmc3k.py](./examples/pbmc3k.py)
  - minimal smoke-test style example
- [scripts/run_hs_gm_ctrl_multiomics_basic.py](./scripts/run_hs_gm_ctrl_multiomics_basic.py)
  - basic viewer for `HS_GM_Ctrl_multiomics_UCSC_final.h5ad`
- [scripts/run_hs_gm_ctrl_multiomics_split.py](./scripts/run_hs_gm_ctrl_multiomics_split.py)
  - split-by-sample viewer for the same dataset
- [scripts/run_hs_gm_ctrl_multiomics_analytics.py](./scripts/run_hs_gm_ctrl_multiomics_analytics.py)
  - analytics-enabled export for the same dataset

Run a dataset script like this:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/mpl \
python scripts/run_hs_gm_ctrl_multiomics_analytics.py
```

## Dataset expectations

The exporter expects:

- an `.h5ad` file readable by `scanpy`
- at least one 2D embedding in `adata.obsm`
- useful `obs` columns for categorical or continuous coloring

Optional data:

- `adata.uns["paga"]` for PAGA overlays
- `adata.obsm["velocity_umap"]` or similar `velocity_*` embeddings for velocity arrows

## Development notes

- The viewer lives almost entirely in [sckaro/exporter.py](./sckaro/exporter.py)
- The HTML template is a Python format string, so literal `{` and `}` inside JS/CSS must be doubled
- Output `.html` and local test `.h5ad` files are gitignored
