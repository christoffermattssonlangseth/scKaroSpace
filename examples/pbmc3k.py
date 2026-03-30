"""
Example: PBMC 3k dataset.

Expects a preprocessed pbmc3k.h5ad with X_umap, leiden, etc.
You can generate it with:

    import scanpy as sc
    adata = sc.datasets.pbmc3k_processed()
    adata.write("pbmc3k.h5ad")
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from sckaro import load_sc_data, export_to_html

dataset = load_sc_data(
    "pbmc3k.h5ad",
    embedding_keys=["X_umap", "X_pca"],
)

export_to_html(
    dataset,
    output_path="pbmc3k_viewer.html",
    color="leiden",
    title="PBMC 3k",
    theme="light",
    spot_size=3.5,
    genes=["CD4", "CD8A", "CD19", "GNLY", "MS4A1", "NKG7", "PPBP", "LYZ"],
    hvg_limit=30,
)
