"""
Package export for HS_GM_Ctrl_multiomics_UCSC_final.h5ad.

Writes:
- hs_gm_ctrl_multiomics.karospace

Run:
    python scripts/run_hs_gm_ctrl_multiomics_package.py
"""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from sckaro import export_to_html, load_sc_data


DATASET_PATH = Path("/Users/chrislangseth/Downloads/HS_GM_Ctrl_multiomics_UCSC_final.h5ad")
OUTPUT_PATH = ROOT / "hs_gm_ctrl_multiomics.karospace"
EMBEDDINGS = ["wnn.umap", "umap.atac"]
GENES = ["MBP", "PLP1", "GFAP", "AQP4", "PDGFRA", "VCAN", "MOG", "MAG"]
MARKER_GROUPBY = ["seurat_clusters", "broadcelltypes", "predicted.id"]
CLUSTER_DE_GROUPBY = ["seurat_clusters", "broadcelltypes"]


def main() -> None:
    dataset = load_sc_data(
        DATASET_PATH,
        embedding_keys=EMBEDDINGS,
    )

    export_to_html(
        dataset,
        output_path=OUTPUT_PATH,
        color="seurat_clusters",
        title="HS GM Ctrl Multiomics Package",
        theme="light",
        spot_size=3.5,
        genes=GENES,
        hvg_limit=20,
        gene_storage="sidecar",
        gene_sidecar_shard_size=256,
        marker_genes_groupby=MARKER_GROUPBY,
        cluster_de_groupby=CLUSTER_DE_GROUPBY,
    )

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {OUTPUT_PATH.with_suffix('.loader.html')}")


if __name__ == "__main__":
    main()
