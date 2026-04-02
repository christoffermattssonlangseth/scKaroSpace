"""
Package export for Microglia_SCT.h5ad.

Run:
    python scripts/run_microglia_sct_package.py
"""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from sckaro import export_to_html, load_sc_data


DATASET_PATH = Path("/Users/chrislangseth/Microglia_SCT.h5ad")
OUTPUT_PATH = ROOT / "microglia_sct.karospace"
EMBEDDINGS = ["umap_Microglia", "umap.unintegrated"]


def main() -> None:
    dataset = load_sc_data(
        DATASET_PATH,
        embedding_keys=EMBEDDINGS,
    )

    export_to_html(
        dataset,
        output_path=OUTPUT_PATH,
        color="Microglia_clusters",
        title="Microglia SCT",
        theme="light",
        spot_size=3.5,
        hvg_limit=20,
        gene_storage="sidecar",
        gene_sidecar_shard_size=256,
        marker_genes_groupby=["Microglia_clusters"],
        cluster_de_groupby=["Microglia_clusters"],
    )

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {OUTPUT_PATH.with_suffix('.loader.html')}")


if __name__ == "__main__":
    main()
