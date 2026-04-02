"""
Basic HTML export for Astrocytes_SCT.h5ad.

Run:
    python scripts/run_astrocytes_sct_basic.py
"""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from sckaro import export_to_html, load_sc_data


DATASET_PATH = Path("/Users/chrislangseth/Astrocytes_SCT.h5ad")
OUTPUT_PATH = ROOT / "astrocytes_sct.html"
EMBEDDINGS = ["umap_Astrocytes"]


def main() -> None:
    dataset = load_sc_data(
        DATASET_PATH,
        embedding_keys=EMBEDDINGS,
    )

    export_to_html(
        dataset,
        output_path=OUTPUT_PATH,
        color="Astrocytes_clusters",
        title="Astrocytes SCT",
        theme="light",
        spot_size=3.5,
        hvg_limit=20,
        marker_genes_groupby=["Astrocytes_clusters"],
        cluster_de_groupby=["Astrocytes_clusters"],
    )

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
