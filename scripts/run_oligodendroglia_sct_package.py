"""
Package export for Oligodendroglia_SCT.h5ad.

Writes:
- oligodendroglia_sct.karospace  (binary KSB1 shards, openable with karospace-package-loader.html)
- oligodendroglia_sct.loader.html

Run:
    python scripts/run_oligodendroglia_sct_package.py
"""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from sckaro import export_to_html, load_sc_data


DATASET_PATH = Path("/Users/chrislangseth/Oligodendroglia_SCT.h5ad")
OUTPUT_PATH = ROOT / "oligodendroglia_sct.karospace"
EMBEDDINGS = ["umap.unintegrated"]


def main() -> None:
    dataset = load_sc_data(
        DATASET_PATH,
        embedding_keys=EMBEDDINGS,
    )

    export_to_html(
        dataset,
        output_path=OUTPUT_PATH,
        color="Oligodendroglia_clusters",
        title="Oligodendroglia SCT",
        theme="light",
        spot_size=3.5,
        hvg_limit=20,
        gene_storage="sidecar",
        gene_sidecar_format="binary-v1",
        gene_value_encoding="uint8",
        gene_sidecar_shard_size=256,
        marker_genes_groupby=["Oligodendroglia_clusters"],
        cluster_de_groupby=["Oligodendroglia_clusters"],
    )

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {OUTPUT_PATH.with_suffix('.loader.html')}")


if __name__ == "__main__":
    main()
