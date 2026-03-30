"""
Split-panel export for HS_GM_Ctrl_multiomics_UCSC_final.h5ad.

Run:
    NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/mpl \
    python scripts/run_hs_gm_ctrl_multiomics_split.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sckaro import export_to_html, load_sc_data


DATASET_PATH = Path("/Users/chrislangseth/Downloads/HS_GM_Ctrl_multiomics_UCSC_final.h5ad")
OUTPUT_PATH = ROOT / "hs_gm_ctrl_multiomics_split_by_sample.html"


dataset = load_sc_data(
    DATASET_PATH,
    embedding_keys=["wnn.umap"],
    split_by="sample",
)

export_to_html(
    dataset,
    output_path=OUTPUT_PATH,
    color="predicted.id",
    title="HS GM Ctrl Multiomics Split by Sample",
    theme="light",
    spot_size=3.5,
    genes=["MBP", "PLP1", "MOG", "CUX2", "SATB2", "AQP4", "PDGFRA", "VCAN"],
    hvg_limit=20,
)

print(f"Wrote {OUTPUT_PATH}")
