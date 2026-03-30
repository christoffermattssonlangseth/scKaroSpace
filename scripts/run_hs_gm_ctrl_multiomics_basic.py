"""
Basic scKaroSpace export for HS_GM_Ctrl_multiomics_UCSC_final.h5ad.

Run:
    NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/mpl \
    python scripts/run_hs_gm_ctrl_multiomics_basic.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sckaro import export_to_html, load_sc_data


DATASET_PATH = Path("/Users/chrislangseth/Downloads/HS_GM_Ctrl_multiomics_UCSC_final.h5ad")
OUTPUT_PATH = ROOT / "hs_gm_ctrl_multiomics_basic.html"


dataset = load_sc_data(
    DATASET_PATH,
    embedding_keys=["wnn.umap", "umap.harmony", "umap.atac"],
)

export_to_html(
    dataset,
    output_path=OUTPUT_PATH,
    color="broadcelltypes",
    title="HS GM Ctrl Multiomics",
    theme="light",
    spot_size=3.5,
    genes=["MBP", "PLP1", "GFAP", "AQP4", "PDGFRA", "VCAN", "MOBP", "MAG"],
    hvg_limit=24,
)

print(f"Wrote {OUTPUT_PATH}")
