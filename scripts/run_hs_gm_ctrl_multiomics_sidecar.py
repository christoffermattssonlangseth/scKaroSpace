"""
Sidecar export for HS_GM_Ctrl_multiomics_UCSC_final.h5ad.

Writes:
- hs_gm_ctrl_multiomics_sidecar.html
- hs_gm_ctrl_multiomics_sidecar.genes.json
- hs_gm_ctrl_multiomics_sidecar.genes/

Run:
    NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/mpl \
    python scripts/run_hs_gm_ctrl_multiomics_sidecar.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sckaro import export_to_html, load_sc_data


DATASET_PATH = Path("/Users/chrislangseth/Downloads/HS_GM_Ctrl_multiomics_UCSC_final.h5ad")
OUTPUT_PATH = ROOT / "hs_gm_ctrl_multiomics_sidecar.html"


dataset = load_sc_data(
    DATASET_PATH,
    embedding_keys=["wnn.umap", "umap.atac"],
)

export_to_html(
    dataset,
    output_path=OUTPUT_PATH,
    color="seurat_clusters",
    title="HS GM Ctrl Multiomics Sidecar",
    theme="light",
    spot_size=3.5,
    genes=["MBP", "PLP1", "GFAP", "AQP4", "PDGFRA", "VCAN", "MOG", "MAG"],
    hvg_limit=20,
    gene_storage="sidecar",
    gene_sidecar_shard_size=256,
    marker_genes_groupby=["seurat_clusters", "broadcelltypes", "predicted.id"],
    cluster_de_groupby=["seurat_clusters", "broadcelltypes"],
)

print(f"Wrote {OUTPUT_PATH}")
print(f"Wrote {OUTPUT_PATH.with_suffix('.genes.json')}")
print(f"Wrote {OUTPUT_PATH.with_suffix('.genes')}")
