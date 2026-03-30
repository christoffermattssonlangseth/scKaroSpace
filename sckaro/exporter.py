"""
Export single-cell data to a standalone HTML viewer.

Produces a single self-contained HTML file with all data embedded as JSON
and a vanilla-JS Canvas-based viewer — no server or Python required.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy.sparse import issparse

from .data_loader import ScDataset

# ── Colour palette (same as KaroSpace) ──────────────────────────────────────

DEFAULT_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#5254a3", "#6b6ecf", "#9c9ede", "#637939",
    "#8ca252", "#b5cf6b", "#cedb9c", "#8c6d31", "#bd9e39",
    "#e7ba52", "#e7cb94", "#843c39", "#ad494a", "#d6616b",
    "#e7969c", "#7b4173", "#a55194", "#ce6dbd", "#de9ed6",
]

# ── Gene encoding helpers ────────────────────────────────────────────────────

def _encode_gene(values: np.ndarray, sparse_threshold: float = 0.8) -> dict:
    """Encode a gene expression vector as dense or sparse JSON."""
    vals = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(vals)
    nnz = int(np.count_nonzero(vals[finite]))
    n = int(vals.size)
    zero_frac = 1.0 - (nnz / n) if n else 1.0

    if zero_frac >= sparse_threshold:
        nonzero = finite & (vals != 0)
        i_list = np.where(nonzero)[0].tolist()
        v_list = [round(float(v), 5) for v in vals[nonzero]]
        return {"sparse": {"i": i_list, "v": v_list}}
    else:
        return {"dense": [round(float(v), 5) if np.isfinite(v) else None for v in vals]}


def _select_hvgs(adata, n: int = 20) -> List[str]:
    """Return top n highly variable genes, falling back to highest-variance genes."""
    if "highly_variable" in adata.var.columns:
        hv = adata.var[adata.var["highly_variable"]].index.tolist()
        if hv:
            return hv[:n]
    # Fallback: variance-ranked
    X = adata.X
    if hasattr(X, "toarray"):
        mean_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
        sq_mean = np.asarray(X.mean(axis=0)).ravel() ** 2
        variances = mean_sq - sq_mean
    else:
        variances = np.var(np.asarray(X, dtype=float), axis=0)
    top_idx = np.argsort(variances)[::-1][:n]
    return [adata.var_names[i] for i in top_idx]


def _ensure_categorical_obs(adata, groupby: str) -> Optional[pd.Series]:
    """Return a categorical obs series, or None if the column is unavailable."""
    if groupby not in adata.obs.columns:
        return None
    series = adata.obs[groupby]
    if not isinstance(series.dtype, CategoricalDtype):
        series = series.astype("category")
    return series


def _pct_expr_for_genes(adata, groupby: str, group: str, genes: List[str]) -> Dict[str, float]:
    """Fraction of cells with non-zero expression for each gene inside one group."""
    if not genes:
        return {}
    series = _ensure_categorical_obs(adata, groupby)
    if series is None:
        return {}
    mask = (series == group).to_numpy()
    if not np.any(mask):
        return {}

    valid_genes = [gene for gene in genes if gene in adata.var_names]
    if not valid_genes:
        return {}
    gene_idx = [adata.var_names.get_loc(gene) for gene in valid_genes]
    x = adata[mask, gene_idx].X
    if issparse(x):
        frac = np.asarray((x > 0).mean(axis=0)).ravel()
    else:
        frac = (np.asarray(x) > 0).mean(axis=0)
    return {gene: round(float(value), 5) for gene, value in zip(valid_genes, frac)}


def _compute_marker_genes(
    adata,
    groupby: str,
    top_n: int = 20,
) -> Dict[str, List[Dict[str, Any]]]:
    """Compute top marker genes per category using Scanpy rank_genes_groups."""
    series = _ensure_categorical_obs(adata, groupby)
    if series is None or series.nunique(dropna=True) < 2:
        return {}

    tmp = adata.copy()
    tmp.obs[groupby] = series
    try:
        import scanpy as sc

        sc.tl.rank_genes_groups(tmp, groupby=groupby, method="wilcoxon", n_genes=top_n)
    except Exception as exc:
        print(f"  Warning: marker gene analysis failed for '{groupby}': {exc}")
        return {}

    markers: Dict[str, List[Dict[str, Any]]] = {}
    for group in series.cat.categories:
        try:
            df = sc.get.rank_genes_groups_df(tmp, group=group).head(top_n)
        except Exception as exc:
            print(f"  Warning: could not extract marker genes for '{groupby}'='{group}': {exc}")
            continue
        genes = df["names"].tolist()
        pct_expr = _pct_expr_for_genes(adata, groupby, group, genes)
        entries = []
        for row in df.itertuples(index=False):
            gene = getattr(row, "names", None)
            if gene is None:
                continue
            entries.append({
                "gene": gene,
                "score": round(float(getattr(row, "scores", 0.0)), 5),
                "pct_expr": pct_expr.get(gene, 0.0),
            })
        if entries:
            markers[str(group)] = entries
    return markers


def _compute_cluster_de(
    adata,
    groupby: str,
    top_n: int = 15,
    method: str = "wilcoxon",
    min_cells: int = 20,
) -> Dict[str, Any]:
    """Compute pairwise differential expression for all category pairs."""
    series = _ensure_categorical_obs(adata, groupby)
    if series is None or series.nunique(dropna=True) < 2:
        return {}

    counts = series.value_counts()
    groups = [str(group) for group in series.cat.categories if counts.get(group, 0) >= min_cells]
    if len(groups) < 2:
        return {}

    comparisons: Dict[str, List[Dict[str, Any]]] = {}
    try:
        import scanpy as sc
    except Exception as exc:
        print(f"  Warning: cluster DE unavailable because Scanpy could not be imported: {exc}")
        return {}

    for group_a in groups:
        for group_b in groups:
            if group_a == group_b:
                continue
            tmp = adata.copy()
            tmp.obs[groupby] = series
            try:
                sc.tl.rank_genes_groups(
                    tmp,
                    groupby=groupby,
                    groups=[group_a],
                    reference=group_b,
                    method=method,
                    n_genes=top_n,
                )
                df = sc.get.rank_genes_groups_df(tmp, group=group_a).head(top_n)
            except Exception as exc:
                print(f"  Warning: cluster DE failed for '{groupby}' {group_a} vs {group_b}: {exc}")
                continue

            entries = []
            for row in df.itertuples(index=False):
                gene = getattr(row, "names", None)
                if gene is None:
                    continue
                pval = getattr(row, "pvals_adj", getattr(row, "pvals", np.nan))
                logfc = getattr(row, "logfoldchanges", 0.0)
                entries.append({
                    "gene": gene,
                    "logfc": round(float(logfc), 5) if pd.notna(logfc) else 0.0,
                    "pval": round(float(pval), 8) if pd.notna(pval) else None,
                })
            if entries:
                comparisons[f"{group_a}__vs__{group_b}"] = entries

    if not comparisons:
        return {}
    return {
        "groups": groups,
        "comparisons": comparisons,
    }


def _collect_analytics_genes(analytics: Dict[str, Any]) -> List[str]:
    """Collect genes referenced by analytics so they can be embedded for the viewer."""
    genes: List[str] = []

    for group_map in analytics.get("marker_genes", {}).values():
        if not isinstance(group_map, dict):
            continue
        for entries in group_map.values():
            if not isinstance(entries, list):
                continue
            genes.extend(entry.get("gene") for entry in entries if isinstance(entry, dict))

    for group_map in analytics.get("cluster_de", {}).values():
        if not isinstance(group_map, dict):
            continue
        comparisons = group_map.get("comparisons", {})
        for entries in comparisons.values():
            if not isinstance(entries, list):
                continue
            genes.extend(entry.get("gene") for entry in entries if isinstance(entry, dict))

    return [gene for gene in genes if gene]


def _extract_paga(adata, groupby: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Extract a compact PAGA payload from adata.uns['paga'] if available."""
    paga = adata.uns.get("paga")
    if not isinstance(paga, dict):
        return None

    inferred_groupby = groupby or paga.get("groups")
    if not inferred_groupby or inferred_groupby not in adata.obs.columns:
        return None

    connectivities = paga.get("connectivities")
    if connectivities is None:
        return None

    try:
        dense = connectivities.toarray() if hasattr(connectivities, "toarray") else np.asarray(connectivities)
    except Exception as exc:
        print(f"  Warning: could not extract PAGA connectivities: {exc}")
        return None

    series = _ensure_categorical_obs(adata, inferred_groupby)
    if series is None:
        return None
    categories = [str(cat) for cat in series.cat.categories]
    if dense.shape[0] != len(categories) or dense.shape[1] != len(categories):
        return None

    edges = []
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            weight = dense[i, j]
            if not np.isfinite(weight) or weight <= 0:
                continue
            edges.append({
                "source": categories[i],
                "target": categories[j],
                "weight": round(float(weight), 5),
            })

    if not edges:
        return None
    return {
        "groupby": inferred_groupby,
        "edges": edges,
    }


def _extract_velocity_embeddings(adata) -> Dict[str, Dict[str, List[float]]]:
    """Extract velocity vectors keyed by embedding name when available."""
    result: Dict[str, Dict[str, List[float]]] = {}
    for key in list(adata.obsm.keys()):
        if not key.startswith("velocity_"):
            continue
        try:
            arr = np.asarray(adata.obsm[key], dtype=np.float32)
        except Exception as exc:
            print(f"  Warning: could not extract velocity embedding '{key}': {exc}")
            continue
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue

        suffix = key[len("velocity_"):]
        embedding_key = f"X_{suffix}" if f"X_{suffix}" in adata.obsm else suffix
        result[embedding_key] = {
            "dx": [round(float(v), 5) if np.isfinite(v) else None for v in arr[:, 0]],
            "dy": [round(float(v), 5) if np.isfinite(v) else None for v in arr[:, 1]],
        }
    return result


# ── HTML template ────────────────────────────────────────────────────────────

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --bg: #f5f5f5;
            --text: #1a1a1a;
            --header-bg: #ffffff;
            --panel-bg: #ffffff;
            --border: #e0e0e0;
            --input-bg: #ffffff;
            --muted: #666666;
            --hover-bg: #f0f0f0;
            --accent: #870052;
            --accent-strong: #4F0433;
            --accent-warm: #FF876F;
            --accent-soft: #FEEEEB;
            --selection-outline: rgba(22,22,22,0.45);
        }}
        :root.dark {{
            --bg: #1a1a1a;
            --text: #e0e0e0;
            --header-bg: #2a2a2a;
            --panel-bg: #2a2a2a;
            --border: #404040;
            --input-bg: #333333;
            --muted: #888888;
            --hover-bg: #3a3a3a;
            --selection-outline: rgba(255,255,255,0.55);
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background:
                radial-gradient(800px 500px at 10% 0%, rgba(255,135,111,0.07), transparent),
                radial-gradient(900px 600px at 100% 20%, rgba(135,0,82,0.07), transparent),
                var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            transition: background 0.3s, color 0.3s;
        }}

        /* ── Header ── */
        .header {{
            padding: 8px 16px;
            background:
                linear-gradient(90deg, rgba(255,135,111,0.10), rgba(135,0,82,0.06)),
                var(--header-bg);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 8px;
            transition: background 0.3s, border-color 0.3s;
            position: sticky; top: 0; z-index: 10;
        }}
        .header-left {{ display: flex; align-items: center; gap: 12px; }}
        .header h1 {{ font-size: 16px; font-weight: 700; letter-spacing: -0.01em; }}
        .header h1 span {{ color: var(--accent); }}
        .stats {{ font-size: 11px; color: var(--muted); }}
        .controls {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
        .control-group {{ display: flex; align-items: center; gap: 4px; }}
        .control-group label {{ font-size: 11px; color: var(--muted); }}
        select, input[type="text"] {{
            padding: 5px 8px;
            border: 1px solid var(--border);
            border-radius: 4px;
            background: var(--input-bg);
            color: var(--text);
            font-size: 12px;
            transition: background 0.2s, border-color 0.2s;
        }}
        select {{ min-width: 110px; }}
        select:focus, input:focus {{
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(135,0,82,0.15);
        }}

        /* Gene input */
        .gene-input-shell {{ position: relative; }}
        .gene-input-shell input {{ width: 160px; }}
        .gene-clear-btn {{
            position: absolute; right: 6px; top: 50%; transform: translateY(-50%);
            background: none; border: none; color: var(--muted); cursor: pointer;
            font-size: 14px; line-height: 1; padding: 0; display: none;
        }}
        .gene-clear-btn.visible {{ display: block; }}

        /* Gene discovery panel */
        .gene-discovery-panel {{
            position: absolute; top: calc(100% + 6px); left: 0;
            width: min(340px, calc(100vw - 32px));
            max-height: 380px; overflow-y: auto;
            padding: 10px;
            border: 1px solid var(--border); border-radius: 10px;
            background:
                linear-gradient(180deg, rgba(255,135,111,0.07), rgba(135,0,82,0.02)),
                var(--panel-bg);
            box-shadow: 0 14px 36px rgba(0,0,0,0.14);
            display: none; z-index: 50;
        }}
        .gene-discovery-panel.open {{ display: block; }}
        .discovery-section + .discovery-section {{
            margin-top: 10px; padding-top: 10px;
            border-top: 1px solid rgba(127,127,127,0.15);
        }}
        .discovery-label {{
            font-size: 10px; font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.04em; color: var(--muted); margin-bottom: 6px;
        }}
        .gene-chip-grid {{ display: flex; flex-wrap: wrap; gap: 5px; }}
        .gene-chip {{
            display: inline-flex; align-items: center; gap: 4px;
            padding: 4px 8px; border: 1px solid var(--border); border-radius: 999px;
            background: var(--input-bg); color: var(--text);
            cursor: pointer; font-size: 11px;
            transition: background 0.15s, border-color 0.15s, transform 0.15s;
        }}
        .gene-chip:hover {{ background: var(--hover-bg); border-color: var(--accent); transform: translateY(-1px); }}
        .gene-chip.active {{ border-color: var(--accent); background: rgba(135,0,82,0.08); }}
        .discovery-empty {{ font-size: 11px; color: var(--muted); }}

        /* Spot size + icon buttons */
        .size-control {{ display: inline-flex; align-items: center; gap: 5px; }}
        .size-step, .icon-btn {{
            width: 26px; height: 26px;
            border: 1px solid var(--border); border-radius: 4px;
            background: var(--input-bg); color: var(--text);
            cursor: pointer; font-size: 13px; line-height: 1;
            display: inline-flex; align-items: center; justify-content: center;
            transition: background 0.2s;
        }}
        .size-step:hover, .icon-btn:hover {{ background: var(--hover-bg); }}
        .size-label {{ font-size: 11px; color: var(--muted); min-width: 24px; text-align: center; }}

        /* ── Main layout ── */
        .main-container {{
            display: flex; flex: 1; min-height: 0;
            overflow: hidden;
        }}
        .content-column {{
            flex: 1; min-width: 0;
            display: flex; flex-direction: column;
        }}
        .grid-container {{
            flex: 1; overflow: auto;
            padding: 10px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(clamp(180px, 22vw, 320px), 1fr));
            gap: 10px;
            align-content: start;
        }}
        .grid-container.single-view-layout {{
            grid-template-columns: minmax(320px, min(100%, 700px));
            justify-content: center;
        }}
        .grid-group-header {{
            grid-column: 1 / -1;
            padding: 4px 2px 0;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: var(--accent);
        }}

        /* ── View panel ── */
        .view-panel {{
            background: var(--panel-bg);
            border: 1px solid var(--border); border-radius: 8px;
            overflow: hidden; cursor: pointer;
            transition: box-shadow 0.2s, transform 0.2s, border-color 0.3s;
            display: flex; flex-direction: column;
            position: relative;
        }}
        .view-panel:hover {{
            box-shadow: 0 4px 16px rgba(0,0,0,0.14);
            transform: translateY(-2px);
            border-color: rgba(135,0,82,0.3);
        }}
        .panel-label {{
            padding: 5px 10px;
            font-size: 11px; font-weight: 600; color: var(--muted);
            letter-spacing: 0.03em; text-transform: uppercase;
            border-bottom: 1px solid var(--border);
            background: var(--header-bg);
        }}
        .view-canvas {{
            display: block; width: 100%;
            aspect-ratio: 1 / 1;
        }}
        .panel-actions {{
            position: absolute;
            top: 8px;
            right: 8px;
            display: flex;
            gap: 4px;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.15s;
        }}
        .view-panel:hover .panel-actions {{
            opacity: 1;
            pointer-events: auto;
        }}
        .panel-toggle {{
            width: 24px;
            height: 24px;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: rgba(255,255,255,0.88);
            color: var(--text);
            cursor: pointer;
            font-size: 11px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s, border-color 0.2s, transform 0.2s;
        }}
        .dark .panel-toggle {{
            background: rgba(42,42,42,0.9);
        }}
        .panel-toggle:hover {{
            background: var(--hover-bg);
            border-color: var(--accent);
            transform: translateY(-1px);
        }}
        .panel-toggle.active {{
            border-color: var(--accent);
            color: var(--accent);
            background: rgba(135,0,82,0.12);
        }}

        /* ── Legend panel ── */
        .legend-panel {{
            width: 190px; flex-shrink: 0;
            border-left: 1px solid var(--border);
            background: var(--panel-bg);
            overflow-y: auto;
            padding: 12px 10px;
            transition: border-color 0.3s, background 0.3s;
        }}
        .legend-title {{
            font-size: 11px; font-weight: 600; color: var(--muted);
            text-transform: uppercase; letter-spacing: 0.04em;
            margin-bottom: 8px;
            word-break: break-word;
        }}
        .legend-item {{
            display: flex; align-items: center; gap: 6px;
            padding: 3px 4px; border-radius: 4px; cursor: pointer;
            transition: background 0.15s;
        }}
        .legend-item:hover {{ background: var(--hover-bg); }}
        .legend-item.hidden-cat {{ opacity: 0.35; }}
        .legend-item.spotlight .legend-swatch {{ box-shadow: 0 0 0 2px var(--accent); }}
        .legend-swatch {{
            width: 10px; height: 10px; border-radius: 50%;
            flex-shrink: 0;
        }}
        .legend-label {{
            font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
        }}
        .legend-hint {{
            font-size: 10px; color: var(--muted); margin-top: 6px; line-height: 1.4;
        }}
        .legend-reset-btn {{
            margin-top: 8px; padding: 3px 10px;
            border: 1px solid var(--border); border-radius: 999px;
            background: var(--input-bg); color: var(--text);
            cursor: pointer; font-size: 10px;
            transition: background 0.2s;
        }}
        .legend-reset-btn:hover {{ background: var(--hover-bg); }}

        /* Colorbar */
        .colorbar-wrapper {{ display: flex; flex-direction: column; gap: 4px; align-items: flex-start; }}
        .colorbar-row {{ display: flex; align-items: stretch; gap: 6px; }}
        .colorbar-canvas {{ border-radius: 3px; flex-shrink: 0; }}
        .colorbar-labels {{ display: flex; flex-direction: column; justify-content: space-between; font-size: 10px; color: var(--muted); }}

        /* ── Modal overlay ── */
        .modal-overlay {{
            position: fixed; inset: 0;
            background: rgba(0,0,0,0.55);
            z-index: 100;
            display: flex; align-items: center; justify-content: center;
            backdrop-filter: blur(2px);
        }}
        .modal-overlay.hidden {{ display: none; }}
        .modal-container {{
            width: min(96vw, 1200px);
            height: min(92vh, 820px);
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            display: flex; flex-direction: column;
            overflow: hidden;
            box-shadow: 0 24px 80px rgba(0,0,0,0.3);
        }}
        .modal-header {{
            padding: 10px 14px;
            border-bottom: 1px solid var(--border);
            display: flex; align-items: center; justify-content: space-between;
            background: var(--header-bg);
            flex-shrink: 0;
        }}
        .modal-header-left {{ display: flex; align-items: center; gap: 10px; }}
        .modal-title {{ font-size: 13px; font-weight: 600; }}
        .modal-toolbar {{ display: flex; align-items: center; gap: 4px; }}
        .toolbar-btn {{
            padding: 4px 8px;
            border: 1px solid var(--border); border-radius: 4px;
            background: var(--input-bg); color: var(--text);
            cursor: pointer; font-size: 13px;
            transition: background 0.2s, border-color 0.2s;
        }}
        .toolbar-btn:hover {{ background: var(--hover-bg); }}
        .toolbar-btn.active {{
            background: rgba(135,0,82,0.12);
            border-color: var(--accent); color: var(--accent);
        }}
        .modal-close {{
            width: 28px; height: 28px;
            border: 1px solid var(--border); border-radius: 4px;
            background: var(--input-bg); color: var(--text);
            cursor: pointer; font-size: 16px;
            display: flex; align-items: center; justify-content: center;
            transition: background 0.2s;
        }}
        .modal-close:hover {{ background: var(--hover-bg); }}
        .modal-body {{
            flex: 1; min-height: 0; display: flex;
        }}
        .modal-canvas-wrapper {{
            flex: 1; min-width: 0; position: relative;
            display: flex; flex-direction: column;
        }}
        #modal-canvas {{
            display: block; flex: 1; min-height: 0;
            touch-action: none;
        }}
        .modal-zoom-hint {{
            position: absolute; bottom: 8px; left: 8px;
            font-size: 10px; color: var(--muted);
            background: var(--panel-bg); padding: 3px 7px;
            border: 1px solid var(--border); border-radius: 4px;
            pointer-events: none;
        }}

        /* ── Insights sidebar ── */
        .insights-panel {{
            width: 260px; flex-shrink: 0;
            border-left: 1px solid var(--border);
            display: flex; flex-direction: column;
            overflow: hidden;
        }}
        .insights-tabs {{
            display: flex; border-bottom: 1px solid var(--border);
            flex-shrink: 0;
        }}
        .insights-tab {{
            flex: 1; padding: 8px;
            border: none; background: none; color: var(--muted);
            cursor: pointer; font-size: 11px; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.03em;
            border-bottom: 2px solid transparent;
            transition: color 0.2s, border-color 0.2s;
        }}
        .insights-tab.active {{
            color: var(--accent); border-bottom-color: var(--accent);
        }}
        .insights-tab:hover {{ color: var(--text); }}
        .insights-content {{
            flex: 1; overflow-y: auto; padding: 12px;
        }}
        .no-selection-msg {{
            text-align: center; padding: 24px 12px;
            color: var(--muted); font-size: 12px; line-height: 1.5;
        }}
        .no-selection-msg .hint {{
            margin-top: 8px; font-size: 11px;
        }}
        .stats-section {{ margin-bottom: 14px; }}
        .stats-section-title {{
            font-size: 10px; font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.04em; color: var(--muted); margin-bottom: 6px;
        }}
        .stats-row {{
            display: flex; align-items: center; justify-content: space-between;
            padding: 3px 0;
        }}
        .stats-key {{ font-size: 11px; display: flex; align-items: center; gap: 5px; }}
        .stats-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
        .stats-val {{ font-size: 11px; color: var(--muted); font-variant-numeric: tabular-nums; }}
        .stats-bar-wrap {{
            height: 3px; background: var(--border); border-radius: 2px;
            margin-top: 2px; overflow: hidden;
        }}
        .stats-bar {{ height: 100%; border-radius: 2px; }}
        .insights-pane.hidden {{ display: none; }}
        .insights-empty {{
            color: var(--muted);
            font-size: 11px;
            line-height: 1.5;
        }}
        .insights-controls {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-bottom: 10px;
        }}
        .insights-control label {{
            display: block;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--muted);
            margin-bottom: 4px;
        }}
        .insights-control select,
        .insights-control input {{
            width: 100%;
            min-width: 0;
        }}
        .marker-list {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .marker-row {{
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(127,127,127,0.12);
        }}
        .marker-head {{
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 4px;
        }}
        .marker-gene {{
            font-size: 12px;
            font-weight: 600;
        }}
        .marker-meta {{
            font-size: 10px;
            color: var(--muted);
        }}
        .marker-bar-track {{
            height: 6px;
            border-radius: 999px;
            background: var(--border);
            overflow: hidden;
        }}
        .marker-bar-fill {{
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent), var(--accent-warm));
        }}
        .de-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
        }}
        .de-table th,
        .de-table td {{
            padding: 5px 0;
            text-align: left;
            border-bottom: 1px solid rgba(127,127,127,0.12);
            vertical-align: middle;
        }}
        .de-bar-track {{
            width: 84px;
            height: 6px;
            border-radius: 999px;
            background: var(--border);
            overflow: hidden;
        }}
        .de-bar-fill {{
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent), var(--accent-warm));
        }}
        .dotplot-shell {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .dotplot-canvas {{
            width: 100%;
            height: 260px;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--panel-bg);
        }}
        .dotplot-note {{
            font-size: 10px;
            color: var(--muted);
            line-height: 1.4;
        }}

        /* ── Tooltip ── */
        .cell-tooltip {{
            position: fixed; pointer-events: none;
            padding: 5px 9px; border-radius: 5px;
            background: var(--header-bg); border: 1px solid var(--border);
            font-size: 11px; color: var(--text);
            box-shadow: 0 4px 12px rgba(0,0,0,0.12);
            display: none; z-index: 200;
        }}
        .cell-tooltip.visible {{ display: block; }}

        /* ── Loading overlay ── */
        .loading-overlay {{
            position: fixed; inset: 0; z-index: 999;
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            background: var(--bg); gap: 12px;
        }}
        .loading-spinner {{
            width: 32px; height: 32px;
            border: 3px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        .loading-text {{ font-size: 13px; color: var(--muted); }}

        @media (max-width: 640px) {{
            .legend-panel {{ display: none; }}
            .insights-panel {{ width: 200px; }}
        }}
    </style>
</head>
<body>
    <div id="loading-overlay" class="loading-overlay">
        <div class="loading-spinner"></div>
        <div class="loading-text">Rendering…</div>
    </div>

    <div class="header">
        <div class="header-left">
            <h1>sc<span>Karo</span>Space</h1>
            <span id="stats-text" class="stats">Loading…</span>
        </div>
        <div class="controls">
            <div class="control-group">
                <label for="color-select">Color</label>
                <select id="color-select"></select>
            </div>
            <div class="control-group" style="position:relative;">
                <label for="gene-input">Gene</label>
                <div class="gene-input-shell">
                    <input type="text" id="gene-input" placeholder="gene name…"
                           autocomplete="off" spellcheck="false" list="gene-datalist" />
                    <datalist id="gene-datalist"></datalist>
                    <button class="gene-clear-btn" id="gene-clear-btn" title="Clear gene">✕</button>
                    <div class="gene-discovery-panel" id="gene-discovery-panel"></div>
                </div>
            </div>
            <div class="control-group">
                <label>Size</label>
                <div class="size-control">
                    <button class="size-step" id="size-down" title="Smaller">−</button>
                    <span class="size-label" id="size-label">1×</span>
                    <button class="size-step" id="size-up" title="Larger">+</button>
                </div>
            </div>
            <button class="icon-btn" id="theme-btn" title="Toggle theme (T)">☀</button>
            <button class="icon-btn" id="screenshot-btn" title="Screenshot (S)">⬇</button>
        </div>
    </div>

    <div class="main-container">
        <div class="content-column">
            <div id="grid" class="grid-container"></div>
        </div>
        <div id="legend" class="legend-panel"></div>
    </div>

    <!-- Modal detail view -->
    <div id="modal-overlay" class="modal-overlay hidden">
        <div class="modal-container">
            <div class="modal-header">
                <div class="modal-header-left">
                    <span class="modal-title" id="modal-title"></span>
                    <div class="modal-toolbar">
                        <button class="toolbar-btn active" id="btn-pan" title="Pan / zoom (P)">✥</button>
                        <button class="toolbar-btn" id="btn-lasso" title="Lasso select (L)">⊙</button>
                        <button class="toolbar-btn" id="btn-clear-sel" title="Clear selection (X)">✕ sel</button>
                        <button class="toolbar-btn" id="btn-hulls" title="Toggle convex hulls">Hulls</button>
                        <button class="toolbar-btn" id="btn-density" title="Toggle density contours">Density</button>
                        <button class="toolbar-btn" id="btn-paga" title="Toggle PAGA graph">PAGA</button>
                        <button class="toolbar-btn" id="btn-velocity" title="Toggle velocity arrows">Vel</button>
                    </div>
                </div>
                <button class="modal-close" id="modal-close" title="Close (Esc)">✕</button>
            </div>
            <div class="modal-body">
                <div class="modal-canvas-wrapper">
                    <canvas id="modal-canvas"></canvas>
                    <div class="modal-zoom-hint" id="zoom-hint">Scroll to zoom · drag to pan · P/L to switch mode</div>
                </div>
                <div class="insights-panel">
                    <div class="insights-tabs">
                        <button class="insights-tab active" data-tab="stats">Stats</button>
                        <button class="insights-tab" data-tab="markers">Markers</button>
                        <button class="insights-tab" data-tab="compare">Compare</button>
                        <button class="insights-tab" data-tab="dotplot">Dotplot</button>
                    </div>
                    <div class="insights-content">
                        <div class="insights-pane" id="insights-stats">
                            <div class="no-selection-msg">
                                <div>No cells selected</div>
                                <div class="hint">Switch to <b>lasso</b> mode and draw a selection</div>
                            </div>
                        </div>
                        <div class="insights-pane hidden" id="insights-markers"></div>
                        <div class="insights-pane hidden" id="insights-compare"></div>
                        <div class="insights-pane hidden" id="insights-dotplot"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="cell-tooltip" id="cell-tooltip"></div>

    <script id="sckaro-data" type="application/json">{data_json}</script>
    <script>
    // ── Parse payload ──────────────────────────────────────────────────────
    const DATA = JSON.parse(document.getElementById('sckaro-data').textContent);
    const PALETTE = {palette_json};

    // ── State ──────────────────────────────────────────────────────────────
    let currentColor = DATA.color;
    let currentGene  = null;   // set when coloring by gene expression
    let spotSize     = DATA.spot_size || 3.0;
    const SPOT_STEPS = [0.5, 0.8, 1.2, 1.6, 2.0, 2.8, 3.5, 4.5, 6.0, 8.0];
    let spotStepIdx  = SPOT_STEPS.reduce((best, v, i) =>
        Math.abs(v - spotSize) < Math.abs(SPOT_STEPS[best] - spotSize) ? i : best, 4);

    let hiddenCategories  = new Set();
    let spotlightCategory = null;
    let selectedCells     = new Set();  // global cell indices
    let hoveredCell       = null;       // global cell index
    let theme             = DATA.theme || 'light';
    const geneCache       = new Map();

    // Modal state
    let modalViewId  = null;
    let modalZoom    = 1.0;
    let modalPanX    = 0;
    let modalPanY    = 0;
    let modalMode    = 'pan';  // 'pan' | 'lasso'
    let lassoPoints  = [];
    let isPointerDown = false;
    let lastPX = 0, lastPY = 0;
    let recentGenes  = [];
    let activeInsightsTab = 'stats';
    let compareGroupby = null;
    let compareGroupA = null;
    let compareGroupB = null;
    let dotplotGeneText = '';
    let showHulls = false;
    let showDensityContours = false;
    let showPaga = false;
    let showVelocity = false;

    let renderAllJobId = 0;

    // ── Theme ──────────────────────────────────────────────────────────────
    function applyTheme(t) {{
        theme = t;
        document.documentElement.classList.toggle('dark',  t === 'dark');
        document.documentElement.classList.toggle('light', t !== 'dark');
        document.getElementById('theme-btn').textContent = t === 'dark' ? '☽' : '☀';
    }}
    function getPanelBg()      {{ return theme === 'dark' ? '#2a2a2a' : '#ffffff'; }}
    function getSelColor()     {{ return theme === 'dark' ? 'rgba(255,255,255,0.6)' : 'rgba(22,22,22,0.45)'; }}
    function getDimColor()     {{ return theme === 'dark' ? '#555555' : '#cccccc'; }}
    function getHiddenColor()  {{ return theme === 'dark' ? '#444444' : '#cccccc'; }}

    // ── Magma colormap ─────────────────────────────────────────────────────
    const MAGMA = [
        [0.001,0.000,0.015],[0.092,0.047,0.256],[0.235,0.073,0.386],
        [0.388,0.100,0.451],[0.531,0.136,0.430],[0.651,0.188,0.392],
        [0.741,0.259,0.331],[0.813,0.354,0.255],[0.870,0.477,0.171],
        [0.918,0.624,0.110],[0.987,0.855,0.185]
    ];
    function magmaRgb(t) {{
        const idx = Math.min(Math.floor(t * 10), 9);
        const f = t * 10 - idx;
        const [r1,g1,b1] = MAGMA[idx], [r2,g2,b2] = MAGMA[idx+1];
        return [
            Math.round((r1 + f*(r2-r1)) * 255),
            Math.round((g1 + f*(g2-g1)) * 255),
            Math.round((b1 + f*(b2-b1)) * 255),
        ];
    }}
    function magma(t) {{
        const [r,g,b] = magmaRgb(Math.max(0, Math.min(1, t)));
        return `rgb(${{r}},${{g}},${{b}})`;
    }}

    // ── Data accessors ─────────────────────────────────────────────────────
    function getView(id) {{ return DATA.views.find(v => v.id === id); }}

    function getColorConfig() {{
        if (currentGene) {{
            const m = DATA.genes_meta[currentGene] || {{}};
            return {{ is_continuous: true, vmin: m.vmin ?? 0, vmax: m.vmax ?? 1 }};
        }}
        return DATA.color_configs[currentColor] || null;
    }}

    function getGeneValues(gene) {{
        if (geneCache.has(gene)) return geneCache.get(gene);
        const entry = DATA.genes && DATA.genes[gene];
        if (!entry) return null;
        let vals;
        if (entry.dense) {{
            vals = entry.dense;
        }} else if (entry.sparse) {{
            vals = new Float32Array(DATA.n_cells);
            const {{i, v}} = entry.sparse;
            for (let k = 0; k < i.length; k++) vals[i[k]] = v[k];
        }} else {{
            return null;
        }}
        geneCache.set(gene, vals);
        return vals;
    }}

    function getColorValues() {{
        if (currentGene) return getGeneValues(currentGene);
        const cfg = DATA.color_configs[currentColor];
        if (!cfg) return null;
        return cfg.is_continuous ? cfg.values : cfg.codes;
    }}

    // Map a local cell index within a view to a global cell index
    function toGlobal(view, localIdx) {{
        return view.cell_indices ? view.cell_indices[localIdx] : localIdx;
    }}

    function getLocalColorValues(view) {{
        const global = getColorValues();
        if (!global) return null;
        if (!view.cell_indices) return global;
        // Return a view-local slice
        return view.cell_indices.map(gi => global[gi]);
    }}

    function findNearestCell(view, tf, sx, sy, maxDist=10) {{
        if (!view || !view.n_cells) return null;
        let bestIdx = null;
        let bestDist2 = maxDist * maxDist;
        for (let i = 0; i < view.n_cells; i++) {{
            const {{x, y}} = tf.dataToScreen(view.x[i], view.y[i]);
            const dx = sx - x;
            const dy = sy - y;
            const dist2 = dx*dx + dy*dy;
            if (dist2 <= bestDist2) {{
                bestDist2 = dist2;
                bestIdx = i;
            }}
        }}
        return bestIdx;
    }}

    function updateHoveredCell(nextHovered) {{
        const normalized = Number.isInteger(nextHovered) ? nextHovered : null;
        if (hoveredCell === normalized) return;
        hoveredCell = normalized;
        renderAllViews();
        if (modalViewId) renderModal();
    }}

    function getCategoryColor(catIdx) {{
        return PALETTE[((catIdx % PALETTE.length) + PALETTE.length) % PALETTE.length];
    }}

    function getCategoryLabelMap(cfg) {{
        if (!cfg || cfg.is_continuous || !cfg.categories) return null;
        const map = new Map();
        cfg.categories.forEach((cat, idx) => map.set(cat, idx));
        return map;
    }}

    function getOverlayAlpha(base, focused) {{
        return focused ? base : Math.max(0.02, base * 0.75);
    }}

    function convexHull(points) {{
        if (!points || points.length < 3) return points || [];
        const sorted = points.slice().sort((a, b) => a.x === b.x ? a.y - b.y : a.x - b.x);
        const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
        const lower = [];
        for (const p of sorted) {{
            while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {{
                lower.pop();
            }}
            lower.push(p);
        }}
        const upper = [];
        for (let i = sorted.length - 1; i >= 0; i--) {{
            const p = sorted[i];
            while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {{
                upper.pop();
            }}
            upper.push(p);
        }}
        upper.pop();
        lower.pop();
        return lower.concat(upper);
    }}

    function hullCentroid(points) {{
        if (!points || !points.length) return null;
        let x = 0, y = 0;
        points.forEach(p => {{ x += p.x; y += p.y; }});
        return {{x: x / points.length, y: y / points.length}};
    }}

    function drawConvexHulls(ctx, view, tf, cfg, values) {{
        if (!showHulls || !cfg || cfg.is_continuous || !cfg.categories) return;
        const pointsByCat = new Map();
        for (let i = 0; i < view.n_cells; i++) {{
            const val = values[i];
            if (val == null || !isFinite(val)) continue;
            const catIdx = Math.round(val);
            const cat = cfg.categories[catIdx];
            if (!cat || hiddenCategories.has(cat)) continue;
            const bucket = pointsByCat.get(catIdx) || [];
            bucket.push(tf.dataToScreen(view.x[i], view.y[i]));
            pointsByCat.set(catIdx, bucket);
        }}
        const focused = spotlightCategory !== null;
        pointsByCat.forEach((pts, catIdx) => {{
            if (pts.length < 3) return;
            const cat = cfg.categories[catIdx];
            const hull = convexHull(pts);
            if (hull.length < 3) return;
            const color = getCategoryColor(catIdx);
            const active = !focused || cat === spotlightCategory;
            ctx.save();
            ctx.globalAlpha = getOverlayAlpha(0.10, active);
            ctx.fillStyle = color;
            ctx.strokeStyle = color;
            ctx.lineWidth = active ? 1.25 : 0.9;
            ctx.beginPath();
            ctx.moveTo(hull[0].x, hull[0].y);
            for (let i = 1; i < hull.length; i++) ctx.lineTo(hull[i].x, hull[i].y);
            ctx.closePath();
            ctx.fill();
            ctx.globalAlpha = getOverlayAlpha(0.42, active);
            ctx.stroke();
            const centroid = hullCentroid(hull);
            if (centroid) {{
                ctx.globalAlpha = 0.9;
                ctx.fillStyle = theme === 'dark' ? '#f0f0f0' : '#222222';
                ctx.font = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(cat, centroid.x, centroid.y);
            }}
            ctx.restore();
        }});
    }}

    function computeDensityGrid(view, tf, width, height, resolution=50) {{
        const minX = 0;
        const minY = 0;
        const maxX = width;
        const maxY = height;
        const cellW = Math.max(1, (maxX - minX) / resolution);
        const cellH = Math.max(1, (maxY - minY) / resolution);
        const grid = Array.from({{length: resolution}}, () => new Float32Array(resolution));
        for (let i = 0; i < view.n_cells; i++) {{
            const p = tf.dataToScreen(view.x[i], view.y[i]);
            const gx = Math.max(0, Math.min(resolution - 1, Math.floor((p.x - minX) / cellW)));
            const gy = Math.max(0, Math.min(resolution - 1, Math.floor((p.y - minY) / cellH)));
            for (let oy = -1; oy <= 1; oy++) {{
                for (let ox = -1; ox <= 1; ox++) {{
                    const x = gx + ox, y = gy + oy;
                    if (x < 0 || y < 0 || x >= resolution || y >= resolution) continue;
                    const weight = ox === 0 && oy === 0 ? 1.0 : 0.35;
                    grid[y][x] += weight;
                }}
            }}
        }}
        return {{grid, resolution, cellW, cellH, minX, minY}};
    }}

    function traceContourCell(ctx, x, y, sizeX, sizeY, state) {{
        const xm = x + sizeX / 2;
        const ym = y + sizeY / 2;
        const segments = {{
            1: [[x, ym], [xm, y + sizeY]],
            2: [[xm, y + sizeY], [x + sizeX, ym]],
            3: [[x, ym], [x + sizeX, ym]],
            4: [[xm, y], [x + sizeX, ym]],
            5: [[x, ym], [xm, y], [xm, y + sizeY], [x + sizeX, ym]],
            6: [[xm, y], [xm, y + sizeY]],
            7: [[x, ym], [xm, y]],
            8: [[x, ym], [xm, y]],
            9: [[xm, y], [xm, y + sizeY]],
            10:[[xm, y], [x + sizeX, ym], [x, ym], [xm, y + sizeY]],
            11:[[xm, y], [x + sizeX, ym]],
            12:[[x, ym], [x + sizeX, ym]],
            13:[[xm, y + sizeY], [x + sizeX, ym]],
            14:[[x, ym], [xm, y + sizeY]],
        }};
        const seg = segments[state];
        if (!seg) return;
        ctx.beginPath();
        ctx.moveTo(seg[0][0], seg[0][1]);
        for (let i = 1; i < seg.length; i++) ctx.lineTo(seg[i][0], seg[i][1]);
        ctx.stroke();
    }}

    function drawDensityContours(ctx, view, tf, width, height) {{
        if (!showDensityContours || view.n_cells < 10) return;
        const density = computeDensityGrid(view, tf, width, height, 52);
        let maxVal = 0;
        for (const row of density.grid) {{
            for (const val of row) maxVal = Math.max(maxVal, val);
        }}
        if (maxVal <= 0) return;
        const levels = [0.18, 0.33, 0.5, 0.7].map(v => v * maxVal);
        ctx.save();
        ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.42)' : 'rgba(34,34,34,0.28)';
        ctx.lineWidth = 1;
        levels.forEach(level => {{
            for (let y = 0; y < density.resolution - 1; y++) {{
                for (let x = 0; x < density.resolution - 1; x++) {{
                    const tl = density.grid[y][x] >= level ? 1 : 0;
                    const tr = density.grid[y][x + 1] >= level ? 1 : 0;
                    const br = density.grid[y + 1][x + 1] >= level ? 1 : 0;
                    const bl = density.grid[y + 1][x] >= level ? 1 : 0;
                    const state = tl * 8 + tr * 4 + br * 2 + bl;
                    if (state === 0 || state === 15) continue;
                    traceContourCell(
                        ctx,
                        density.minX + x * density.cellW,
                        density.minY + y * density.cellH,
                        density.cellW,
                        density.cellH,
                        state,
                    );
                }}
            }}
        }});
        ctx.restore();
    }}

    function getVelocityVectors(view) {{
        const velMap = DATA.velocity_embedding || {{}};
        const entry = velMap[view.embedding_key];
        if (!entry || !entry.dx || !entry.dy) return null;
        if (!view.cell_indices) return entry;
        return {{
            dx: view.cell_indices.map(gi => entry.dx[gi]),
            dy: view.cell_indices.map(gi => entry.dy[gi]),
        }};
    }}

    function drawVelocity(ctx, view, tf) {{
        if (!showVelocity) return;
        const velocity = getVelocityVectors(view);
        if (!velocity) return;
        const step = Math.max(1, Math.floor(view.n_cells / 500));
        ctx.save();
        ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.5)' : 'rgba(20,20,20,0.4)';
        ctx.lineWidth = 0.9;
        for (let i = 0; i < view.n_cells; i += step) {{
            const dx = velocity.dx[i];
            const dy = velocity.dy[i];
            if (!Number.isFinite(dx) || !Number.isFinite(dy)) continue;
            const start = tf.dataToScreen(view.x[i], view.y[i]);
            const end = tf.dataToScreen(view.x[i] + dx, view.y[i] + dy);
            const vx = end.x - start.x;
            const vy = end.y - start.y;
            const len = Math.hypot(vx, vy);
            if (len < 2) continue;
            const scale = Math.min(10, Math.max(3, len));
            const ux = vx / len;
            const uy = vy / len;
            const ex = start.x + ux * scale;
            const ey = start.y + uy * scale;
            ctx.beginPath();
            ctx.moveTo(start.x, start.y);
            ctx.lineTo(ex, ey);
            ctx.lineTo(ex - ux * 3 - uy * 2, ey - uy * 3 + ux * 2);
            ctx.moveTo(ex, ey);
            ctx.lineTo(ex - ux * 3 + uy * 2, ey - uy * 3 - ux * 2);
            ctx.stroke();
        }}
        ctx.restore();
    }}

    function drawPAGA(ctx, view, tf, cfg, values) {{
        if (!showPaga || !DATA.paga || !cfg || cfg.is_continuous || !cfg.categories) return;
        if (DATA.paga.groupby !== currentColor) return;
        const centroids = new Map();
        const counts = new Map();
        for (let i = 0; i < view.n_cells; i++) {{
            const raw = values[i];
            if (raw == null || !Number.isFinite(raw)) continue;
            const code = Math.round(raw);
            const cat = cfg.categories[code];
            if (!cat || hiddenCategories.has(cat)) continue;
            const p = tf.dataToScreen(view.x[i], view.y[i]);
            const prev = centroids.get(cat) || {{x: 0, y: 0}};
            centroids.set(cat, {{x: prev.x + p.x, y: prev.y + p.y}});
            counts.set(cat, (counts.get(cat) || 0) + 1);
        }}
        centroids.forEach((val, cat) => {{
            const n = counts.get(cat) || 1;
            val.x /= n;
            val.y /= n;
        }});
        ctx.save();
        ctx.strokeStyle = theme === 'dark' ? 'rgba(255,255,255,0.35)' : 'rgba(30,30,30,0.28)';
        DATA.paga.edges.forEach(edge => {{
            const a = edge.source;
            const b = edge.target;
            if (!centroids.has(a) || !centroids.has(b)) return;
            const p1 = centroids.get(a), p2 = centroids.get(b);
            ctx.lineWidth = 0.8 + edge.weight * 4;
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
        }});
        ctx.fillStyle = theme === 'dark' ? '#f0f0f0' : '#222222';
        ctx.font = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
        centroids.forEach((p, cat) => {{
            ctx.beginPath();
            ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillText(cat, p.x + 6, p.y - 6);
        }});
        ctx.restore();
    }}

    function syncOverlayButtons() {{
        const pairs = [
            ['btn-hulls', showHulls],
            ['btn-density', showDensityContours],
            ['btn-paga', showPaga],
            ['btn-velocity', showVelocity],
        ];
        pairs.forEach(([id, active]) => {{
            const el = document.getElementById(id);
            if (el) el.classList.toggle('active', active);
        }});
        document.querySelectorAll('.panel-toggle[data-overlay]').forEach(btn => {{
            const name = btn.dataset.overlay;
            const active = (
                (name === 'hulls' && showHulls) ||
                (name === 'density' && showDensityContours) ||
                (name === 'paga' && showPaga) ||
                (name === 'velocity' && showVelocity)
            );
            btn.classList.toggle('active', active);
        }});
    }}

    // ── View transform ─────────────────────────────────────────────────────
    function createViewTransform(view, {{width, height, padding=8, zoom=1, panX=0, panY=0}}) {{
        const xr = (view.xmax - view.xmin) || 1;
        const yr = (view.ymax - view.ymin) || 1;
        const scale = Math.min((width-2*padding)/xr, (height-2*padding)/yr) * zoom;
        const cx = width/2 + panX;
        const cy = height/2 + panY;
        const dcx = (view.xmin + view.xmax) / 2;
        const dcy = (view.ymin + view.ymax) / 2;
        return {{
            scale, cx, cy, dcx, dcy,
            dataToScreen(x, y) {{
                return {{x: cx + (x-dcx)*scale, y: cy - (y-dcy)*scale}};
            }},
            screenToData(sx, sy) {{
                return {{x: dcx + (sx-cx)/scale, y: dcy - (sy-cy)/scale}};
            }},
        }};
    }}

    // ── Core renderer ──────────────────────────────────────────────────────
    function renderView(view, canvas, isModal) {{
        const ctx  = canvas.getContext('2d');
        const dpr  = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        if (!rect.width || !rect.height) return;
        canvas.width  = rect.width  * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);

        const w = rect.width, h = rect.height;
        ctx.fillStyle = getPanelBg();
        ctx.fillRect(0, 0, w, h);
        if (!view.n_cells) return;

        const tf = createViewTransform(view, {{
            width: w, height: h,
            padding: isModal ? 20 : 8,
            zoom:  isModal ? modalZoom : 1,
            panX:  isModal ? modalPanX : 0,
            panY:  isModal ? modalPanY : 0,
        }});

        const values = getLocalColorValues(view);
        const cfg    = getColorConfig();
        if (!values || !cfg) return;

        const r = Math.max(0.5, SPOT_STEPS[spotStepIdx]);

        drawDensityContours(ctx, view, tf, w, h);
        drawConvexHulls(ctx, view, tf, cfg, values);

        // Pass 1 — hidden categories (ghost)
        if (hiddenCategories.size > 0 && !cfg.is_continuous) {{
            ctx.globalAlpha = 0.18;
            ctx.fillStyle = getHiddenColor();
            for (let i = 0; i < view.n_cells; i++) {{
                const val = values[i];
                if (val == null || !isFinite(val)) continue;
                const cat = cfg.categories?.[Math.round(val)];
                if (!cat || !hiddenCategories.has(cat)) continue;
                const {{x, y}} = tf.dataToScreen(view.x[i], view.y[i]);
                ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI*2); ctx.fill();
            }}
            ctx.globalAlpha = 1;
        }}

        // Pass 2 — visible cells
        const hasFocus = !cfg.is_continuous && spotlightCategory !== null;
        for (let i = 0; i < view.n_cells; i++) {{
            const val = values[i];
            if (val == null || !isFinite(val)) continue;

            let color, dimmed = false;
            if (cfg.is_continuous) {{
                const t = (val - cfg.vmin) / Math.max(cfg.vmax - cfg.vmin, 1e-10);
                color = magma(t);
            }} else {{
                const catIdx = Math.round(val);
                const cat    = cfg.categories?.[catIdx];
                if (!cat || hiddenCategories.has(cat)) continue;
                color  = PALETTE[catIdx % PALETTE.length];
                dimmed = hasFocus && cat !== spotlightCategory;
            }}

            const {{x, y}} = tf.dataToScreen(view.x[i], view.y[i]);
            ctx.fillStyle   = dimmed ? getDimColor() : color;
            ctx.globalAlpha = dimmed ? 0.15 : 1;
            ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI*2); ctx.fill();
        }}
        ctx.globalAlpha = 1;

        drawPAGA(ctx, view, tf, cfg, values);
        drawVelocity(ctx, view, tf);

        // Pass 3 — selection outlines
        if (selectedCells.size > 0) {{
            ctx.strokeStyle = getSelColor();
            ctx.lineWidth   = 1.3;
            for (let i = 0; i < view.n_cells; i++) {{
                const gi = toGlobal(view, i);
                if (!selectedCells.has(gi)) continue;
                const {{x, y}} = tf.dataToScreen(view.x[i], view.y[i]);
                ctx.beginPath(); ctx.arc(x, y, r + 0.8, 0, Math.PI*2); ctx.stroke();
            }}
        }}

        // Pass 4 — hovered cell
        if (hoveredCell !== null) {{
            for (let i = 0; i < view.n_cells; i++) {{
                if (toGlobal(view, i) !== hoveredCell) continue;
                const {{x, y}} = tf.dataToScreen(view.x[i], view.y[i]);
                ctx.save();
                ctx.strokeStyle = '#FF876F';
                ctx.lineWidth   = 2;
                ctx.beginPath();
                ctx.arc(x, y, r + 2.4, 0, Math.PI*2);
                ctx.stroke();
                ctx.restore();
                break;
            }}
        }}

        // Lasso overlay (modal only)
        if (isModal && lassoPoints.length > 1) {{
            ctx.save();
            ctx.strokeStyle = '#870052';
            ctx.lineWidth   = 1.5;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(lassoPoints[0].x, lassoPoints[0].y);
            for (let k = 1; k < lassoPoints.length; k++) ctx.lineTo(lassoPoints[k].x, lassoPoints[k].y);
            if (!isPointerDown) ctx.closePath();
            ctx.stroke();
            ctx.restore();
        }}
    }}

    function renderAllViews() {{
        renderAllJobId++;
        const jobId = renderAllJobId;
        const panels  = document.querySelectorAll('.view-panel');
        const grid    = document.getElementById('grid');
        const gridRect = grid?.getBoundingClientRect();

        let totalCells = 0;
        const drawList = [];
        DATA.views.forEach((view, idx) => {{
            totalCells += view.n_cells;
            const panel  = panels[idx];
            if (!panel) return;
            const canvas = panel.querySelector('canvas');
            if (!canvas) return;
            if (gridRect) {{
                const pr = panel.getBoundingClientRect();
                if (pr.bottom < gridRect.top - 300 || pr.top > gridRect.bottom + 300) return;
            }}
            drawList.push({{view, canvas}});
        }});

        const colorLabel = currentGene || currentColor;
        document.getElementById('stats-text').textContent =
            `${{DATA.n_views}} view${{DATA.n_views>1?'s':''}} · ${{DATA.n_cells.toLocaleString()}} cells · ${{colorLabel}}`;

        let i = 0;
        function step() {{
            if (jobId !== renderAllJobId) return;
            const t0 = performance.now();
            while (i < drawList.length && performance.now() - t0 < 12) {{
                const {{view, canvas}} = drawList[i++];
                try {{ renderView(view, canvas, false); }}
                catch(e) {{ console.error('renderView failed', e); }}
            }}
            if (i < drawList.length) requestAnimationFrame(step);
        }}
        requestAnimationFrame(step);
    }}

    function renderModal() {{
        const view = getView(modalViewId);
        if (!view) return;
        const canvas = document.getElementById('modal-canvas');
        if (!canvas) return;
        renderView(view, canvas, true);
    }}

    // ── Grid construction ──────────────────────────────────────────────────
    function buildGrid() {{
        const grid = document.getElementById('grid');
        grid.innerHTML = '';
        grid.classList.toggle('single-view-layout', DATA.n_views === 1);

        const embeddingCounts = new Map();
        DATA.views.forEach(view => {{
            if (!view.embedding_key) return;
            embeddingCounts.set(view.embedding_key, (embeddingCounts.get(view.embedding_key) || 0) + 1);
        }});

        let lastEmbeddingKey = null;
        DATA.views.forEach(view => {{
            if (view.embedding_key && view.embedding_key !== lastEmbeddingKey &&
                (embeddingCounts.get(view.embedding_key) || 0) > 1) {{
                const header = document.createElement('div');
                header.className = 'grid-group-header';
                header.textContent = view.name.includes(' — ') ? view.name.split(' — ')[0] : view.embedding_key;
                grid.appendChild(header);
            }}
            lastEmbeddingKey = view.embedding_key || null;

            const panel = document.createElement('div');
            panel.className = 'view-panel';
            panel.dataset.viewId = view.id;

            const lbl = document.createElement('div');
            lbl.className = 'panel-label';
            lbl.textContent = view.name;

            const canvas = document.createElement('canvas');
            canvas.className = 'view-canvas';

            const actions = document.createElement('div');
            actions.className = 'panel-actions';
            [
                ['hulls', 'Hu', 'Toggle convex hulls'],
                ['density', 'Dn', 'Toggle density contours'],
                ['paga', 'Pg', 'Toggle PAGA graph'],
                ['velocity', 'Ve', 'Toggle velocity arrows'],
            ].forEach(([name, label, title]) => {{
                const btn = document.createElement('button');
                btn.className = 'panel-toggle';
                btn.dataset.overlay = name;
                btn.title = title;
                btn.textContent = label;
                btn.addEventListener('click', e => {{
                    e.stopPropagation();
                    toggleOverlay(name);
                }});
                actions.appendChild(btn);
            }});

            panel.appendChild(lbl);
            panel.appendChild(actions);
            panel.appendChild(canvas);
            panel.addEventListener('click', () => openModal(view.id));
            grid.appendChild(panel);
        }});
        syncOverlayButtons();
    }}

    // ── Color / gene selectors ─────────────────────────────────────────────
    function buildColorSelector() {{
        const sel = document.getElementById('color-select');
        DATA.obs_columns.forEach(col => {{
            const opt = document.createElement('option');
            opt.value = col;
            opt.textContent = col;
            if (col === currentColor) opt.selected = true;
            sel.appendChild(opt);
        }});
        sel.addEventListener('change', () => setColor(sel.value));
    }}

    function buildGeneDatalist() {{
        const dl = document.getElementById('gene-datalist');
        DATA.available_genes.forEach(g => {{
            const opt = document.createElement('option');
            opt.value = g;
            dl.appendChild(opt);
        }});
    }}

    function setColor(key) {{
        currentColor = key;
        currentGene  = null;
        hiddenCategories.clear();
        spotlightCategory = null;
        document.getElementById('gene-input').value = '';
        document.getElementById('gene-clear-btn').classList.remove('visible');
        renderAllViews();
        if (modalViewId) renderModal();
        renderLegend();
        updateGeneDiscovery();
        updateInsights();
    }}

    function setGene(name) {{
        if (!name) {{ clearGene(); return; }}
        // Case-insensitive lookup
        const lower = name.toLowerCase();
        const match = DATA.available_genes.find(g => g.toLowerCase() === lower);
        if (!match) return;
        if (!DATA.genes || !DATA.genes[match]) return;
        currentGene = match;
        document.getElementById('gene-input').value = match;
        document.getElementById('gene-clear-btn').classList.add('visible');
        addRecentGene(match);
        renderAllViews();
        if (modalViewId) renderModal();
        renderLegend();
        updateGeneDiscovery();
        updateInsights();
    }}

    function clearGene() {{
        currentGene = null;
        document.getElementById('gene-input').value = '';
        document.getElementById('gene-clear-btn').classList.remove('visible');
        renderAllViews();
        if (modalViewId) renderModal();
        renderLegend();
        updateGeneDiscovery();
        updateInsights();
    }}

    function addRecentGene(name) {{
        recentGenes = [name, ...recentGenes.filter(g => g !== name)].slice(0, 12);
        try {{ localStorage.setItem('sckaro_recent_genes', JSON.stringify(recentGenes)); }} catch(_) {{}}
    }}

    function loadRecentGenes() {{
        try {{
            const stored = localStorage.getItem('sckaro_recent_genes');
            if (stored) recentGenes = JSON.parse(stored).slice(0, 12);
        }} catch(_) {{}}
    }}

    // ── Gene discovery panel ───────────────────────────────────────────────
    function updateGeneDiscovery() {{
        const panel = document.getElementById('gene-discovery-panel');
        panel.innerHTML = '';

        // Recent genes
        const validRecent = recentGenes.filter(g => DATA.genes && DATA.genes[g]);
        if (validRecent.length) {{
            const section = makeDiscoverySection('Recent genes', validRecent);
            panel.appendChild(section);
        }}

        // Embedded (HVG) genes — show first 20 when no query
        const inputVal = document.getElementById('gene-input').value.trim().toLowerCase();
        const embedded = Object.keys(DATA.genes || {{}});
        if (!inputVal && embedded.length) {{
            const shown = embedded.slice(0, 20);
            const section = makeDiscoverySection('Preloaded genes', shown);
            panel.appendChild(section);
        }}

        // Search results
        if (inputVal.length >= 1) {{
            const hits = DATA.available_genes
                .filter(g => g.toLowerCase().includes(inputVal))
                .slice(0, 30);
            if (hits.length) {{
                const section = makeDiscoverySection(
                    `Search results (${{hits.length}}${{hits.length===30?'+':''}})`, hits);
                panel.appendChild(section);
            }} else {{
                const empty = document.createElement('div');
                empty.className = 'discovery-empty';
                empty.textContent = `No genes matching "${{inputVal}}"`;
                panel.appendChild(empty);
            }}
        }}
    }}

    function makeDiscoverySection(title, genes) {{
        const sec = document.createElement('div');
        sec.className = 'discovery-section';

        const lbl = document.createElement('div');
        lbl.className = 'discovery-label';
        lbl.textContent = title;
        sec.appendChild(lbl);

        const grid = document.createElement('div');
        grid.className = 'gene-chip-grid';
        genes.forEach(gene => {{
            const chip = document.createElement('button');
            chip.className = 'gene-chip';
            if (currentGene === gene) chip.classList.add('active');
            chip.textContent = gene;
            chip.addEventListener('click', (e) => {{
                e.stopPropagation();
                setGene(gene);
                closeGeneDiscovery();
            }});
            grid.appendChild(chip);
        }});
        sec.appendChild(grid);
        return sec;
    }}

    function openGeneDiscovery() {{
        updateGeneDiscovery();
        document.getElementById('gene-discovery-panel').classList.add('open');
    }}
    function closeGeneDiscovery() {{
        document.getElementById('gene-discovery-panel').classList.remove('open');
    }}

    // ── Legend ─────────────────────────────────────────────────────────────
    function renderLegend() {{
        const legend = document.getElementById('legend');
        legend.innerHTML = '';
        const cfg = getColorConfig();
        if (!cfg) return;

        const title = document.createElement('div');
        title.className = 'legend-title';
        title.textContent = currentGene || currentColor;
        legend.appendChild(title);

        if (cfg.is_continuous) {{
            const wrap = document.createElement('div');
            wrap.className = 'colorbar-wrapper';

            const row = document.createElement('div');
            row.className = 'colorbar-row';

            const bc = document.createElement('canvas');
            bc.className = 'colorbar-canvas';
            bc.width = 14; bc.height = 140;
            const bctx = bc.getContext('2d');
            for (let row = 0; row < 140; row++) {{
                const [r,g,b] = magmaRgb(1 - row/139);
                bctx.fillStyle = `rgb(${{r}},${{g}},${{b}})`;
                bctx.fillRect(0, row, 14, 1);
            }}

            const lbls = document.createElement('div');
            lbls.className = 'colorbar-labels';
            const hi = document.createElement('div');
            hi.textContent = fmtNum(cfg.vmax);
            const lo = document.createElement('div');
            lo.textContent = fmtNum(cfg.vmin);
            lbls.appendChild(hi);
            lbls.appendChild(lo);

            row.appendChild(bc);
            row.appendChild(lbls);
            wrap.appendChild(row);
            legend.appendChild(wrap);
        }} else {{
            (cfg.categories || []).forEach((cat, idx) => {{
                const item = document.createElement('div');
                item.className = 'legend-item';
                if (hiddenCategories.has(cat)) item.classList.add('hidden-cat');
                if (spotlightCategory === cat) item.classList.add('spotlight');

                const dot = document.createElement('div');
                dot.className = 'legend-swatch';
                dot.style.background = PALETTE[idx % PALETTE.length];

                const lbl = document.createElement('span');
                lbl.className = 'legend-label';
                lbl.textContent = cat;
                lbl.title = cat;

                item.appendChild(dot);
                item.appendChild(lbl);
                item.addEventListener('click', (e) => {{
                    if (e.shiftKey) {{
                        spotlightCategory = (spotlightCategory === cat) ? null : cat;
                    }} else {{
                        if (hiddenCategories.has(cat)) hiddenCategories.delete(cat);
                        else hiddenCategories.add(cat);
                    }}
                    renderLegend();
                    renderAllViews();
                    if (modalViewId) renderModal();
                    updateInsights();
                }});
                legend.appendChild(item);
            }});

            const hint = document.createElement('div');
            hint.className = 'legend-hint';
            hint.textContent = 'Click to hide · Shift+click to spotlight';
            legend.appendChild(hint);

            if (hiddenCategories.size > 0 || spotlightCategory) {{
                const resetBtn = document.createElement('button');
                resetBtn.className = 'legend-reset-btn';
                resetBtn.textContent = 'Show all';
                resetBtn.addEventListener('click', () => {{
                    hiddenCategories.clear();
                    spotlightCategory = null;
                    renderLegend();
                    renderAllViews();
                    if (modalViewId) renderModal();
                    updateInsights();
                }});
                legend.appendChild(resetBtn);
            }}
        }}
    }}

    function fmtNum(v) {{
        if (!Number.isFinite(v)) return 'n/a';
        const abs = Math.abs(v);
        if (abs >= 1e4 || (abs > 0 && abs < 1e-2)) return v.toExponential(2);
        return parseFloat(v.toFixed(3)).toString();
    }}

    // ── Modal ──────────────────────────────────────────────────────────────
    function openModal(viewId) {{
        modalViewId = viewId;
        modalZoom   = 1;
        modalPanX   = 0;
        modalPanY   = 0;
        lassoPoints = [];
        const view  = getView(viewId);
        document.getElementById('modal-title').textContent = view?.name || viewId;
        document.getElementById('modal-overlay').classList.remove('hidden');
        setModalMode('pan');
        requestAnimationFrame(renderModal);
        updateInsights();
    }}

    function closeModal() {{
        document.getElementById('modal-overlay').classList.add('hidden');
        modalViewId = null;
        lassoPoints = [];
        updateHoveredCell(null);
    }}

    function setModalMode(mode) {{
        modalMode = mode;
        const canvas = document.getElementById('modal-canvas');
        canvas.style.cursor = (mode === 'lasso') ? 'crosshair' : 'grab';
        document.getElementById('btn-pan').classList.toggle('active',   mode === 'pan');
        document.getElementById('btn-lasso').classList.toggle('active', mode === 'lasso');
    }}

    function toggleOverlay(name) {{
        if (name === 'hulls') showHulls = !showHulls;
        if (name === 'density') showDensityContours = !showDensityContours;
        if (name === 'paga') showPaga = !showPaga;
        if (name === 'velocity') showVelocity = !showVelocity;
        syncOverlayButtons();
        renderAllViews();
        if (modalViewId) renderModal();
    }}

    function setupModalInteractions() {{
        const canvas = document.getElementById('modal-canvas');

        canvas.addEventListener('pointerdown', e => {{
            e.preventDefault();
            canvas.setPointerCapture(e.pointerId);
            isPointerDown = true;
            lastPX = e.offsetX; lastPY = e.offsetY;
            if (modalMode === 'lasso') {{
                lassoPoints = [{{x: e.offsetX, y: e.offsetY}}];
            }}
        }});

        canvas.addEventListener('pointermove', e => {{
            const view = getView(modalViewId);
            const rect = canvas.getBoundingClientRect();
            if (view && rect.width && rect.height) {{
                const tf = createViewTransform(view, {{
                    width: rect.width, height: rect.height,
                    padding: 20, zoom: modalZoom, panX: modalPanX, panY: modalPanY,
                }});
                const nearest = findNearestCell(
                    view, tf, e.offsetX, e.offsetY, Math.max(8, SPOT_STEPS[spotStepIdx] + 5));
                updateHoveredCell(nearest === null ? null : toGlobal(view, nearest));
            }}

            if (!isPointerDown) return;
            const dx = e.offsetX - lastPX;
            const dy = e.offsetY - lastPY;
            if (modalMode === 'pan') {{
                modalPanX += dx; modalPanY += dy;
                renderModal();
            }} else if (modalMode === 'lasso') {{
                lassoPoints.push({{x: e.offsetX, y: e.offsetY}});
                renderModal();
            }}
            lastPX = e.offsetX; lastPY = e.offsetY;
        }});

        canvas.addEventListener('pointerup', () => {{
            isPointerDown = false;
            if (modalMode === 'lasso' && lassoPoints.length > 3) {{
                applyLasso();
            }}
        }});

        canvas.addEventListener('pointerleave', () => {{
            isPointerDown = false;
            updateHoveredCell(null);
        }});

        canvas.addEventListener('wheel', e => {{
            e.preventDefault();
            const factor   = e.deltaY < 0 ? 1.15 : 1/1.15;
            const newZoom  = Math.max(0.05, Math.min(80, modalZoom * factor));
            const rect     = canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            const w = rect.width, h = rect.height;
            // Zoom toward mouse: keep data point under cursor fixed
            modalPanX = mx - w/2 - (mx - w/2 - modalPanX) * (newZoom / modalZoom);
            modalPanY = my - h/2 - (my - h/2 - modalPanY) * (newZoom / modalZoom);
            modalZoom = newZoom;
            renderModal();
        }}, {{passive: false}});
    }}

    // ── Lasso selection ────────────────────────────────────────────────────
    function pointInPolygon(px, py, polygon) {{
        let inside = false;
        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {{
            const xi = polygon[i].x, yi = polygon[i].y;
            const xj = polygon[j].x, yj = polygon[j].y;
            if (((yi > py) !== (yj > py)) && px < (xj-xi)*(py-yi)/(yj-yi) + xi) {{
                inside = !inside;
            }}
        }}
        return inside;
    }}

    function applyLasso() {{
        const view = getView(modalViewId);
        if (!view) return;
        const canvas = document.getElementById('modal-canvas');
        const rect   = canvas.getBoundingClientRect();
        const tf = createViewTransform(view, {{
            width: rect.width, height: rect.height,
            padding: 20, zoom: modalZoom, panX: modalPanX, panY: modalPanY,
        }});

        selectedCells.clear();
        for (let i = 0; i < view.n_cells; i++) {{
            const {{x, y}} = tf.dataToScreen(view.x[i], view.y[i]);
            if (pointInPolygon(x, y, lassoPoints)) {{
                selectedCells.add(toGlobal(view, i));
            }}
        }}

        lassoPoints = [];
        renderModal();
        renderAllViews();
        updateInsights();
    }}

    // ── Insights panel ─────────────────────────────────────────────────────
    function escapeHtml(text) {{
        return String(text ?? '')
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#39;');
    }}

    function getCategoricalConfig(groupby) {{
        const cfg = DATA.color_configs[groupby];
        if (!cfg || cfg.is_continuous || !cfg.categories) return null;
        return cfg;
    }}

    function getSelectionCounts(groupby) {{
        const cfg = getCategoricalConfig(groupby);
        if (!cfg) return null;
        const counts = {{}};
        cfg.categories.forEach(cat => counts[cat] = 0);
        for (const gi of selectedCells) {{
            const code = cfg.codes?.[gi];
            if (code == null || !Number.isFinite(code)) continue;
            const cat = cfg.categories[Math.round(code)];
            if (cat !== undefined) counts[cat] = (counts[cat] || 0) + 1;
        }}
        return counts;
    }}

    function getSingleSelectedCategory(groupby) {{
        const counts = getSelectionCounts(groupby);
        if (!counts) return null;
        const selectedCats = Object.entries(counts).filter(([, count]) => count > 0);
        return selectedCats.length === 1 ? selectedCats[0][0] : null;
    }}

    function getDefaultCompareGroupby() {{
        const keys = Object.keys(DATA.analytics?.cluster_de || {{}});
        if (!keys.length) return null;
        return keys.includes(currentColor) ? currentColor : keys[0];
    }}

    function setInsightsTab(tabName) {{
        activeInsightsTab = tabName;
        document.querySelectorAll('.insights-tab').forEach(btn => {{
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        }});
        document.querySelectorAll('.insights-pane').forEach(pane => {{
            pane.classList.toggle('hidden', pane.id !== `insights-${{tabName}}`);
        }});
    }}

    function renderStatsInsights() {{
        const panel = document.getElementById('insights-stats');
        if (!panel) return;

        if (selectedCells.size === 0) {{
            panel.innerHTML = `
                <div class="no-selection-msg">
                    <div>No cells selected</div>
                    <div class="hint">Switch to <b>lasso</b> mode and draw a selection</div>
                </div>`;
            return;
        }}

        const cfg = getCategoricalConfig(currentColor);
        let html = `<div class="stats-section">
            <div class="stats-section-title">Selection</div>
            <div class="stats-row">
                <span class="stats-key">Cells selected</span>
                <span class="stats-val">${{selectedCells.size.toLocaleString()}}</span>
            </div>
            <div class="stats-row">
                <span class="stats-key">Total cells</span>
                <span class="stats-val">${{DATA.n_cells.toLocaleString()}}</span>
            </div>
            <div class="stats-row">
                <span class="stats-key">Fraction</span>
                <span class="stats-val">${{(selectedCells.size / DATA.n_cells * 100).toFixed(1)}}%</span>
            </div>
        </div>`;

        if (cfg) {{
            const counts = getSelectionCounts(currentColor) || {{}};
            const sorted = Object.entries(counts)
                .filter(([, count]) => count > 0)
                .sort((a, b) => b[1] - a[1]);
            if (sorted.length) {{
                html += `<div class="stats-section">
                    <div class="stats-section-title">Composition · ${{escapeHtml(currentColor)}}</div>`;
                const total = sorted.reduce((sum, [, count]) => sum + count, 0);
                sorted.forEach(([cat, count]) => {{
                    const catIdx = cfg.categories.indexOf(cat);
                    const col = PALETTE[catIdx % PALETTE.length];
                    const pct = total > 0 ? (count / total * 100) : 0;
                    html += `<div class="stats-row">
                        <span class="stats-key">
                            <span class="stats-dot" style="background:${{col}}"></span>
                            ${{escapeHtml(cat)}}
                        </span>
                        <span class="stats-val">${{count}} (${{pct.toFixed(1)}}%)</span>
                    </div>
                    <div class="stats-bar-wrap">
                        <div class="stats-bar" style="width:${{pct}}%;background:${{col}}"></div>
                    </div>`;
                }});
                html += `</div>`;
            }}
        }}

        panel.innerHTML = html;
    }}

    function renderMarkersInsights() {{
        const panel = document.getElementById('insights-markers');
        if (!panel) return;

        const groupMarkers = DATA.analytics?.marker_genes?.[currentColor];
        if (!groupMarkers) {{
            panel.innerHTML = `<div class="insights-empty">No marker genes were precomputed for <b>${{escapeHtml(currentColor)}}</b>.</div>`;
            return;
        }}

        const activeCategory = spotlightCategory || getSingleSelectedCategory(currentColor);
        if (!activeCategory) {{
            panel.innerHTML = `<div class="insights-empty">Spotlight a legend category or lasso cells from a single <b>${{escapeHtml(currentColor)}}</b> category to view marker genes.</div>`;
            return;
        }}

        const entries = groupMarkers[activeCategory] || [];
        if (!entries.length) {{
            panel.innerHTML = `<div class="insights-empty">No marker genes available for <b>${{escapeHtml(activeCategory)}}</b>.</div>`;
            return;
        }}

        const maxScore = Math.max(...entries.map(entry => Math.abs(entry.score || 0)), 1e-9);
        let html = `<div class="stats-section">
            <div class="stats-section-title">Markers · ${{escapeHtml(currentColor)}}</div>
            <div class="stats-row">
                <span class="stats-key">${{escapeHtml(activeCategory)}}</span>
                <span class="stats-val">${{entries.length}} genes</span>
            </div>
        </div><div class="marker-list">`;
        entries.forEach(entry => {{
            const pct = Math.max(0, Math.min(100, Math.abs(entry.score || 0) / maxScore * 100));
            html += `<div class="marker-row">
                <div class="marker-head">
                    <span class="marker-gene">${{escapeHtml(entry.gene)}}</span>
                    <span class="marker-meta">score ${{fmtNum(entry.score)}} · pct ${{((entry.pct_expr || 0) * 100).toFixed(1)}}%</span>
                </div>
                <div class="marker-bar-track"><div class="marker-bar-fill" style="width:${{pct}}%"></div></div>
            </div>`;
        }});
        html += `</div>`;
        panel.innerHTML = html;
    }}

    function renderCompareInsights() {{
        const panel = document.getElementById('insights-compare');
        if (!panel) return;

        const clusterDe = DATA.analytics?.cluster_de || {{}};
        const groupbyKeys = Object.keys(clusterDe);
        if (!groupbyKeys.length) {{
            panel.innerHTML = `<div class="insights-empty">No pairwise differential expression was precomputed for this export.</div>`;
            return;
        }}

        compareGroupby = groupbyKeys.includes(compareGroupby) ? compareGroupby : getDefaultCompareGroupby();
        const payload = clusterDe[compareGroupby];
        const groups = payload?.groups || [];
        if (groups.length < 2) {{
            panel.innerHTML = `<div class="insights-empty">No valid comparison groups are available for <b>${{escapeHtml(compareGroupby)}}</b>.</div>`;
            return;
        }}

        if (!groups.includes(compareGroupA)) compareGroupA = groups[0];
        if (!groups.includes(compareGroupB) || compareGroupB === compareGroupA) {{
            compareGroupB = groups.find(group => group !== compareGroupA) || groups[0];
        }}

        const pairKey = `${{compareGroupA}}__vs__${{compareGroupB}}`;
        const entries = payload?.comparisons?.[pairKey] || [];
        const maxAbsLogfc = Math.max(...entries.map(entry => Math.abs(entry.logfc || 0)), 1e-9);

        panel.innerHTML = `
            <div class="insights-controls">
                <div class="insights-control">
                    <label for="compare-groupby-select">Grouping</label>
                    <select id="compare-groupby-select">
                        ${{groupbyKeys.map(key => `<option value="${{escapeHtml(key)}}"${{key === compareGroupby ? ' selected' : ''}}>${{escapeHtml(key)}}</option>`).join('')}}
                    </select>
                </div>
                <div class="insights-control">
                    <label for="compare-groupa-select">Group A</label>
                    <select id="compare-groupa-select">
                        ${{groups.map(group => `<option value="${{escapeHtml(group)}}"${{group === compareGroupA ? ' selected' : ''}}>${{escapeHtml(group)}}</option>`).join('')}}
                    </select>
                </div>
                <div class="insights-control">
                    <label for="compare-groupb-select">Group B</label>
                    <select id="compare-groupb-select">
                        ${{groups.map(group => `<option value="${{escapeHtml(group)}}"${{group === compareGroupB ? ' selected' : ''}}>${{escapeHtml(group)}}</option>`).join('')}}
                    </select>
                </div>
            </div>
            ${{
                entries.length
                    ? `<table class="de-table">
                        <thead>
                            <tr><th>Gene</th><th>logFC</th><th>p</th><th></th></tr>
                        </thead>
                        <tbody>
                            ${{
                                entries.map(entry => {{
                                    const width = Math.max(0, Math.min(100, Math.abs(entry.logfc || 0) / maxAbsLogfc * 100));
                                    return `<tr>
                                        <td>${{escapeHtml(entry.gene)}}</td>
                                        <td>${{fmtNum(entry.logfc)}}</td>
                                        <td>${{entry.pval == null ? 'n/a' : fmtNum(entry.pval)}}</td>
                                        <td><div class="de-bar-track"><div class="de-bar-fill" style="width:${{width}}%"></div></div></td>
                                    </tr>`;
                                }}).join('')
                            }}
                        </tbody>
                    </table>`
                    : `<div class="insights-empty">No DE results are available for <b>${{escapeHtml(compareGroupA)}}</b> vs <b>${{escapeHtml(compareGroupB)}}</b>.</div>`
            }}`;

        document.getElementById('compare-groupby-select')?.addEventListener('change', e => {{
            compareGroupby = e.target.value;
            compareGroupA = null;
            compareGroupB = null;
            renderCompareInsights();
        }});
        document.getElementById('compare-groupa-select')?.addEventListener('change', e => {{
            compareGroupA = e.target.value;
            if (compareGroupA === compareGroupB) compareGroupB = null;
            renderCompareInsights();
        }});
        document.getElementById('compare-groupb-select')?.addEventListener('change', e => {{
            compareGroupB = e.target.value;
            if (compareGroupA === compareGroupB) compareGroupA = null;
            renderCompareInsights();
        }});
    }}

    function drawDotplot(canvas, groupby, genes, categories) {{
        const cfg = getCategoricalConfig(groupby);
        if (!canvas || !cfg || !genes.length || !categories.length) return;

        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        if (!rect.width || !rect.height) return;
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;

        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.fillStyle = getPanelBg();
        ctx.fillRect(0, 0, rect.width, rect.height);

        const left = 78;
        const top = 46;
        const right = 14;
        const bottom = 18;
        const innerW = Math.max(40, rect.width - left - right);
        const innerH = Math.max(40, rect.height - top - bottom);
        const colW = innerW / Math.max(categories.length, 1);
        const rowH = innerH / Math.max(genes.length, 1);

        const categoryIndices = categories.map(cat => {{
            const catIdx = cfg.categories.indexOf(cat);
            const indices = [];
            for (let gi = 0; gi < DATA.n_cells; gi++) {{
                const code = cfg.codes?.[gi];
                if (code != null && Number.isFinite(code) && Math.round(code) === catIdx) {{
                    indices.push(gi);
                }}
            }}
            return indices;
        }});

        ctx.font = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = theme === 'dark' ? '#cfcfcf' : '#444444';

        genes.forEach((gene, rowIdx) => {{
            const y = top + rowH * (rowIdx + 0.5);
            ctx.fillText(gene, 8, y);

            const values = getGeneValues(gene);
            const meta = DATA.genes_meta?.[gene] || {{}};
            const vmax = Math.max(meta.vmax ?? 1, 1e-9);

            categories.forEach((cat, colIdx) => {{
                const indices = categoryIndices[colIdx];
                const x = left + colW * (colIdx + 0.5);
                if (!values || !indices.length) return;

                let sum = 0;
                let expressed = 0;
                indices.forEach(gi => {{
                    const value = values[gi] || 0;
                    sum += value;
                    if (value > 0) expressed++;
                }});
                const mean = indices.length ? sum / indices.length : 0;
                const frac = indices.length ? (expressed / indices.length) : 0;
                const radius = Math.max(1.5, Math.min(colW, rowH) * 0.38 * Math.sqrt(frac));
                ctx.fillStyle = magma(mean / vmax);
                ctx.beginPath();
                ctx.arc(x, y, radius, 0, Math.PI * 2);
                ctx.fill();
            }});
        }});

        ctx.save();
        ctx.translate(left, 18);
        categories.forEach((cat, colIdx) => {{
            const x = colW * (colIdx + 0.5);
            ctx.save();
            ctx.translate(x, 0);
            ctx.rotate(-Math.PI / 5);
            ctx.fillText(cat, 0, 0);
            ctx.restore();
        }});
        ctx.restore();
    }}

    function renderDotplotInsights() {{
        const panel = document.getElementById('insights-dotplot');
        if (!panel) return;

        const cfg = getCategoricalConfig(currentColor);
        if (!cfg) {{
            panel.innerHTML = `<div class="insights-empty">Dotplot currently requires a categorical obs column as the active color.</div>`;
            return;
        }}

        const markerGroup = DATA.analytics?.marker_genes?.[currentColor] || null;
        const activeCategory = spotlightCategory || getSingleSelectedCategory(currentColor);
        const defaultGenes = activeCategory && markerGroup?.[activeCategory]
            ? markerGroup[activeCategory].slice(0, 8).map(entry => entry.gene)
            : [];
        const requestedGenes = dotplotGeneText
            .split(',')
            .map(gene => gene.trim())
            .filter(Boolean);
        const genes = (requestedGenes.length ? requestedGenes : defaultGenes)
            .filter(gene => DATA.genes_meta?.[gene]);
        const categories = (spotlightCategory ? [spotlightCategory] : cfg.categories.filter(cat => !hiddenCategories.has(cat))).slice(0, 12);

        panel.innerHTML = `
            <div class="dotplot-shell">
                <div class="insights-control">
                    <label for="dotplot-gene-input">Genes</label>
                    <input type="text" id="dotplot-gene-input" placeholder="GeneA, GeneB, GeneC" value="${{escapeHtml(dotplotGeneText)}}" />
                </div>
                <canvas class="dotplot-canvas" id="dotplot-canvas"></canvas>
                <div class="dotplot-note">Rows use the active marker list when available. Comma-separated genes override it. Only embedded genes are drawn.</div>
            </div>`;

        document.getElementById('dotplot-gene-input')?.addEventListener('input', e => {{
            dotplotGeneText = e.target.value;
            renderDotplotInsights();
        }});

        if (!genes.length || !categories.length) {{
            const canvas = document.getElementById('dotplot-canvas');
            if (canvas) {{
                const ctx = canvas.getContext('2d');
                const rect = canvas.getBoundingClientRect();
                const dpr = window.devicePixelRatio || 1;
                canvas.width = rect.width * dpr;
                canvas.height = rect.height * dpr;
                ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
                ctx.fillStyle = getPanelBg();
                ctx.fillRect(0, 0, rect.width, rect.height);
                ctx.fillStyle = theme === 'dark' ? '#9a9a9a' : '#666666';
                ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
                ctx.fillText('No dotplot genes available for the current view.', 14, 22);
            }}
            return;
        }}

        drawDotplot(document.getElementById('dotplot-canvas'), currentColor, genes, categories);
    }}

    function updateInsights() {{
        renderStatsInsights();
        renderMarkersInsights();
        renderCompareInsights();
        renderDotplotInsights();
        setInsightsTab(activeInsightsTab);
    }}

    // ── Spot size ──────────────────────────────────────────────────────────
    function updateSpotSizeLabel() {{
        document.getElementById('size-label').textContent =
            SPOT_STEPS[spotStepIdx].toFixed(1) + 'px';
    }}
    function adjustSpotSize(delta) {{
        spotStepIdx = Math.max(0, Math.min(SPOT_STEPS.length-1, spotStepIdx + delta));
        updateSpotSizeLabel();
        renderAllViews();
        if (modalViewId) renderModal();
    }}

    // ── Screenshot ────────────────────────────────────────────────────────
    function takeScreenshot() {{
        const grid    = document.getElementById('grid');
        const canvases = Array.from(grid.querySelectorAll('canvas'));
        if (!canvases.length) return;
        const c = canvases[0];
        const a = document.createElement('a');
        a.download = 'sckaro_screenshot.png';
        a.href = c.toDataURL('image/png');
        a.click();
    }}

    // ── Keyboard shortcuts ─────────────────────────────────────────────────
    document.addEventListener('keydown', e => {{
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
        if (e.key === 'Escape') {{ closeModal(); closeGeneDiscovery(); }}
        if (e.key === 'T' || e.key === 't') {{ applyTheme(theme==='dark'?'light':'dark'); renderAllViews(); if(modalViewId) renderModal(); if(modalViewId) updateInsights(); }}
        if (e.key === 'S' || e.key === 's') takeScreenshot();
        if (modalViewId) {{
            if (e.key === 'P' || e.key === 'p') setModalMode('pan');
            if (e.key === 'L' || e.key === 'l') setModalMode('lasso');
            if (e.key === 'X' || e.key === 'x') {{ selectedCells.clear(); renderModal(); renderAllViews(); updateInsights(); }}
        }}
    }});

    // ── Wiring up all UI events ────────────────────────────────────────────
    function setupEvents() {{
        document.getElementById('theme-btn').addEventListener('click', () => {{
            applyTheme(theme === 'dark' ? 'light' : 'dark');
            renderAllViews();
            if (modalViewId) renderModal();
            renderLegend();
            updateInsights();
        }});

        document.getElementById('size-up').addEventListener('click',   () => adjustSpotSize(+1));
        document.getElementById('size-down').addEventListener('click',  () => adjustSpotSize(-1));
        document.getElementById('screenshot-btn').addEventListener('click', takeScreenshot);
        document.getElementById('modal-close').addEventListener('click', closeModal);
        document.getElementById('btn-pan').addEventListener('click',   () => setModalMode('pan'));
        document.getElementById('btn-lasso').addEventListener('click', () => setModalMode('lasso'));
        document.getElementById('btn-clear-sel').addEventListener('click', () => {{
            selectedCells.clear();
            lassoPoints = [];
            renderModal();
            renderAllViews();
            updateInsights();
        }});
        document.getElementById('btn-hulls').addEventListener('click', () => toggleOverlay('hulls'));
        document.getElementById('btn-density').addEventListener('click', () => toggleOverlay('density'));
        document.getElementById('btn-paga').addEventListener('click', () => toggleOverlay('paga'));
        document.getElementById('btn-velocity').addEventListener('click', () => toggleOverlay('velocity'));

        document.querySelectorAll('.insights-tab').forEach(btn => {{
            btn.addEventListener('click', () => {{
                setInsightsTab(btn.dataset.tab);
                updateInsights();
            }});
        }});

        // Close modal on backdrop click
        document.getElementById('modal-overlay').addEventListener('click', e => {{
            if (e.target === e.currentTarget) closeModal();
        }});

        // Gene input
        const geneInput = document.getElementById('gene-input');
        geneInput.addEventListener('focus', openGeneDiscovery);
        geneInput.addEventListener('input', () => {{
            updateGeneDiscovery();
            document.getElementById('gene-discovery-panel').classList.add('open');
            const val = geneInput.value.trim();
            document.getElementById('gene-clear-btn').classList.toggle('visible', val.length > 0);
        }});
        geneInput.addEventListener('keydown', e => {{
            if (e.key === 'Enter') {{
                setGene(geneInput.value.trim());
                closeGeneDiscovery();
                geneInput.blur();
            }}
            if (e.key === 'Escape') {{ closeGeneDiscovery(); geneInput.blur(); }}
        }});
        geneInput.addEventListener('change', () => {{
            const val = geneInput.value.trim();
            if (val) setGene(val);
        }});
        document.getElementById('gene-clear-btn').addEventListener('click', clearGene);

        document.addEventListener('click', e => {{
            const panel = document.getElementById('gene-discovery-panel');
            const shell = document.querySelector('.gene-input-shell');
            if (shell && !shell.contains(e.target)) closeGeneDiscovery();
        }});

        // Scroll-to-render for grid
        document.getElementById('grid').addEventListener('scroll', () => {{
            requestAnimationFrame(renderAllViews);
        }});

        window.addEventListener('resize', () => {{
            renderAllViews();
            if (modalViewId) renderModal();
            if (modalViewId) updateInsights();
        }});

        setupModalInteractions();
    }}

    // ── Initialise ─────────────────────────────────────────────────────────
    function init() {{
        applyTheme(DATA.theme || 'light');
        loadRecentGenes();
        buildGrid();
        buildColorSelector();
        buildGeneDatalist();
        updateSpotSizeLabel();
        renderLegend();
        syncOverlayButtons();
        requestAnimationFrame(renderAllViews);
        setupEvents();

        // Hide loader once first frame renders
        requestAnimationFrame(() => {{
            requestAnimationFrame(() => {{
                const loader = document.getElementById('loading-overlay');
                if (loader) loader.style.display = 'none';
            }});
        }});
    }}

    document.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>
'''


# ── Main export function ─────────────────────────────────────────────────────

def export_to_html(
    dataset: ScDataset,
    output_path: Union[str, Path],
    color: str = "leiden",
    title: str = "scKaroSpace",
    theme: str = "light",
    spot_size: float = 3.0,
    genes: Optional[List[str]] = None,
    hvg_limit: int = 20,
    gene_sparse_threshold: float = 0.8,
    marker_genes_groupby: Optional[List[str]] = None,
    cluster_de_groupby: Optional[List[str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> str:
    """
    Export a ScDataset to a standalone interactive HTML file.

    Parameters
    ----------
    dataset
        Dataset returned by ``load_sc_data()``.
    output_path
        Path for the output HTML file.
    color
        Initial obs column or gene name to colour by.
    title
        Page title shown in the browser tab and header.
    theme
        ``'light'`` (default) or ``'dark'``.
    spot_size
        Default cell dot radius in screen pixels.
    genes
        Explicit list of genes to embed for expression colouring.
        Augmented by the top ``hvg_limit`` highly variable genes.
    hvg_limit
        Number of highly variable genes to auto-include.
    gene_sparse_threshold
        Zero-fraction threshold above which sparse encoding is used (0–1).
    vmin, vmax
        Optional fixed min/max for continuous colour scales.

    Returns
    -------
    str
        Path to the written HTML file.
    """
    adata = dataset.adata

    # ── Resolve initial color ────────────────────────────────────────────
    if color not in adata.obs.columns and color not in adata.var_names:
        # Fall back to first available obs column
        if dataset.obs_columns:
            print(f"  Warning: color '{color}' not found; falling back to '{dataset.obs_columns[0]}'")
            color = dataset.obs_columns[0]
        else:
            raise ValueError(f"color '{color}' not found and no obs columns available")

    # ── Build obs color configs (categorical + continuous) ────────────────
    print("  Building color configs…")
    color_configs: Dict[str, dict] = {}
    for col in dataset.obs_columns:
        try:
            vals, is_cont, cats = dataset.get_color_data(col)
        except Exception as e:
            print(f"    Skipping obs column '{col}': {e}")
            continue

        if is_cont:
            finite = np.isfinite(vals)
            color_configs[col] = {
                "is_continuous": True,
                "values": [round(float(v), 5) if np.isfinite(v) else None for v in vals],
                "vmin": float(np.nanmin(vals[finite])) if finite.any() else 0.0,
                "vmax": float(np.nanmax(vals[finite])) if finite.any() else 1.0,
            }
        else:
            color_configs[col] = {
                "is_continuous": False,
                "categories": cats,
                "codes": [None if np.isnan(v) else int(round(v)) for v in vals],
            }

    # ── Optional analytics payload ──────────────────────────────────────
    analytics = {
        "marker_genes": {},
        "cluster_de": {},
    }
    paga_payload = _extract_paga(adata)
    velocity_payload = _extract_velocity_embeddings(adata)

    for groupby in marker_genes_groupby or []:
        if groupby not in adata.obs.columns:
            print(f"  Warning: marker genes skipped; obs column '{groupby}' not found")
            continue
        print(f"  Computing marker genes for '{groupby}'…")
        result = _compute_marker_genes(adata, groupby=groupby)
        if result:
            analytics["marker_genes"][groupby] = result

    for groupby in cluster_de_groupby or []:
        if groupby not in adata.obs.columns:
            print(f"  Warning: cluster DE skipped; obs column '{groupby}' not found")
            continue
        print(f"  Computing cluster DE for '{groupby}'…")
        result = _compute_cluster_de(adata, groupby=groupby)
        if result:
            analytics["cluster_de"][groupby] = result

    # ── Determine genes to embed ─────────────────────────────────────────
    explicit_genes = list(genes or [])
    hvgs = _select_hvgs(adata, n=hvg_limit) if hvg_limit > 0 else []

    # If the initial color is a gene, include it
    initial_gene = color if color in adata.var_names else None
    if initial_gene:
        explicit_genes = [initial_gene] + [g for g in explicit_genes if g != initial_gene]

    embed_genes = list(dict.fromkeys(  # deduplicate, preserve order
        g for g in explicit_genes + hvgs + _collect_analytics_genes(analytics)
        if g in adata.var_names
    ))
    print(f"  Embedding {len(embed_genes)} genes…")

    gene_data = dataset._collect_gene_data(embed_genes)
    genes_payload: Dict[str, dict] = {}
    genes_meta: Dict[str, dict] = {}
    for gene, gd in gene_data.items():
        genes_payload[gene] = _encode_gene(gd["values"], sparse_threshold=gene_sparse_threshold)
        genes_meta[gene] = {"vmin": round(gd["vmin"], 5), "vmax": round(gd["vmax"], 5)}

    # ── Build view payloads ──────────────────────────────────────────────
    view_payloads = []
    for view in dataset.views:
        xmin, xmax, ymin, ymax = view.bounds
        view_payloads.append({
            "id":           view.id,
            "name":         view.name,
            "embedding_key": view.embedding_key,
            "split_value":  view.split_value,
            "x":            [round(float(v), 4) for v in view.x],
            "y":            [round(float(v), 4) for v in view.y],
            "n_cells":      view.n_cells,
            "xmin":         round(float(xmin), 4),
            "xmax":         round(float(xmax), 4),
            "ymin":         round(float(ymin), 4),
            "ymax":         round(float(ymax), 4),
            "cell_indices": view.cell_indices.tolist() if view.cell_indices is not None else None,
        })

    # ── Assemble data payload ────────────────────────────────────────────
    payload = {
        "title":          title,
        "theme":          theme,
        "n_cells":        dataset.n_cells,
        "n_views":        dataset.n_views,
        "color":          color if color in color_configs else (dataset.obs_columns[0] if dataset.obs_columns else ""),
        "spot_size":      float(spot_size),
        "obs_columns":    dataset.obs_columns,
        "available_genes": dataset.var_names,
        "views":          view_payloads,
        "color_configs":  color_configs,
        "genes_meta":     genes_meta,
        "genes":          genes_payload,
        "analytics":      analytics,
        "paga":           paga_payload,
        "velocity_embedding": velocity_payload,
    }

    # ── Render HTML ──────────────────────────────────────────────────────
    data_json = json.dumps(payload, separators=(",", ":"))
    palette_json = json.dumps(DEFAULT_PALETTE)

    html = HTML_TEMPLATE.format(
        title=title,
        data_json=data_json,
        palette_json=palette_json,
    )

    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    size_mb = output_path.stat().st_size / 1e6
    print(f"  Written to {output_path} ({size_mb:.1f} MB)")
    return str(output_path)
