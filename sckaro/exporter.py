"""
Export single-cell data to a standalone HTML viewer.

Produces a single self-contained HTML file with all data embedded as JSON
and a vanilla-JS Canvas-based viewer — no server or Python required.
"""

import json
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
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

GENE_SIDECAR_SHARD_SIZE = 256
KAROSPACE_PACKAGE_MANIFEST = "karospace-package.json"
KAROSPACE_PACKAGE_LOADER_FILENAME = "karospace-package-loader.html"


def _chunked(values: List[str], size: int) -> List[List[str]]:
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    return [values[i:i + size] for i in range(0, len(values), size)]


def _isoformat_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _guess_package_media_type(path: Union[str, Path]) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".html":
        return "text/html"
    if suffix == ".json":
        return "application/json"
    return "application/octet-stream"


def _find_karospace_package_loader_template() -> Optional[Path]:
    candidates = [
        Path(__file__).resolve().with_name("package_loader.html"),
        Path(__file__).resolve().parent.parent / KAROSPACE_PACKAGE_LOADER_FILENAME,
        Path(__file__).resolve().parents[2] / "KaroSpace" / KAROSPACE_PACKAGE_LOADER_FILENAME,
        Path("/Users/chrislangseth/work/karolinska_institutet/projects/KaroSpace") / KAROSPACE_PACKAGE_LOADER_FILENAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _write_karospace_package_loader(
    *,
    loader_path: Path,
) -> Optional[Path]:
    template_path = _find_karospace_package_loader_template()
    if template_path is None:
        return None
    loader_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template_path, loader_path)
    return loader_path

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


def _build_gene_sidecar_shard(
    dataset: ScDataset,
    genes: List[str],
    *,
    gene_sparse_threshold: float,
) -> Dict[str, Any]:
    gene_data = dataset._collect_gene_data(genes)
    genes_payload: Dict[str, dict] = {}
    genes_meta: Dict[str, dict] = {}
    gene_encodings: Dict[str, str] = {}
    for gene, gd in gene_data.items():
        encoded = _encode_gene(gd["values"], sparse_threshold=gene_sparse_threshold)
        genes_payload[gene] = encoded
        genes_meta[gene] = {
            "vmin": round(gd["vmin"], 5),
            "vmax": round(gd["vmax"], 5),
        }
        gene_encodings[gene] = "sparse" if "sparse" in encoded else "dense"
    return {
        "format": "karospace-gene-sidecar-shard-v2",
        "genes": genes_payload,
        "genes_meta": genes_meta,
        "gene_encodings": gene_encodings,
    }


def _build_karospace_package_manifest(
    *,
    source_root: Path,
    entry_html: str,
    gene_manifest_path: str,
    gene_shard_dir: str,
    title: str,
    n_views: int,
    total_cells: int,
) -> dict:
    files = {}
    for file_path in sorted(p for p in source_root.rglob("*") if p.is_file()):
        rel_path = file_path.relative_to(source_root).as_posix()
        files[rel_path] = {
            "media_type": _guess_package_media_type(rel_path),
            "size_bytes": int(file_path.stat().st_size),
        }

    return {
        "format": "karospace-package-v1",
        "package_version": 1,
        "entry_html": entry_html,
        "created_at": _isoformat_utc_now(),
        "producer": {
            "name": "sckaro",
            "version": "0.1.0",
        },
        "title": title,
        "n_sections": int(n_views),
        "total_cells": int(total_cells),
        "viewer": {
            "mode": "sidecar-package",
            "gene_storage": "sidecar",
            "gene_manifest_path": gene_manifest_path,
            "gene_shard_dir": gene_shard_dir,
        },
        "files": files,
    }


def _write_karospace_package(
    *,
    package_path: Path,
    source_root: Path,
    entry_html: str,
    gene_manifest_path: str,
    gene_shard_dir: str,
    title: str,
    n_views: int,
    total_cells: int,
) -> None:
    package_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _build_karospace_package_manifest(
        source_root=source_root,
        entry_html=entry_html,
        gene_manifest_path=gene_manifest_path,
        gene_shard_dir=gene_shard_dir,
        title=title,
        n_views=n_views,
        total_cells=total_cells,
    )
    with zipfile.ZipFile(package_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            KAROSPACE_PACKAGE_MANIFEST,
            json.dumps(manifest, separators=(",", ":")),
            compress_type=zipfile.ZIP_DEFLATED,
        )
        for file_path in sorted((p for p in source_root.rglob("*") if p.is_file())):
            rel_path = file_path.relative_to(source_root).as_posix()
            zf.write(file_path, arcname=rel_path, compress_type=zipfile.ZIP_DEFLATED)


def _extract_embedded_viewer_data(html_text: str) -> dict:
    match = re.search(
        r'<script id="sckaro-data" type="application/json">(.*?)</script>',
        html_text,
        re.DOTALL,
    )
    if not match:
        raise ValueError("embedded sckaro-data script not found in HTML")
    return json.loads(match.group(1).replace("<\\/", "</"))


def _extract_html_title(html_text: str) -> str:
    match = re.search(r"<title>(.*?)</title>", html_text, re.DOTALL | re.IGNORECASE)
    if not match:
        return "scKaroSpace"
    return str(match.group(1)).strip() or "scKaroSpace"


def package_sidecar_viewer(
    html_path: Union[str, Path],
    *,
    output_path: Optional[Union[str, Path]] = None,
    gene_manifest_path: Optional[Union[str, Path]] = None,
    gene_shard_dir: Optional[Union[str, Path]] = None,
    loader_output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Package an existing sidecar viewer bundle into a `.karospace` archive."""
    source_html_path = Path(html_path).expanduser().resolve()
    if not source_html_path.exists():
        raise FileNotFoundError(f"sidecar HTML not found: {source_html_path}")

    html_text = source_html_path.read_text(encoding="utf-8")
    data = _extract_embedded_viewer_data(html_text)
    package_title = _extract_html_title(html_text)

    gene_aux_url = str(data.get("gene_aux_url") or "").strip()
    if not gene_aux_url:
        raise ValueError("HTML viewer does not reference a sidecar gene manifest")

    html_parent = source_html_path.parent
    package_gene_manifest_rel = Path(gene_aux_url).as_posix()
    actual_gene_manifest_path = (
        Path(gene_manifest_path).expanduser().resolve()
        if gene_manifest_path is not None
        else (html_parent / package_gene_manifest_rel).resolve()
    )
    if not actual_gene_manifest_path.exists():
        raise FileNotFoundError(f"sidecar gene manifest not found: {actual_gene_manifest_path}")

    package_gene_shard_rel = Path(package_gene_manifest_rel).with_suffix("").as_posix()
    actual_gene_shard_dir = (
        Path(gene_shard_dir).expanduser().resolve()
        if gene_shard_dir is not None
        else (html_parent / package_gene_shard_rel).resolve()
    )
    if not actual_gene_shard_dir.exists():
        raise FileNotFoundError(f"sidecar shard directory not found: {actual_gene_shard_dir}")

    resolved_output_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else source_html_path.with_suffix(".karospace")
    )
    if resolved_output_path.suffix.lower() != ".karospace":
        raise ValueError("output_path must end with .karospace")

    with tempfile.TemporaryDirectory(prefix="sckaro-package-") as tmpdir:
        source_root = Path(tmpdir)
        entry_html = "index.html"
        (source_root / entry_html).write_text(html_text, encoding="utf-8")
        staged_manifest = source_root / package_gene_manifest_rel
        staged_manifest.parent.mkdir(parents=True, exist_ok=True)
        staged_manifest.write_text(actual_gene_manifest_path.read_text(encoding="utf-8"), encoding="utf-8")

        staged_shard_dir = source_root / package_gene_shard_rel
        staged_shard_dir.mkdir(parents=True, exist_ok=True)
        for file_path in actual_gene_shard_dir.rglob("*"):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(actual_gene_shard_dir)
            target = staged_shard_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(file_path.read_bytes())

        _write_karospace_package(
            package_path=resolved_output_path,
            source_root=source_root,
            entry_html=entry_html,
            gene_manifest_path=package_gene_manifest_rel,
            gene_shard_dir=package_gene_shard_rel,
            title=package_title,
            n_views=int(data.get("n_views") or 0),
            total_cells=int(data.get("n_cells") or 0),
        )

    resolved_loader_output_path = (
        Path(loader_output_path).expanduser().resolve()
        if loader_output_path is not None
        else resolved_output_path.with_suffix(".loader.html")
    )
    _write_karospace_package_loader(loader_path=resolved_loader_output_path)
    return str(resolved_output_path)


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
                radial-gradient(800px 500px at 10% 0%, rgba(255,135,111,0.08), transparent),
                radial-gradient(900px 600px at 100% 20%, rgba(135,0,82,0.08), transparent),
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
                linear-gradient(90deg, rgba(255,135,111,0.12), rgba(135,0,82,0.08)),
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
        .header h1 {{ font-size: 16px; font-weight: 600; letter-spacing: -0.01em; }}
        .header h1 span {{ color: var(--accent); }}
        .stats {{ font-size: 11px; color: var(--muted); font-variant-numeric: tabular-nums; }}
        .controls {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
        .control-group {{ display: flex; align-items: center; gap: 4px; }}
        .control-group label {{ font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }}
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
            border-color: var(--accent-strong);
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
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
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

        /* ── Analysis sidebar ── */
        .analysis-sidebar {{
            width: 280px; flex-shrink: 0;
            border-left: 1px solid var(--border);
            background: var(--panel-bg);
            display: flex; flex-direction: column;
            overflow: hidden;
            transition: border-color 0.3s, background 0.3s;
        }}
        .sidebar-analytics {{
            flex: 1; min-height: 0;
            display: flex; flex-direction: column;
            overflow: hidden;
            border-top: 1px solid var(--border);
        }}

        /* ── Legend panel ── */
        .legend-panel {{
            flex-shrink: 0;
            background: var(--panel-bg);
            overflow-y: auto;
            padding: 12px 10px;
            max-height: 220px;
            transition: background 0.3s;
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
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
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
            padding: 4px 10px;
            border: 1px solid var(--border); border-radius: 999px;
            background: var(--input-bg); color: var(--text);
            cursor: pointer; font-size: 11px; font-weight: 600;
            transition: background 0.2s, border-color 0.2s, color 0.2s;
        }}
        .toolbar-btn:hover {{ background: var(--hover-bg); }}
        .toolbar-btn.active {{
            background: var(--accent-strong);
            border-color: var(--accent-strong); color: #fff;
        }}
        .modal-close {{
            width: 28px; height: 28px;
            border: 1px solid var(--border); border-radius: 999px;
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

        /* ── Insights sidebar (shared by sidebar and modal stats) ── */
        .insights-tabs {{
            display: grid; grid-template-columns: repeat(5, 1fr);
            gap: 3px; padding: 6px;
            flex-shrink: 0; border-bottom: 1px solid var(--border);
        }}
        .insights-tab {{
            padding: 5px 4px;
            border: 1px solid var(--border); border-radius: 999px;
            background: var(--input-bg); color: var(--muted);
            cursor: pointer; font-size: 9px; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.03em;
            white-space: nowrap; text-align: center;
            transition: background 0.2s, border-color 0.2s, color 0.2s;
        }}
        .insights-tab.active {{
            background: var(--accent-strong);
            border-color: var(--accent-strong); color: #fff;
        }}
        .insights-tab:hover:not(.active) {{ background: var(--hover-bg); color: var(--text); }}
        .insights-content {{
            flex: 1; overflow-y: auto; padding: 12px;
        }}
        /* Modal stats panel (selection info only) */
        .modal-stats-panel {{
            width: 190px; flex-shrink: 0;
            border-left: 1px solid var(--border);
            display: flex; flex-direction: column;
            overflow: hidden;
        }}
        .modal-stats-panel .insights-content {{
            flex: 1; overflow-y: auto; padding: 10px;
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
        .insights-search {{
            width: 100%;
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
        .gene-link-btn {{
            background: none; border: none; padding: 0;
            color: var(--accent); font-size: 12px; font-weight: 600;
            cursor: pointer; text-align: left;
            transition: opacity 0.15s;
        }}
        .gene-link-btn:hover {{ opacity: 0.7; text-decoration: underline; }}
        .cluster-chip-row {{
            display: flex; flex-wrap: wrap; gap: 4px; margin-top: 2px;
        }}
        .cluster-chip {{
            padding: 2px 9px; border-radius: 999px; cursor: pointer;
            font-size: 11px; font-weight: 500;
            border: 1.5px solid var(--chip-color, var(--border));
            background: none; color: var(--text);
            transition: background 0.15s, color 0.15s;
        }}
        .cluster-chip:hover {{ opacity: 0.75; }}
        .cluster-chip.active {{
            background: var(--chip-color, var(--accent));
            color: #fff;
        }}
        .marker-dotplot-wrap {{
            margin-top: 10px;
            padding-top: 8px;
            border-top: 1px solid rgba(127,127,127,0.12);
        }}
        .marker-dotplot-label {{
            font-size: 10px; font-weight: 600; text-transform: uppercase;
            letter-spacing: .04em; color: var(--muted); margin-bottom: 6px;
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
        .insights-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
        }}
        .insights-table th,
        .insights-table td {{
            padding: 6px 0;
            text-align: left;
            border-bottom: 1px solid rgba(127,127,127,0.12);
            vertical-align: top;
        }}
        .insights-table td:last-child,
        .insights-table th:last-child {{
            text-align: right;
        }}
        .table-subtle {{
            color: var(--muted);
            font-size: 10px;
        }}
        .dotplot-shell {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .dotplot-canvas {{
            width: 100%;
            height: 240px;
            display: block;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--panel-bg);
        }}
        .dotplot-note {{
            font-size: 10px;
            color: var(--muted);
            line-height: 1.4;
        }}
        .gene-chip-input {{
            display: flex; flex-wrap: wrap; gap: 4px; align-items: center;
            border: 1px solid var(--border); border-radius: 6px;
            padding: 4px 6px; background: var(--input-bg);
            min-height: 32px; cursor: text;
            transition: border-color 0.2s;
        }}
        .gene-chip-input:focus-within {{ border-color: var(--accent-strong); }}
        .gene-chip {{
            display: inline-flex; align-items: center; gap: 3px;
            background: var(--accent-strong); color: #fff;
            border: none; border-radius: 999px;
            padding: 2px 6px 2px 9px; font-size: 11px; font-weight: 600;
            cursor: default; white-space: nowrap;
        }}
        .chip-remove {{
            display: inline-flex; align-items: center; justify-content: center;
            width: 14px; height: 14px; border-radius: 50%;
            font-size: 13px; line-height: 1; opacity: 0.7;
            cursor: pointer; background: rgba(255,255,255,0.2);
            border: none; color: inherit; padding: 0; flex-shrink: 0;
        }}
        .chip-remove:hover {{ opacity: 1; background: rgba(255,255,255,0.35); }}
        .gene-chip-input input {{
            border: none; background: none; outline: none; box-shadow: none;
            font-size: 12px; color: var(--text); min-width: 70px; flex: 1; padding: 0;
        }}
        .boxplot-canvas {{
            width: 100%;
            height: 220px;
            display: block;
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
            .analysis-sidebar {{ display: none; }}
            .modal-stats-panel {{ width: 150px; }}
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
            <div class="control-group" style="position:relative;">
                <label for="gene2-input">Gene B</label>
                <div class="gene-input-shell">
                    <input type="text" id="gene2-input" placeholder="compare gene…"
                           autocomplete="off" spellcheck="false" list="gene-datalist" />
                    <button class="gene-clear-btn" id="gene2-clear-btn" title="Clear compare gene">✕</button>
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
        <div class="analysis-sidebar">
            <div id="legend" class="legend-panel"></div>
            <div class="sidebar-analytics">
                <div class="insights-tabs" id="sidebar-tabs">
                    <button class="insights-tab active" data-tab="markers">Markers</button>
                    <button class="insights-tab" data-tab="boxplot">Boxplot</button>
                    <button class="insights-tab" data-tab="table">Table</button>
                    <button class="insights-tab" data-tab="compare">Compare</button>
                    <button class="insights-tab" data-tab="dotplot">Dotplot</button>
                </div>
                <div class="insights-content">
                    <div class="insights-pane" id="insights-markers"></div>
                    <div class="insights-pane hidden" id="insights-boxplot"></div>
                    <div class="insights-pane hidden" id="insights-table"></div>
                    <div class="insights-pane hidden" id="insights-compare"></div>
                    <div class="insights-pane hidden" id="insights-dotplot"></div>
                </div>
            </div>
        </div>
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
                    </div>
                </div>
                <button class="modal-close" id="modal-close" title="Close (Esc)">✕</button>
            </div>
            <div class="modal-body">
                <div class="modal-canvas-wrapper">
                    <canvas id="modal-canvas"></canvas>
                    <div class="modal-zoom-hint" id="zoom-hint">Scroll to zoom · drag to pan · P/L to switch mode</div>
                </div>
                <div class="modal-stats-panel">
                    <div class="insights-content">
                        <div id="insights-stats">
                            <div class="no-selection-msg">
                                <div>No cells selected</div>
                                <div class="hint">Switch to <b>lasso</b> mode and draw a selection</div>
                            </div>
                        </div>
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
    let currentGene  = null;   // primary gene when coloring by gene expression
    let compareGene  = null;   // optional second gene for side-by-side compare mode
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
    const AVAILABLE_GENE_SET = new Set(DATA.available_genes || []);
    let geneAuxManifest = null;
    let geneAuxManifestPromise = null;
    const geneAuxShardCache = new Map();
    const geneAuxShardPromises = new Map();

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
    let activeInsightsTab = 'markers';
    let compareGroupby = null;
    let compareGroupA = null;
    let compareGroupB = null;
    let dotplotGeneText = '';
    let tableGeneText = '';
    let markerInsightsGroupby = null;
    let markerInsightsGroup = null;

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

    function getColorConfig(geneName=null, colorKey=null) {{
        if (geneName) {{
            const m = DATA.genes_meta[geneName] || {{}};
            return {{ is_continuous: true, vmin: m.vmin ?? 0, vmax: m.vmax ?? 1 }};
        }}
        return DATA.color_configs[colorKey || currentColor] || null;
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

    function hydrateGeneFromAux(gene, shardData) {{
        const geneEntry = shardData?.genes?.[gene];
        if (!geneEntry) return false;
        DATA.genes = DATA.genes || {{}};
        DATA.genes_meta = DATA.genes_meta || {{}};
        DATA.genes[gene] = geneEntry;
        if (shardData?.genes_meta?.[gene]) {{
            DATA.genes_meta[gene] = shardData.genes_meta[gene];
        }}
        geneCache.delete(gene);
        return true;
    }}

    async function loadGeneAuxManifest() {{
        if (geneAuxManifest) return geneAuxManifest;
        if (geneAuxManifestPromise) return geneAuxManifestPromise;
        if (!DATA.gene_aux_url) return null;
        if (window.location.protocol === 'file:' && !window.__karospacePackageMode) {{
            alert('This viewer was exported with sidecar gene loading. Open it over HTTP(S) to load additional genes.');
            return null;
        }}
        geneAuxManifestPromise = fetch(DATA.gene_aux_url, {{ credentials: 'same-origin' }})
            .then((response) => {{
                if (!response.ok) {{
                    throw new Error(`HTTP ${{response.status}} while loading gene sidecar manifest`);
                }}
                return response.json();
            }})
            .then((payload) => {{
                if (!payload || payload.format !== 'karospace-gene-sidecar-manifest-v2') {{
                    throw new Error('Unsupported gene sidecar manifest format');
                }}
                geneAuxManifest = payload;
                return payload;
            }})
            .catch((error) => {{
                console.error('Failed to load gene sidecar manifest:', error);
                geneAuxManifestPromise = null;
                alert(`Failed to load auxiliary gene manifest: ${{error?.message || 'Unknown error'}}`);
                return null;
            }});
        return geneAuxManifestPromise;
    }}

    async function loadGeneAuxShard(shardUrl) {{
        if (!shardUrl) return null;
        if (geneAuxShardCache.has(shardUrl)) return geneAuxShardCache.get(shardUrl);
        if (geneAuxShardPromises.has(shardUrl)) return geneAuxShardPromises.get(shardUrl);
        const promise = fetch(shardUrl, {{ credentials: 'same-origin' }})
            .then((response) => {{
                if (!response.ok) {{
                    throw new Error(`HTTP ${{response.status}} while loading gene shard`);
                }}
                return response.json();
            }})
            .then((payload) => {{
                if (!payload || payload.format !== 'karospace-gene-sidecar-shard-v2') {{
                    throw new Error('Unsupported gene sidecar shard format');
                }}
                geneAuxShardCache.set(shardUrl, payload);
                return payload;
            }})
            .catch((error) => {{
                console.error('Failed to load gene shard:', error);
                geneAuxShardPromises.delete(shardUrl);
                alert(`Failed to load requested gene data: ${{error?.message || 'Unknown error'}}`);
                return null;
            }});
        geneAuxShardPromises.set(shardUrl, promise);
        return promise;
    }}

    async function ensureGeneLoaded(gene, options = {{}}) {{
        const token = String(gene || '').trim();
        const showErrors = options.showErrors !== false;
        if (!token) return false;
        if (DATA.genes && DATA.genes[token]) return true;
        if (!AVAILABLE_GENE_SET.has(token)) {{
            if (showErrors) alert(`Gene "${{token}}" was not found in this dataset.`);
            return false;
        }}
        const manifest = await loadGeneAuxManifest();
        if (!manifest) return false;
        const shardUrl = manifest?.gene_to_shard?.[token];
        if (!shardUrl) {{
            if (showErrors) alert(`Gene "${{token}}" is listed in the dataset but missing from the sidecar manifest.`);
            return false;
        }}
        const shardData = await loadGeneAuxShard(shardUrl);
        if (!shardData) return false;
        const hydrated = hydrateGeneFromAux(token, shardData);
        if (!hydrated && showErrors) {{
            alert(`Gene "${{token}}" is listed in the dataset but was not found in the sidecar shard.`);
        }}
        return hydrated;
    }}

    function getColorValues(geneName=null, colorKey=null) {{
        if (geneName) return getGeneValues(geneName);
        const cfg = DATA.color_configs[colorKey || currentColor];
        if (!cfg) return null;
        return cfg.is_continuous ? cfg.values : cfg.codes;
    }}

    // Map a local cell index within a view to a global cell index
    function toGlobal(view, localIdx) {{
        return view.cell_indices ? view.cell_indices[localIdx] : localIdx;
    }}

    function getLocalColorValues(view, geneName=null, colorKey=null) {{
        const global = getColorValues(geneName, colorKey);
        if (!global) return null;
        if (!view.cell_indices) return global;
        // Return a view-local slice
        return view.cell_indices.map(gi => global[gi]);
    }}

    function getRenderedSpecs() {{
        const compareMode = Boolean(currentGene && compareGene && compareGene !== currentGene);
        const specs = [];
        DATA.views.forEach(view => {{
            if (compareMode) {{
                specs.push({{
                    panel_id: `${{view.id}}__geneA`,
                    view_id: view.id,
                    label: `${{view.name}} · ${{currentGene}}`,
                    gene: currentGene,
                    color: currentColor,
                }});
                specs.push({{
                    panel_id: `${{view.id}}__geneB`,
                    view_id: view.id,
                    label: `${{view.name}} · ${{compareGene}}`,
                    gene: compareGene,
                    color: currentColor,
                }});
            }} else {{
                specs.push({{
                    panel_id: view.id,
                    view_id: view.id,
                    label: view.name,
                    gene: currentGene,
                    color: currentColor,
                }});
            }}
        }});
        return specs;
    }}

    function getRenderSpec(panelId) {{
        return getRenderedSpecs().find(spec => spec.panel_id === panelId) || null;
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
    function renderView(view, canvas, isModal, spec=null) {{
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

        const geneName = spec?.gene ?? currentGene;
        const colorKey = spec?.color ?? currentColor;
        const values = getLocalColorValues(view, geneName, colorKey);
        const cfg    = getColorConfig(geneName, colorKey);
        if (!values || !cfg) return;

        const r = Math.max(0.5, SPOT_STEPS[spotStepIdx]);


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
        const specs = getRenderedSpecs();

        let totalCells = 0;
        const drawList = [];
        specs.forEach((spec, idx) => {{
            const view = getView(spec.view_id);
            if (!view) return;
            totalCells += view.n_cells;
            const panel  = panels[idx];
            if (!panel) return;
            const canvas = panel.querySelector('canvas');
            if (!canvas) return;
            if (gridRect) {{
                const pr = panel.getBoundingClientRect();
                if (pr.bottom < gridRect.top - 300 || pr.top > gridRect.bottom + 300) return;
            }}
            drawList.push({{view, canvas, spec}});
        }});

        const colorLabel = currentGene
            ? (compareGene && compareGene !== currentGene ? `${{currentGene}} vs ${{compareGene}}` : currentGene)
            : currentColor;
        document.getElementById('stats-text').textContent =
            `${{specs.length}} view${{specs.length>1?'s':''}} · ${{DATA.n_cells.toLocaleString()}} cells · ${{colorLabel}}`;

        let i = 0;
        function step() {{
            if (jobId !== renderAllJobId) return;
            const t0 = performance.now();
            while (i < drawList.length && performance.now() - t0 < 12) {{
                const {{view, canvas, spec}} = drawList[i++];
                try {{ renderView(view, canvas, false, spec); }}
                catch(e) {{ console.error('renderView failed', e); }}
            }}
            if (i < drawList.length) requestAnimationFrame(step);
        }}
        requestAnimationFrame(step);
    }}

    function renderModal() {{
        const spec = getRenderSpec(modalViewId);
        const view = spec ? getView(spec.view_id) : null;
        if (!view) return;
        const canvas = document.getElementById('modal-canvas');
        if (!canvas) return;
        renderView(view, canvas, true, spec);
    }}

    // ── Grid construction ──────────────────────────────────────────────────
    function buildGrid() {{
        const grid = document.getElementById('grid');
        grid.innerHTML = '';
        const specs = getRenderedSpecs();
        grid.classList.toggle('single-view-layout', specs.length === 1);

        const embeddingCounts = new Map();
        specs.forEach(spec => {{
            const view = getView(spec.view_id);
            if (!view || !view.embedding_key) return;
            const key = `${{view.embedding_key}}__${{spec.gene || 'base'}}`;
            embeddingCounts.set(key, (embeddingCounts.get(key) || 0) + 1);
        }});

        let lastGroupKey = null;
        specs.forEach(spec => {{
            const view = getView(spec.view_id);
            if (!view) return;
            const groupKey = `${{view.embedding_key || view.id}}__${{spec.gene || 'base'}}`;
            if (view.embedding_key && groupKey !== lastGroupKey &&
                (embeddingCounts.get(groupKey) || 0) > 1) {{
                const header = document.createElement('div');
                header.className = 'grid-group-header';
                header.textContent = spec.gene
                    ? `${{view.name.includes(' — ') ? view.name.split(' — ')[0] : view.embedding_key}} · ${{spec.gene}}`
                    : (view.name.includes(' — ') ? view.name.split(' — ')[0] : view.embedding_key);
                grid.appendChild(header);
            }}
            lastGroupKey = groupKey;

            const panel = document.createElement('div');
            panel.className = 'view-panel';
            panel.dataset.viewId = spec.panel_id;

            const lbl = document.createElement('div');
            lbl.className = 'panel-label';
            lbl.textContent = spec.label;

            const canvas = document.createElement('canvas');
            canvas.className = 'view-canvas';

            panel.appendChild(lbl);
            panel.appendChild(canvas);
            panel.addEventListener('click', () => openModal(spec.panel_id));
            grid.appendChild(panel);
        }});
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
        compareGene  = null;
        hiddenCategories.clear();
        spotlightCategory   = null;
        markerInsightsGroupby = null;
        markerInsightsGroup   = null;
        document.getElementById('gene-input').value = '';
        document.getElementById('gene2-input').value = '';
        document.getElementById('gene-clear-btn').classList.remove('visible');
        document.getElementById('gene2-clear-btn').classList.remove('visible');
        buildGrid();
        renderAllViews();
        if (modalViewId) renderModal();
        renderLegend();
        updateGeneDiscovery();
        updateInsights();
    }}

    async function setGene(name) {{
        if (!name) {{ clearGene(); return; }}
        // Case-insensitive lookup
        const lower = name.toLowerCase();
        const match = DATA.available_genes.find(g => g.toLowerCase() === lower);
        if (!match) return;
        const ready = await ensureGeneLoaded(match);
        if (!ready) return;
        currentGene = match;
        if (compareGene === currentGene) {{
            compareGene = null;
            document.getElementById('gene2-input').value = '';
            document.getElementById('gene2-clear-btn').classList.remove('visible');
        }}
        document.getElementById('gene-input').value = match;
        document.getElementById('gene-clear-btn').classList.add('visible');
        addRecentGene(match);
        buildGrid();
        renderAllViews();
        if (modalViewId) renderModal();
        renderLegend();
        updateGeneDiscovery();
        updateInsights();
    }}

    function clearGene() {{
        currentGene = null;
        compareGene = null;
        document.getElementById('gene-input').value = '';
        document.getElementById('gene2-input').value = '';
        document.getElementById('gene-clear-btn').classList.remove('visible');
        document.getElementById('gene2-clear-btn').classList.remove('visible');
        buildGrid();
        renderAllViews();
        if (modalViewId) renderModal();
        renderLegend();
        updateGeneDiscovery();
        updateInsights();
    }}

    async function setCompareGene(name) {{
        const value = name.trim();
        if (!value) {{ clearCompareGene(); return; }}
        const lower = value.toLowerCase();
        const match = DATA.available_genes.find(g => g.toLowerCase() === lower);
        if (!match) return;
        if (!currentGene) {{
            await setGene(match);
            return;
        }}
        const ready = await ensureGeneLoaded(match);
        if (!ready) return;
        compareGene = (match === currentGene) ? null : match;
        document.getElementById('gene2-input').value = compareGene || '';
        document.getElementById('gene2-clear-btn').classList.toggle('visible', Boolean(compareGene));
        if (compareGene) addRecentGene(compareGene);
        buildGrid();
        renderAllViews();
        if (modalViewId) renderModal();
        updateInsights();
    }}

    function clearCompareGene() {{
        compareGene = null;
        document.getElementById('gene2-input').value = '';
        document.getElementById('gene2-clear-btn').classList.remove('visible');
        buildGrid();
        renderAllViews();
        if (modalViewId) renderModal();
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
        const validRecent = recentGenes.filter(g => AVAILABLE_GENE_SET.has(g));
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
    function openModal(panelId) {{
        modalViewId = panelId;
        modalZoom   = 1;
        modalPanX   = 0;
        modalPanY   = 0;
        lassoPoints = [];
        const spec = getRenderSpec(panelId);
        document.getElementById('modal-title').textContent = spec?.label || panelId;
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
            const spec = getRenderSpec(modalViewId);
            const view = spec ? getView(spec.view_id) : null;
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
        const spec = getRenderSpec(modalViewId);
        const view = spec ? getView(spec.view_id) : null;
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

    async function renderMarkersInsights() {{
        const panel = document.getElementById('insights-markers');
        if (!panel) return;

        const allMarkers = DATA.analytics?.marker_genes || {{}};
        const groupbyKeys = Object.keys(allMarkers);
        if (!groupbyKeys.length) {{
            panel.innerHTML = `<div class="insights-empty">No marker genes were precomputed for this export.</div>`;
            return;
        }}

        // Resolve active groupby
        if (!groupbyKeys.includes(markerInsightsGroupby)) {{
            markerInsightsGroupby = groupbyKeys.includes(currentColor) ? currentColor : groupbyKeys[0];
        }}
        const groupMap   = allMarkers[markerInsightsGroupby] || {{}};
        const groupKeys  = Object.keys(groupMap);
        const cfg        = getCategoricalConfig(markerInsightsGroupby);

        // Resolve active cluster — prefer current spotlight, then keep last, then first
        if (!groupKeys.includes(markerInsightsGroup)) {{
            markerInsightsGroup = (spotlightCategory && groupKeys.includes(spotlightCategory))
                ? spotlightCategory
                : groupKeys[0] || null;
        }}

        const activeGroup = markerInsightsGroup;
        const entries     = (groupMap[activeGroup] || []).slice(0, 30);
        const activeIdx   = cfg ? cfg.categories.indexOf(activeGroup) : -1;
        const activeColor = activeIdx >= 0 ? PALETTE[activeIdx % PALETTE.length] : 'var(--accent)';

        // ── Build HTML ──
        let html = '';

        // Groupby selector (only when multiple groupbys exist)
        if (groupbyKeys.length > 1) {{
            html += `<div class="insights-control" style="margin-bottom:6px;">
                <label for="marker-ins-groupby">Grouping</label>
                <select id="marker-ins-groupby">
                    ${{groupbyKeys.map(k =>
                        `<option value="${{escapeHtml(k)}}"${{k === markerInsightsGroupby ? ' selected' : ''}}>${{escapeHtml(k)}}</option>`
                    ).join('')}}
                </select>
            </div>`;
        }}

        // Cluster chip row
        html += `<div class="cluster-chip-row">`;
        groupKeys.forEach(g => {{
            const idx = cfg ? cfg.categories.indexOf(g) : -1;
            const col = idx >= 0 ? PALETTE[idx % PALETTE.length] : '#888';
            html += `<button class="cluster-chip${{g === activeGroup ? ' active' : ''}}"
                        data-group="${{escapeHtml(g)}}"
                        style="--chip-color:${{col}}">${{escapeHtml(g)}}</button>`;
        }});
        html += `</div>`;

        // Gene list
        if (entries.length) {{
            const maxScore = Math.max(...entries.map(e => Math.abs(e.score || 0)), 1e-9);
            html += `<div class="marker-list" style="margin-top:10px;">`;
            entries.forEach(entry => {{
                const pct = Math.max(0, Math.min(100, Math.abs(entry.score || 0) / maxScore * 100));
                html += `<div class="marker-row">
                    <div class="marker-head">
                        <button class="gene-link-btn" data-gene="${{escapeHtml(entry.gene)}}">${{escapeHtml(entry.gene)}}</button>
                        <span class="marker-meta">score ${{fmtNum(entry.score)}} · ${{((entry.pct_expr || 0) * 100).toFixed(1)}}%</span>
                    </div>
                    <div class="marker-bar-track">
                        <div class="marker-bar-fill" style="width:${{pct}}%;background:${{activeColor}}"></div>
                    </div>
                </div>`;
            }});
            html += `</div>`;

            // Dotplot of top marker genes across all clusters
            html += `<div class="marker-dotplot-wrap">
                <div class="marker-dotplot-label">Expression across clusters</div>
                <canvas class="dotplot-canvas" id="marker-dotplot-canvas"></canvas>
            </div>`;
        }} else {{
            html += `<div class="insights-empty" style="margin-top:8px;">No markers for this cluster.</div>`;
        }}

        panel.innerHTML = html;

        // Wire cluster chip clicks
        panel.querySelectorAll('.cluster-chip').forEach(btn => {{
            btn.addEventListener('click', () => {{
                markerInsightsGroup = btn.dataset.group;
                renderMarkersInsights();
            }});
        }});

        // Wire groupby selector
        document.getElementById('marker-ins-groupby')?.addEventListener('change', e => {{
            markerInsightsGroupby = e.target.value;
            markerInsightsGroup   = null;
            renderMarkersInsights();
        }});

        // Wire gene link clicks
        panel.querySelectorAll('.gene-link-btn').forEach(btn => {{
            btn.addEventListener('click', () => setGene(btn.dataset.gene));
        }});

        // Draw dotplot: top 15 marker genes × all visible clusters (load from sidecar on demand)
        if (entries.length && cfg) {{
            const dotGenes = entries.slice(0, 15).map(e => e.gene)
                .filter(g => AVAILABLE_GENE_SET.has(g));
            const dotCats  = cfg.categories.filter(cat => !hiddenCategories.has(cat));
            if (dotGenes.length && dotCats.length) {{
                const markerCanvas = document.getElementById('marker-dotplot-canvas');
                const toLoad = dotGenes.filter(g => !(DATA.genes?.[g]));
                if (toLoad.length > 0) {{
                    _paintCanvasMsg(markerCanvas,
                        `Loading ${{toLoad.length}} gene${{toLoad.length > 1 ? 's' : ''}}…`);
                    await Promise.all(toLoad.map(g => ensureGeneLoaded(g, {{ showErrors: false }})));
                }}
                const readyGenes = dotGenes.filter(g => DATA.genes?.[g]);
                if (readyGenes.length) {{
                    drawDotplot(
                        document.getElementById('marker-dotplot-canvas'),
                        markerInsightsGroupby,
                        readyGenes,
                        dotCats,
                    );
                }}
            }}
        }}
    }}

    async function renderTableInsights() {{
        const panel = document.getElementById('insights-table');
        if (!panel) return;

        const cfg = getCategoricalConfig(currentColor);

        // Resolve gene: prefer explicit tableGeneText, fall back to currentGene
        const geneInput = tableGeneText.trim();
        const lower = geneInput.toLowerCase();
        const gene = geneInput
            ? (DATA.available_genes.find(g => g.toLowerCase() === lower) || null)
            : currentGene;

        // Build UI immediately so input stays responsive
        const inputVal = tableGeneText;
        panel.innerHTML = `
            <div class="insights-controls">
                <div class="insights-control">
                    <label for="table-gene-input">Gene</label>
                    <input type="text" id="table-gene-input"
                           placeholder="Type any gene…"
                           value="${{escapeHtml(inputVal)}}"
                           list="gene-datalist" autocomplete="off" />
                </div>
            </div>
            <div id="table-body-area"></div>`;

        // Wire input — only re-render on commit (Enter or blur) to avoid re-render per keystroke
        const inputEl = document.getElementById('table-gene-input');
        inputEl?.addEventListener('input', e => {{
            tableGeneText = e.target.value;
        }});
        inputEl?.addEventListener('change', () => renderTableInsights());
        inputEl?.addEventListener('keydown', e => {{
            if (e.key === 'Enter') renderTableInsights();
        }});

        const bodyEl = document.getElementById('table-body-area');

        if (!gene) {{
            bodyEl.innerHTML = `<div class="insights-empty">Type a gene name above to view per-cluster expression.</div>`;
            return;
        }}
        if (!cfg) {{
            bodyEl.innerHTML = `<div class="insights-empty">Select a categorical obs column as the color to group cells.</div>`;
            return;
        }}

        // Load gene on demand
        if (!getGeneValues(gene)) {{
            bodyEl.innerHTML = `<div class="insights-empty">Loading <b>${{escapeHtml(gene)}}</b>…</div>`;
            const loaded = await ensureGeneLoaded(gene, {{ showErrors: false }});
            if (!loaded) {{
                bodyEl.innerHTML = `<div class="insights-empty">Gene <b>${{escapeHtml(gene)}}</b> is not available.</div>`;
                return;
            }}
        }}

        const values = getGeneValues(gene);
        const cats = cfg.categories.filter(cat => !hiddenCategories.has(cat));

        const rows = cats.map(cat => {{
            const catIdx = cfg.categories.indexOf(cat);
            const mask = Array.from(cfg.codes).map(c => c === catIdx);
            const cellVals = values.filter((_, i) => mask[i] && isFinite(values[i]));
            const n = cellVals.length;
            const mean = n > 0 ? cellVals.reduce((s, v) => s + v, 0) / n : 0;
            const pctExpr = n > 0 ? cellVals.filter(v => v > 0).length / n : 0;
            const paletteColor = PALETTE[catIdx % PALETTE.length];
            return {{ cat, n, mean, pctExpr, paletteColor }};
        }});

        bodyEl.innerHTML = `
            <div style="padding:4px 0 6px;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);">
                ${{escapeHtml(gene)}} · ${{escapeHtml(currentColor)}}
            </div>
            <table class="insights-table">
                <thead>
                    <tr><th>Cluster</th><th>Mean</th><th>% expr</th><th>N</th></tr>
                </thead>
                <tbody>
                    ${{rows.map(r => `
                        <tr>
                            <td><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${{r.paletteColor}};margin-right:5px;vertical-align:middle;"></span>${{escapeHtml(r.cat)}}</td>
                            <td>${{r.mean.toFixed(3)}}</td>
                            <td>${{(r.pctExpr * 100).toFixed(1)}}%</td>
                            <td>${{r.n}}</td>
                        </tr>
                    `).join('')}}
                </tbody>
            </table>`;
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
                                        <td><button class="gene-link-btn" data-gene="${{escapeHtml(entry.gene)}}">${{escapeHtml(entry.gene)}}</button></td>
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

        panel.querySelectorAll('.gene-link-btn').forEach(btn => {{
            btn.addEventListener('click', () => setGene(btn.dataset.gene));
        }});

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

    function _getDotplotGenes() {{
        return dotplotGeneText.split(',').map(g => g.trim()).filter(Boolean);
    }}

    function _setDotplotGenes(genes) {{
        dotplotGeneText = [...new Set(genes)].join(', ');
    }}

    function _renderDotplotChips() {{
        const box = document.getElementById('dotplot-chip-box');
        if (!box) return;
        const inputEl = box.querySelector('input');
        box.querySelectorAll('.gene-chip').forEach(c => c.remove());
        _getDotplotGenes().forEach(gene => {{
            const chip = document.createElement('span');
            chip.className = 'gene-chip';
            const rm = document.createElement('button');
            rm.className = 'chip-remove';
            rm.textContent = '×';
            rm.title = `Remove ${{gene}}`;
            rm.addEventListener('click', e => {{
                e.stopPropagation();
                _setDotplotGenes(_getDotplotGenes().filter(g => g !== gene));
                _renderDotplotChips();
                renderDotplotInsights();
            }});
            chip.appendChild(document.createTextNode(gene + ' '));
            chip.appendChild(rm);
            box.insertBefore(chip, inputEl);
        }});
        if (inputEl) {{
            inputEl.placeholder = _getDotplotGenes().length ? 'Add gene…' : 'Gene1, Gene2…';
        }}
    }}

    function _setupDotplotChipInput() {{
        const box = document.getElementById('dotplot-chip-box');
        if (!box) return;

        const inputEl = document.createElement('input');
        inputEl.type = 'text';
        inputEl.setAttribute('list', 'gene-datalist');
        inputEl.setAttribute('autocomplete', 'off');
        box.appendChild(inputEl);

        box.addEventListener('click', () => inputEl.focus());

        inputEl.addEventListener('keydown', e => {{
            if (e.key === 'Enter' || e.key === ',') {{
                e.preventDefault();
                const raw = inputEl.value.replace(/,/g, '').trim();
                if (!raw) return;
                const lower = raw.toLowerCase();
                const match = DATA.available_genes.find(g => g.toLowerCase() === lower);
                inputEl.value = '';
                if (match) {{
                    _setDotplotGenes([..._getDotplotGenes(), match]);
                    _renderDotplotChips();
                    renderDotplotInsights();
                }}
            }} else if (e.key === 'Backspace' && !inputEl.value) {{
                const genes = _getDotplotGenes();
                if (genes.length) {{
                    _setDotplotGenes(genes.slice(0, -1));
                    _renderDotplotChips();
                    renderDotplotInsights();
                }}
            }}
        }});
    }}

    async function renderDotplotInsights() {{
        const panel = document.getElementById('insights-dotplot');
        if (!panel) return;

        const cfg = getCategoricalConfig(currentColor);
        if (!cfg) {{
            panel.innerHTML = `<div class="insights-empty">Dotplot requires a categorical obs column as the active color.</div>`;
            return;
        }}

        // Build the shell once — do NOT rebuild on subsequent calls to preserve chip input focus
        if (!panel.querySelector('.dotplot-shell')) {{
            panel.innerHTML = `
                <div class="dotplot-shell">
                    <div class="insights-control">
                        <label>Genes</label>
                        <div class="gene-chip-input" id="dotplot-chip-box"></div>
                    </div>
                    <canvas class="dotplot-canvas" id="dotplot-canvas"></canvas>
                    <div class="dotplot-note">Enter or , to add · Backspace to remove</div>
                </div>`;
            _setupDotplotChipInput();
        }}
        _renderDotplotChips();

        // Determine candidate genes
        const explicitGenes = _getDotplotGenes();
        let candidateGenes;
        if (explicitGenes.length) {{
            candidateGenes = explicitGenes.filter(g => AVAILABLE_GENE_SET.has(g));
        }} else {{
            const markerGroup = DATA.analytics?.marker_genes?.[currentColor] || null;
            const activeCategory = spotlightCategory || getSingleSelectedCategory(currentColor);
            const defaultGenes = activeCategory && markerGroup?.[activeCategory]
                ? markerGroup[activeCategory].slice(0, 8).map(entry => entry.gene)
                : [];
            if (currentGene) defaultGenes.unshift(currentGene);
            if (compareGene) defaultGenes.unshift(compareGene);
            candidateGenes = [...new Set(defaultGenes)].filter(g => AVAILABLE_GENE_SET.has(g));
        }}

        const categories = (spotlightCategory
            ? [spotlightCategory]
            : cfg.categories.filter(cat => !hiddenCategories.has(cat))
        ).slice(0, 12);

        // Load genes on demand
        const toLoad = candidateGenes.filter(g => !(DATA.genes?.[g]));
        if (toLoad.length > 0) {{
            _paintCanvasMsg(document.getElementById('dotplot-canvas'),
                `Loading ${{toLoad.length}} gene${{toLoad.length > 1 ? 's' : ''}}…`);
            await Promise.all(toLoad.map(g => ensureGeneLoaded(g, {{ showErrors: false }})));
        }}

        const genes = candidateGenes.filter(g => DATA.genes?.[g]);

        if (!genes.length || !categories.length) {{
            _paintCanvasMsg(document.getElementById('dotplot-canvas'),
                genes.length === 0
                    ? 'Add genes above to display the dotplot.'
                    : 'No categories to display.');
            return;
        }}

        drawDotplot(document.getElementById('dotplot-canvas'), currentColor, genes, categories);
    }}

    // ── Canvas utility ────────────────────────────────────────────────────
    function _paintCanvasMsg(canvas, msg) {{
        if (!canvas) return;
        const dpr  = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width  = (rect.width  || 200) * dpr;
        canvas.height = (rect.height || 60)  * dpr;
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.fillStyle = getPanelBg();
        ctx.fillRect(0, 0, rect.width || 200, rect.height || 60);
        ctx.fillStyle = theme === 'dark' ? '#888' : '#999';
        ctx.font = '11px -apple-system, sans-serif';
        ctx.textBaseline = 'middle';
        ctx.fillText(msg, 10, (rect.height || 60) / 2);
    }}

    // ── Boxplot ────────────────────────────────────────────────────────────
    function computeBoxStats(values, codes, catIdx) {{
        const cellVals = [];
        for (let i = 0; i < DATA.n_cells; i++) {{
            const code = codes?.[i];
            if (code == null || !Number.isFinite(code)) continue;
            if (Math.round(code) !== catIdx) continue;
            const v = values[i];
            if (v != null && Number.isFinite(v)) cellVals.push(v);
        }}
        if (!cellVals.length) return null;
        cellVals.sort((a, b) => a - b);
        const n = cellVals.length;
        const q1  = cellVals[Math.floor(n * 0.25)];
        const med = cellVals[Math.floor(n * 0.5)];
        const q3  = cellVals[Math.floor(n * 0.75)];
        const iqr = q3 - q1;
        const wlo = Math.max(cellVals[0],     q1 - 1.5 * iqr);
        const whi = Math.min(cellVals[n - 1], q3 + 1.5 * iqr);
        return {{ q1, med, q3, wlo, whi, n }};
    }}

    function drawBoxplot(canvas, statsList, catMeta) {{
        if (!canvas || !statsList.length) return;
        const dpr  = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        if (!rect.width || !rect.height) return;
        canvas.width  = rect.width  * dpr;
        canvas.height = rect.height * dpr;

        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.fillStyle = getPanelBg();
        ctx.fillRect(0, 0, rect.width, rect.height);

        const left = 44, right = 6, top = 10, bottom = 58;
        const innerW = rect.width  - left - right;
        const innerH = rect.height - top  - bottom;

        const allVals = statsList.flatMap(s => s ? [s.wlo, s.q1, s.med, s.q3, s.whi] : []);
        if (!allVals.length) return;
        const ymin   = Math.min(...allVals);
        const ymax   = Math.max(...allVals);
        const yrange = Math.max(ymax - ymin, 1e-9);
        const toY    = v => top + innerH * (1 - (v - ymin) / yrange);

        const slotW = innerW / statsList.length;
        const barW  = Math.max(6, Math.min(28, slotW * 0.55));

        // Gridlines + y-axis labels
        ctx.font = '9px -apple-system, sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = theme === 'dark' ? '#888' : '#999';
        for (let t = 0; t <= 4; t++) {{
            const v = ymin + yrange * t / 4;
            const y = toY(v);
            ctx.fillText(fmtNum(v), left - 4, y);
            ctx.strokeStyle = theme === 'dark' ? '#333' : '#f0f0f0';
            ctx.lineWidth = 0.8;
            ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + innerW, y); ctx.stroke();
        }}
        // Y axis spine
        ctx.strokeStyle = theme === 'dark' ? '#555' : '#ccc';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(left, top); ctx.lineTo(left, top + innerH); ctx.stroke();

        // Draw each box
        statsList.forEach((s, idx) => {{
            if (!s) return;
            const meta  = catMeta[idx];
            const color = PALETTE[meta.catIdx % PALETTE.length];
            const xc    = left + (idx + 0.5) * slotW;

            ctx.strokeStyle = color;
            ctx.lineWidth   = 1.5;

            // Whisker lines
            ctx.globalAlpha = 0.7;
            ctx.beginPath();
            ctx.moveTo(xc, toY(s.whi)); ctx.lineTo(xc, toY(s.q3));
            ctx.moveTo(xc - barW * 0.25, toY(s.whi)); ctx.lineTo(xc + barW * 0.25, toY(s.whi));
            ctx.moveTo(xc, toY(s.q1));  ctx.lineTo(xc, toY(s.wlo));
            ctx.moveTo(xc - barW * 0.25, toY(s.wlo)); ctx.lineTo(xc + barW * 0.25, toY(s.wlo));
            ctx.stroke();

            // IQR box
            const boxTop = toY(s.q3);
            const boxH   = Math.max(1, toY(s.q1) - toY(s.q3));
            ctx.globalAlpha = 0.18;
            ctx.fillStyle   = color;
            ctx.fillRect(xc - barW / 2, boxTop, barW, boxH);
            ctx.globalAlpha = 0.85;
            ctx.strokeRect(xc - barW / 2, boxTop, barW, boxH);

            // Median line
            ctx.globalAlpha = 1;
            ctx.lineWidth   = 2;
            ctx.beginPath();
            ctx.moveTo(xc - barW / 2, toY(s.med));
            ctx.lineTo(xc + barW / 2, toY(s.med));
            ctx.stroke();
        }});
        ctx.globalAlpha = 1;

        // X-axis category labels (rotated)
        ctx.fillStyle = theme === 'dark' ? '#cfcfcf' : '#444';
        ctx.font = '10px -apple-system, sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        statsList.forEach((_, idx) => {{
            const xc = left + (idx + 0.5) * slotW;
            ctx.save();
            ctx.translate(xc, top + innerH + 5);
            ctx.rotate(-Math.PI / 4);
            ctx.fillText(catMeta[idx].cat, 0, 0);
            ctx.restore();
        }});
    }}

    async function renderBoxplotInsights() {{
        const panel = document.getElementById('insights-boxplot');
        if (!panel) return;

        const gene = currentGene;
        if (!gene) {{
            panel.innerHTML = `<div class="insights-empty">Select a gene above to view per-group expression distributions.</div>`;
            return;
        }}

        const cfg = getCategoricalConfig(currentColor);
        if (!cfg) {{
            panel.innerHTML = `<div class="insights-empty">Select a categorical obs column as the color to group cells.</div>`;
            return;
        }}

        // Load gene on demand (works with sidecar / karospace exports)
        if (!getGeneValues(gene)) {{
            panel.innerHTML = `<div class="insights-empty">Loading <b>${{escapeHtml(gene)}}</b>…</div>`;
            const loaded = await ensureGeneLoaded(gene, {{ showErrors: false }});
            if (!loaded) {{
                panel.innerHTML = `<div class="insights-empty">Gene <b>${{escapeHtml(gene)}}</b> is not available in this export.</div>`;
                return;
            }}
        }}

        const values = getGeneValues(gene);
        const catMeta = cfg.categories
            .map((cat, catIdx) => ({{ cat, catIdx }}))
            .filter(c => !hiddenCategories.has(c.cat));
        const statsList = catMeta.map(c => computeBoxStats(values, cfg.codes, c.catIdx));

        panel.innerHTML = `
            <div style="padding:4px 0 6px;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);">
                ${{escapeHtml(gene)}} · ${{escapeHtml(currentColor)}}
            </div>
            <canvas class="boxplot-canvas" id="boxplot-canvas"></canvas>
            <div class="dotplot-note" style="margin-top:4px;">Box: IQR · line: median · whiskers: 1.5×IQR</div>`;

        requestAnimationFrame(() => {{
            drawBoxplot(document.getElementById('boxplot-canvas'), statsList, catMeta);
        }});
    }}

    function updateInsights() {{
        renderStatsInsights();
        renderMarkersInsights();
        renderBoxplotInsights();
        renderTableInsights();
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

        document.querySelectorAll('#sidebar-tabs .insights-tab').forEach(btn => {{
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

        const gene2Input = document.getElementById('gene2-input');
        gene2Input.addEventListener('input', () => {{
            const val = gene2Input.value.trim();
            document.getElementById('gene2-clear-btn').classList.toggle('visible', val.length > 0);
        }});
        gene2Input.addEventListener('keydown', e => {{
            if (e.key === 'Enter') {{
                setCompareGene(gene2Input.value.trim());
                gene2Input.blur();
            }}
            if (e.key === 'Escape') {{
                clearCompareGene();
                gene2Input.blur();
            }}
        }});
        gene2Input.addEventListener('change', () => {{
            setCompareGene(gene2Input.value.trim());
        }});
        document.getElementById('gene2-clear-btn').addEventListener('click', clearCompareGene);

        document.addEventListener('click', e => {{
            const shell = document.getElementById('gene-input')?.closest('.gene-input-shell');
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
        requestAnimationFrame(renderAllViews);
        setupEvents();
        updateInsights();

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
    gene_storage: str = "embedded",
    gene_aux_path: Optional[Union[str, Path]] = None,
    gene_sidecar_shard_size: int = GENE_SIDECAR_SHARD_SIZE,
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
    gene_storage
        ``'embedded'`` keeps selected genes inside the HTML. ``'sidecar'``
        stores only selected genes in the HTML and writes the remaining genes
        to a JSON manifest plus shard files.
    gene_aux_path
        Optional path for the sidecar manifest when ``gene_storage='sidecar'``.
    gene_sidecar_shard_size
        Number of genes per sidecar shard.
    vmin, vmax
        Optional fixed min/max for continuous colour scales.

    Returns
    -------
    str
        Path to the written HTML file.
    """
    adata = dataset.adata
    requested_output_path = Path(output_path).expanduser().resolve()
    package_mode = requested_output_path.suffix.lower() == ".karospace"
    gene_storage = str(gene_storage or "embedded").strip().lower()
    if gene_storage not in {"embedded", "sidecar"}:
        raise ValueError("gene_storage must be one of: 'embedded', 'sidecar'")
    gene_sidecar_shard_size = int(gene_sidecar_shard_size)
    if gene_sidecar_shard_size < 1:
        raise ValueError("gene_sidecar_shard_size must be >= 1")
    if package_mode and gene_storage != "sidecar":
        raise ValueError(".karospace export requires gene_storage='sidecar'")

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
    sidecar_genes = []
    if gene_storage == "sidecar":
        embedded_set = set(embed_genes)
        sidecar_genes = [gene for gene in dataset.var_names if gene not in embedded_set]
    print(f"  Embedding {len(embed_genes)} genes…")
    if gene_storage == "sidecar":
        print(f"  Sidecar genes: {len(sidecar_genes)}")

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
        "gene_aux_url":   None,
        "analytics":      analytics,
        "paga":           paga_payload,
        "velocity_embedding": velocity_payload,
    }

    package_output_path: Optional[Path] = None
    html_output_path: Path
    resolved_gene_aux_path: Optional[Path] = None
    resolved_gene_aux_dir: Optional[Path] = None
    if package_mode:
        package_output_path = requested_output_path
        html_output_path = requested_output_path.with_suffix(".html")
    else:
        html_output_path = requested_output_path

    if gene_storage == "sidecar":
        if gene_aux_path is not None:
            aux_candidate = Path(gene_aux_path).expanduser()
            if not aux_candidate.is_absolute():
                aux_candidate = (html_output_path.parent / aux_candidate).resolve()
            resolved_gene_aux_path = aux_candidate
        else:
            resolved_gene_aux_path = html_output_path.with_suffix(".genes.json")
        resolved_gene_aux_dir = resolved_gene_aux_path.with_suffix("")
        if package_mode:
            payload["gene_aux_url"] = resolved_gene_aux_path.name
        else:
            payload["gene_aux_url"] = Path(
                os.path.relpath(resolved_gene_aux_path, start=html_output_path.parent)
            ).as_posix()

    # ── Render HTML ──────────────────────────────────────────────────────
    data_json = json.dumps(payload, separators=(",", ":"))
    palette_json = json.dumps(DEFAULT_PALETTE)

    html = HTML_TEMPLATE.format(
        title=title,
        data_json=data_json,
        palette_json=palette_json,
    )

    html_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(html_output_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    if resolved_gene_aux_path is not None:
        assert resolved_gene_aux_dir is not None
        resolved_gene_aux_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_gene_aux_dir.mkdir(parents=True, exist_ok=True)
        shard_groups = _chunked(sidecar_genes, gene_sidecar_shard_size)
        manifest = {
            "format": "karospace-gene-sidecar-manifest-v2",
            "gene_to_shard": {},
            "genes_meta": {},
            "gene_encodings": {},
            "shards": {},
        }
        output_parent = html_output_path.parent
        total_sidecar_genes = len(sidecar_genes)
        total_shards = len(shard_groups)
        if total_sidecar_genes:
            print(
                f"  Building gene sidecar: {total_sidecar_genes} genes across "
                f"{total_shards} shard{'s' if total_shards != 1 else ''}…"
            )
        genes_written = 0
        for shard_idx, shard_genes in enumerate(shard_groups):
            shard_filename = f"{shard_idx:03d}.json"
            shard_path = resolved_gene_aux_dir / shard_filename
            shard_rel = Path(os.path.relpath(shard_path, start=output_parent)).as_posix()
            shard_data = _build_gene_sidecar_shard(
                dataset,
                shard_genes,
                gene_sparse_threshold=gene_sparse_threshold,
            )
            manifest["shards"][shard_rel] = shard_genes
            for gene in shard_genes:
                manifest["gene_to_shard"][gene] = shard_rel
                if gene in shard_data.get("genes_meta", {}):
                    manifest["genes_meta"][gene] = shard_data["genes_meta"][gene]
                if gene in shard_data.get("gene_encodings", {}):
                    manifest["gene_encodings"][gene] = shard_data["gene_encodings"][gene]
            with open(shard_path, "w", encoding="utf-8") as fh:
                json.dump(shard_data, fh, separators=(",", ":"))
            genes_written += len(shard_genes)
            print(f"    wrote {shard_filename} ({genes_written}/{total_sidecar_genes} genes)")
        with open(resolved_gene_aux_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, separators=(",", ":"))

    if package_mode:
        assert package_output_path is not None
        assert resolved_gene_aux_path is not None
        assert resolved_gene_aux_dir is not None
        written_loader_path = None
        with tempfile.TemporaryDirectory(prefix="sckaro-package-") as tmpdir:
            bundle_root = Path(tmpdir)
            entry_html = "index.html"
            bundle_html_path = bundle_root / entry_html
            bundle_html_path.write_text(html, encoding="utf-8")
            package_gene_manifest_name = resolved_gene_aux_path.name
            package_gene_manifest_path = bundle_root / package_gene_manifest_name
            package_gene_manifest_path.write_text(
                resolved_gene_aux_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            package_gene_shard_dir = bundle_root / Path(package_gene_manifest_name).with_suffix("")
            package_gene_shard_dir.mkdir(parents=True, exist_ok=True)
            for shard_file in sorted(resolved_gene_aux_dir.glob("*.json")):
                (package_gene_shard_dir / shard_file.name).write_text(
                    shard_file.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            _write_karospace_package(
                package_path=package_output_path,
                source_root=bundle_root,
                entry_html=entry_html,
                gene_manifest_path=package_gene_manifest_name,
                gene_shard_dir=Path(package_gene_manifest_name).with_suffix("").as_posix(),
                title=title,
                n_views=dataset.n_views,
                total_cells=dataset.n_cells,
            )
        written_loader_path = _write_karospace_package_loader(
            loader_path=package_output_path.with_suffix(".loader.html")
        )
        size_mb = package_output_path.stat().st_size / 1e6
        print(f"  Written package to {package_output_path} ({size_mb:.1f} MB)")
        if written_loader_path is not None:
            print(f"  Written loader to {written_loader_path}")
        else:
            print("  Warning: package loader template not found; no loader HTML was written")
        return str(package_output_path)

    size_mb = html_output_path.stat().st_size / 1e6
    print(f"  Written to {html_output_path} ({size_mb:.1f} MB)")
    return str(html_output_path)
