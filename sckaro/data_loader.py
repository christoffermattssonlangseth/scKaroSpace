"""
Data loading for single-cell RNA-seq data.

Loads h5ad files, extracts 2D embedding coordinates from obsm,
and builds EmbeddingView panels for the interactive viewer.
"""

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
from pandas.api.types import CategoricalDtype
from scipy.sparse import issparse


# Human-readable names for common obsm keys
EMBEDDING_DISPLAY_NAMES: Dict[str, str] = {
    "X_umap": "UMAP",
    "X_tsne": "t-SNE",
    "X_pca": "PCA",
    "X_diffmap": "Diffmap",
    "X_draw_graph_fr": "Force Atlas",
    "X_draw_graph_kk": "Kamada-Kawai",
    "X_phate": "PHATE",
    "X_scvi": "scVI",
    "X_harmony": "Harmony",
    "X_monocle3_umap": "Monocle3",
    "X_palantir_diff_comp": "Palantir",
}


# ── H5AD loading helpers (ported from KaroSpace) ────────────────────────────

def _strip_null_encoded_h5ad_entries(src_path: str) -> Tuple[str, List[str]]:
    """Copy an h5ad and remove null-encoded datasets unsupported by older anndata."""
    import h5py

    fd, tmp_path = tempfile.mkstemp(suffix=".h5ad")
    os.close(fd)
    shutil.copy2(src_path, tmp_path)

    removed: List[str] = []
    with h5py.File(tmp_path, "r+") as f:
        def _walk(group, prefix=""):
            for key in list(group.keys()):
                obj = group[key]
                path = f"{prefix}/{key}" if prefix else f"/{key}"
                if obj.attrs.get("encoding-type") == "null":
                    removed.append(path)
                    del group[key]
                    continue
                if isinstance(obj, h5py.Group):
                    _walk(obj, path)
        _walk(f)

    return tmp_path, removed


def _read_h5ad_with_fallback(path: str) -> sc.AnnData:
    """Read h5ad, retrying with null-encoded entries stripped if needed."""
    try:
        return sc.read_h5ad(path)
    except Exception as exc:
        if "encoding_type='null'" not in str(exc):
            raise
        print("  Detected unsupported null-encoded H5AD fields; retrying with sanitized copy...")
        tmp_path = None
        try:
            tmp_path, removed = _strip_null_encoded_h5ad_entries(path)
            if removed:
                print(f"  Removed {len(removed)} null field(s): {', '.join(removed)}")
            return sc.read_h5ad(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class EmbeddingView:
    """One panel in the viewer grid — a 2D projection of cells."""
    id: str                            # unique panel id, e.g. "X_umap" or "X_umap__sample_A"
    name: str                          # display label, e.g. "UMAP" or "UMAP — sample_A"
    embedding_key: str                 # obsm key this came from
    x: np.ndarray                      # (n_cells,) float32 x coords
    y: np.ndarray                      # (n_cells,) float32 y coords
    cell_indices: Optional[np.ndarray] = None  # global obs indices (None = all in order)
    split_value: Optional[str] = None  # group value when created via split_by

    @property
    def n_cells(self) -> int:
        return len(self.x)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (
            float(np.nanmin(self.x)),
            float(np.nanmax(self.x)),
            float(np.nanmin(self.y)),
            float(np.nanmax(self.y)),
        )


@dataclass
class ScDataset:
    """Container for a single-cell dataset ready for export."""
    adata: sc.AnnData
    views: List[EmbeddingView]
    obs_columns: List[str]    # obs columns available for coloring
    var_names: List[str]      # gene names

    @property
    def n_cells(self) -> int:
        return self.adata.n_obs

    @property
    def n_views(self) -> int:
        return len(self.views)

    def get_color_data(
        self,
        color: str,
        cell_indices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool, Optional[List[str]]]:
        """
        Return (values, is_continuous, categories).

        For categorical obs: values are float category codes (NaN for missing).
        For continuous obs/gene: values are float expression/metadata values.
        """
        adata = self.adata if cell_indices is None else self.adata[cell_indices]

        if color in adata.obs.columns:
            col = adata.obs[color]
            if isinstance(col.dtype, CategoricalDtype):
                categories = list(col.cat.categories)
                codes = col.cat.codes.to_numpy().astype(float)
                codes[codes < 0] = np.nan
                return codes, False, categories
            elif pd.api.types.is_numeric_dtype(col):
                return col.to_numpy(dtype=float), True, None
            else:
                cat = col.astype("category")
                categories = list(cat.cat.categories)
                codes = cat.cat.codes.to_numpy().astype(float)
                codes[codes < 0] = np.nan
                return codes, False, categories

        elif color in adata.var_names:
            gene_idx = adata.var_names.get_loc(color)
            # Prefer a normalised/log1p layer when available
            layer = None
            for layer_key in ("normalized", "log1p", "lognorm"):
                if layer_key in adata.layers:
                    layer = adata.layers[layer_key]
                    break
            x = layer[:, gene_idx] if layer is not None else adata.X[:, gene_idx]
            if issparse(x):
                values = np.asarray(x.toarray()).ravel()
            else:
                values = np.asarray(x).ravel()
            return values, True, None

        raise KeyError(f"{color!r} not found in obs columns or var_names")

    def _collect_gene_data(
        self,
        genes: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Collect gene expression arrays and value ranges for export."""
        result: Dict[str, Dict[str, Any]] = {}
        for gene in genes or []:
            if gene not in self.adata.var_names:
                continue
            try:
                vals, _, _ = self.get_color_data(gene)
                finite = np.isfinite(vals)
                result[gene] = {
                    "values": vals,
                    "vmin": float(np.nanmin(vals[finite])) if finite.any() else 0.0,
                    "vmax": float(np.nanmax(vals[finite])) if finite.any() else 1.0,
                }
            except Exception as e:
                print(f"  Warning: could not load gene '{gene}': {e}")
        return result


# ── obs column selection ─────────────────────────────────────────────────────

def _select_obs_columns(
    adata: sc.AnnData,
    max_cat_categories: int = 200,
) -> List[str]:
    """Return obs columns suitable for cell colouring."""
    result = []
    for col in adata.obs.columns:
        series = adata.obs[col]
        if series.isna().mean() > 0.9:
            continue
        dtype = series.dtype
        if isinstance(dtype, CategoricalDtype):
            if 1 < len(series.cat.categories) <= max_cat_categories:
                result.append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            result.append(col)
        else:
            unique_n = series.nunique(dropna=True)
            if 1 < unique_n <= max_cat_categories:
                result.append(col)
    return result


# ── Embedding extraction ─────────────────────────────────────────────────────

def _extract_2d_embedding(
    adata: sc.AnnData,
    key: str,
    dims: Tuple[int, int] = (0, 1),
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract two float32 coordinate arrays from adata.obsm[key]."""
    if key not in adata.obsm:
        return None
    emb = np.asarray(adata.obsm[key])
    if emb.ndim != 2 or emb.shape[1] <= max(dims):
        return None
    x = emb[:, dims[0]].astype(np.float32)
    y = emb[:, dims[1]].astype(np.float32)
    return x, y


def _embedding_display_name(key: str) -> str:
    return EMBEDDING_DISPLAY_NAMES.get(key, key.replace("X_", "").upper())


# ── Top-level loader ─────────────────────────────────────────────────────────

def load_sc_data(
    path: Union[str, Path],
    embedding_keys: Optional[List[str]] = None,
    split_by: Optional[str] = None,
    split_order: Optional[List[str]] = None,
    pca_dims: Tuple[int, int] = (0, 1),
    downsample: Optional[int] = None,
) -> ScDataset:
    """
    Load single-cell data from an h5ad file.

    Parameters
    ----------
    path
        Path to .h5ad file.
    embedding_keys
        List of obsm keys to show as view panels (e.g. ["X_umap", "X_tsne"]).
        Defaults to all recognised 2D embeddings found in the file.
    split_by
        obs column whose unique values each become a separate panel (same
        embedding shown per group — like KaroSpace sections).
        If both ``embedding_keys`` and ``split_by`` are set, one panel is
        created per (embedding_key × group) combination.
    split_order
        Optional explicit ordering for ``split_by`` group values.
    pca_dims
        Which PCA dimensions to use when X_pca is included (0-indexed).
    downsample
        Randomly downsample to at most this many cells (deterministic seed).

    Returns
    -------
    ScDataset
    """
    path = str(path)
    print(f"Loading {path}...")
    adata = _read_h5ad_with_fallback(path)
    print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # Optional downsampling
    if downsample and adata.n_obs > downsample:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(adata.n_obs, size=downsample, replace=False))
        adata = adata[idx].copy()
        print(f"  Downsampled to {adata.n_obs:,} cells")

    # Determine which embedding keys to use
    if embedding_keys is None:
        priority = list(EMBEDDING_DISPLAY_NAMES.keys())
        # Add any remaining obsm keys not in priority list
        extra = [k for k in adata.obsm.keys() if k not in priority]
        candidates = priority + extra
        embedding_keys = [k for k in candidates if k in adata.obsm]
        # Always try to include X_umap and X_tsne if present
        if not embedding_keys:
            embedding_keys = [k for k in adata.obsm.keys()]

    # Filter to keys that actually have 2D data
    valid_keys = []
    for key in embedding_keys:
        result = _extract_2d_embedding(adata, key, dims=(0, 1) if key != "X_pca" else pca_dims)
        if result is not None:
            valid_keys.append(key)
    if not valid_keys:
        raise ValueError(
            f"No valid 2D embeddings found. Checked: {embedding_keys}. "
            f"Available obsm keys: {list(adata.obsm.keys())}"
        )
    print(f"  Using embeddings: {valid_keys}")

    # Build views
    views: List[EmbeddingView] = []

    if split_by is not None:
        if split_by not in adata.obs.columns:
            raise ValueError(f"split_by column '{split_by}' not found in adata.obs")
        groups = adata.obs[split_by].astype(str)
        unique_groups = list(groups.unique())
        if split_order:
            unique_groups = [g for g in split_order if g in set(unique_groups)]
        else:
            unique_groups = sorted(unique_groups)

        for emb_key in valid_keys:
            dims = (0, 1) if emb_key != "X_pca" else pca_dims
            xy = _extract_2d_embedding(adata, emb_key, dims=dims)
            if xy is None:
                continue
            x_all, y_all = xy
            emb_name = _embedding_display_name(emb_key)

            for group in unique_groups:
                mask = groups.values == group
                cell_idx = np.where(mask)[0].astype(np.int32)
                view_id = f"{emb_key}__{group}"
                view_name = f"{emb_name} — {group}" if len(valid_keys) > 1 else group
                views.append(EmbeddingView(
                    id=view_id,
                    name=view_name,
                    embedding_key=emb_key,
                    x=x_all[cell_idx],
                    y=y_all[cell_idx],
                    cell_indices=cell_idx,
                    split_value=group,
                ))
    else:
        for emb_key in valid_keys:
            dims = (0, 1) if emb_key != "X_pca" else pca_dims
            xy = _extract_2d_embedding(adata, emb_key, dims=dims)
            if xy is None:
                continue
            x, y = xy
            views.append(EmbeddingView(
                id=emb_key,
                name=_embedding_display_name(emb_key),
                embedding_key=emb_key,
                x=x,
                y=y,
                cell_indices=None,
            ))

    print(f"  Created {len(views)} view panel(s)")

    obs_columns = _select_obs_columns(adata)
    var_names = list(adata.var_names)

    return ScDataset(
        adata=adata,
        views=views,
        obs_columns=obs_columns,
        var_names=var_names,
    )
