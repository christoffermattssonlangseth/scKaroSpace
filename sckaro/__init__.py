"""
scKaroSpace — embedding-central interactive HTML viewer for scRNA-seq data.

Converts h5ad files into standalone browser-based visualizations,
with 2D embeddings (UMAP, t-SNE, PCA, …) as the primary stage.
"""

from importlib import import_module

__version__ = "0.1.0"
__all__ = [
    "load_sc_data",
    "ScDataset",
    "EmbeddingView",
    "export_to_html",
    "package_sidecar_viewer",
]


def __getattr__(name):
    if name in {"load_sc_data", "ScDataset", "EmbeddingView"}:
        module = import_module(".data_loader", __name__)
        return getattr(module, name)
    if name in {"export_to_html", "package_sidecar_viewer"}:
        module = import_module(".exporter", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
