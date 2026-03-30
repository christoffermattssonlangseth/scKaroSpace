"""
Command-line interface for scKaroSpace.

Usage:
    sckaro input.h5ad -o viewer.html --color leiden --embeddings X_umap X_tsne
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="sckaro",
        description="Generate an embedding-central HTML viewer for scRNA-seq data",
    )
    parser.add_argument("input", type=str, help="Path to input .h5ad file")
    parser.add_argument(
        "-o", "--output",
        type=str, default="sckaro.html",
        help="Output HTML file path (default: sckaro.html)",
    )
    parser.add_argument(
        "-c", "--color",
        type=str, default="leiden",
        help="Initial coloring: obs column or gene name (default: leiden)",
    )
    parser.add_argument(
        "--title",
        type=str, default="scKaroSpace",
        help="Page title (default: scKaroSpace)",
    )
    parser.add_argument(
        "--embeddings",
        type=str, nargs="+", default=None,
        metavar="KEY",
        help="obsm keys to use as embedding panels (e.g. X_umap X_tsne). "
             "Default: all recognised embeddings found in the file.",
    )
    parser.add_argument(
        "--split-by",
        type=str, default=None,
        metavar="COLUMN",
        help="obs column to split into separate panels (one per group value).",
    )
    parser.add_argument(
        "--theme",
        choices=["light", "dark"], default="light",
        help="Colour theme (default: light)",
    )
    parser.add_argument(
        "--spot-size",
        type=float, default=3.0,
        help="Default dot radius in screen pixels (default: 3.0)",
    )
    parser.add_argument(
        "--downsample",
        type=int, default=None,
        metavar="N",
        help="Randomly downsample to N cells before export",
    )
    parser.add_argument(
        "--genes",
        type=str, nargs="+", default=None,
        metavar="GENE",
        help="Additional genes to embed for expression coloring",
    )
    parser.add_argument(
        "--hvg-limit",
        type=int, default=20,
        help="Number of highly variable genes to auto-include (default: 20)",
    )
    parser.add_argument(
        "--gene-sparse-threshold",
        type=float, default=0.8,
        help="Zero-fraction above which sparse gene encoding is used (default: 0.8)",
    )
    parser.add_argument(
        "--marker-genes-groupby",
        type=str, nargs="+", default=None,
        metavar="COLUMN",
        help="obs columns for precomputed marker-gene analysis in the Insights panel",
    )
    parser.add_argument(
        "--cluster-de-groupby",
        type=str, nargs="+", default=None,
        metavar="COLUMN",
        help="obs columns for precomputed pairwise differential expression in the Insights panel",
    )

    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if input_path.suffix.lower() != ".h5ad":
        print(f"Warning: expected .h5ad file, got '{input_path.suffix}'", file=sys.stderr)

    # Import lazily to keep --help fast
    from .data_loader import load_sc_data
    from .exporter import export_to_html

    print(f"Loading data from: {args.input}")
    dataset = load_sc_data(
        args.input,
        embedding_keys=args.embeddings,
        split_by=args.split_by,
        downsample=args.downsample,
    )

    print("Exporting to HTML…")
    out = export_to_html(
        dataset,
        output_path=args.output,
        color=args.color,
        title=args.title,
        theme=args.theme,
        spot_size=args.spot_size,
        genes=args.genes,
        hvg_limit=args.hvg_limit,
        gene_sparse_threshold=args.gene_sparse_threshold,
        marker_genes_groupby=args.marker_genes_groupby,
        cluster_de_groupby=args.cluster_de_groupby,
    )
    print(f"Done! Open {out} in a browser.")


if __name__ == "__main__":
    main()
