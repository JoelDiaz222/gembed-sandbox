from pathlib import Path

import numpy as np
import pandas as pd

# Adapted from plot_utils.py for more concise table headers.
LABEL_MAP = {
    'pg': 'PostgreSQL',
    'pg_local': 'PG Local',
    'internal': 'PG Internal',
    'pg_grpc': 'PG gRPC',
    'ext_direct': 'Py Direct',
    'ext_grpc': 'Ext. gRPC',
    'ext_http': 'Ext. HTTP',
    'chroma': 'ChromaDB',
    'qdrant': 'Qdrant',

    # Benchmark 2
    'pg_local_indexed': 'PG Local Indexed',
    'pg_local_deferred': 'PG Local Deferred',
    'pg_grpc_indexed': 'PG gRPC Indexed',
    'pg_grpc_deferred': 'PG gRPC Deferred',
    'ext_direct_indexed': 'Ext Direct Indexed',
    'ext_direct_deferred': 'Ext Direct Deferred',
    'qd_indexed': 'QD Indexed',
    'qd_deferred': 'QD Deferred',

    # Benchmark 3
    'unified': 'Unified',
    'poly_chroma': 'Poly-Store (PG, ChromaDB)',
    'poly_qdrant': 'Dist. Qdrant',
    'poly_qdrant_deferred': 'Poly-Store (PG, QD Deferred)',
    'mono_pg_unified_deferred': 'Mono-Store (PG Local)',
    'mono_pg_direct_deferred': 'Mono-Store (Direct)',
    'mono_ext_direct_deferred': 'Mono-Store (Ext Direct)',

    # Benchmark 4
    'pg_indexed': 'PG Indexed',
    'pg_deferred': 'PG Deferred',
    'PostgreSQL': 'PostgreSQL',
    'ChromaDB': 'ChromaDB',
    'Qdrant': 'Qdrant',

    # Benchmark 6 backends
    'embed_anything': 'EmbedAnything (Candle, CUDA)',
    'ort': 'ONNX Runtime (CPU)',
}


def generate_tables_from_csv(csv_file: str, output_dir: str, timestamp: str, metrics: list = ['throughput'],
                             baseline_candidates: list = None):
    """
    Generate LaTeX tables (standard metrics and speedups) from a benchmark CSV file in a single pass.
    """
    if baseline_candidates is None:
        baseline_candidates = ['ext_direct', 'pg_direct', 'ext_direct_indexed', 'mono_pg_direct_deferred']

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file}")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Dynamically find method labels from CSV columns, preserving order
    methods = []
    for col in df.columns:
        if col.endswith('_time_s'):
            method_name = col.replace('_time_s', '')
            if method_name not in methods:
                methods.append(method_name)

    if not methods:
        print(f"No methods found in CSV: {csv_file}")
        # Check whether this is a benchmark-8 overhead CSV despite having no
        # standard methods — if so, delegate to the overhead table generator.
        if _is_overhead_csv(df):
            return generate_tables_b8(df, output_path, timestamp)
        return

    # Place ChromaDB before Qdrant
    if 'chroma' in methods and 'qd_indexed' in methods:
        methods.remove('chroma')
        methods.insert(methods.index('qd_indexed'), 'chroma')

    sizes = sorted(df['size'].unique())

    b6_backends = {'embed_anything', 'ort'}

    def _method_label(m: str) -> str:
        """Human-readable column header, with B6-aware fallback."""
        if m in LABEL_MAP:
            return LABEL_MAP[m]
        for b in b6_backends:
            if m.startswith(f"{b}_"):
                model_short = m[len(b) + 1:]
                backend_lbl = LABEL_MAP.get(b, b.replace('_', ' ').title())
                return f"{backend_lbl} / {model_short}"
        return m.replace('_', ' ').title()

    baseline_method = next(
        (c for c in baseline_candidates if c in methods),
        next((m for m in methods if m.startswith('ort_')), None)
    )

    for metric in metrics:
        table_data = []
        speedup_data = []

        for size in sizes:
            row = {'Input Size': size}
            speedup_row = {'Input Size': size}
            size_df = df[df['size'] == size]

            # Pre-calculate baseline mean if this is throughput
            baseline_mean = None
            if metric == 'throughput' and baseline_method:
                metric_col = f'{baseline_method}_{metric}'
                if metric_col in size_df.columns:
                    b_vals = size_df[metric_col].dropna()
                    if not b_vals.empty:
                        baseline_mean = np.mean(b_vals)

            for method in methods:
                metric_col = f'{method}_{metric}'
                if metric_col in size_df.columns:
                    values = size_df[metric_col].dropna()
                    if not values.empty:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        # Format to one decimal place for consistency
                        row[method] = f"{mean_val:.1f} ({std_val:.1f})"

                        if metric == 'throughput' and baseline_mean and baseline_mean > 0:
                            speedup_row[method] = mean_val / baseline_mean
                        else:
                            speedup_row[method] = None
                    else:
                        row[method] = "-"
                        speedup_row[method] = None
                else:
                    row[method] = "-"
                    speedup_row[method] = None

            table_data.append(row)
            if metric == 'throughput' and baseline_method:
                speedup_data.append(speedup_row)

        if not table_data:
            print(f"No data to generate table for metric '{metric}' in {csv_file}")
            continue

        # --- Generate Standard Table ---
        # Bold the best value in each row (highest mean for throughput, lowest for time_s)
        for row_data in table_data:
            max_mean = -1.0 if metric == 'throughput' else float('inf')
            max_method = None
            for method in methods:
                val = row_data.get(method)
                if val and val != "-":
                    try:
                        mean_val = float(val.split(' ')[0])
                        if (metric == 'throughput' and mean_val > max_mean) or \
                                (metric != 'throughput' and mean_val < max_mean):
                            max_mean = mean_val
                            max_method = method
                    except (ValueError, IndexError):
                        continue  # Ignore malformed entries

            if max_method:
                row_data[max_method] = f"\\textbf{{{row_data[max_method]}}}"

        # Create LaTeX table string
        header_labels = [_method_label(m) for m in methods]

        latex_string = f"% Auto-generated from {Path(csv_file).name}\n"
        latex_string += "\\begin{tabular}{l" + "c" * len(methods) + "}\n"
        latex_string += "\\toprule\n"
        latex_string += "\\textbf{Input Size} & " + " & ".join([f"\\textbf{{{l}}}" for l in header_labels]) + " \\\\\n"
        latex_string += "\\midrule\n"

        for row_data in table_data:
            row_values = [str(row_data['Input Size'])] + [row_data.get(method, "-") for method in methods]
            latex_string += " & ".join(row_values) + " \\\\\n"

        latex_string += "\\bottomrule\n"
        latex_string += "\\end{tabular}\n"

        table_filename = f"{metric}_table_{timestamp}.tex"
        table_filepath = output_path / table_filename
        with open(table_filepath, 'w') as f:
            f.write(latex_string)

        print(f"LaTeX table saved to {table_filepath}")

        # --- Generate Speedup Table ---
        if metric == 'throughput' and baseline_method and speedup_data:
            display_methods = [m for m in methods if m != baseline_method]
            sp_header_labels = [_method_label(m) for m in display_methods]

            sp_latex = f"% Speedup relative to {LABEL_MAP.get(baseline_method, baseline_method)} \n"
            sp_latex += "\\begin{tabular}{l" + "c" * len(display_methods) + "}\n"
            sp_latex += "\\toprule\n"
            sp_latex += "\\textbf{Input Size} & " + " & ".join(
                [f"\\textbf{{{l}}}" for l in sp_header_labels]) + " \\\\\n"
            sp_latex += "\\midrule\n"

            for row_data in speedup_data:
                max_speedup_candidates = [row_data.get(m, 0) or 0 for m in display_methods]
                max_speedup = max(max_speedup_candidates) if max_speedup_candidates else 0

                row_values = [str(row_data['Input Size'])]
                for method in display_methods:
                    val = row_data.get(method)
                    if val is not None:
                        formatted_val = f"{val:.1f}x"
                        if val == max_speedup and val > 1.01:
                            row_values.append(f"\\textbf{{{formatted_val}}}")
                        else:
                            row_values.append(formatted_val)
                    else:
                        row_values.append("-")

                sp_latex += " & ".join(row_values) + " \\\\\n"

            sp_latex += "\\bottomrule\n"
            sp_latex += "\\end{tabular}\n"

            sp_filename = f"speedup_table_{timestamp}.tex"
            sp_filepath = output_path / sp_filename
            with open(sp_filepath, 'w') as f:
                f.write(sp_latex)

            print(f"Speedup table saved to {sp_filepath}")


# ---------------------------------------------------------------------------
# Benchmark 8: Measure Gembed overhead
# ---------------------------------------------------------------------------

def _is_overhead_csv(df) -> bool:
    """Return True when the CSV has the benchmark-8 overhead column schema."""
    return 'wall_time_us' in df.columns and 'ffi_roundtrip_us' in df.columns


def generate_tables_b8(df, output_path: Path, timestamp: str):
    """
    Generate a booktabs LaTeX table for Benchmark 8 (Gembed overhead).

    Rows: overhead component / total wall time / stack overhead %.
    Columns: batch sizes.
    All values are mean ± std over all rows for that size.
    """
    sizes = sorted(df['size'].unique().tolist())

    components = [
        ('validate_backend_us',  r'\texttt{validate\_backend()}'),
        ('validate_model_us',    r'\texttt{validate\_model()}'),
        ('pre_ffi_overhead_us',  r'Pre-FFI C overhead'),
        ('ffi_roundtrip_us',     r'FFI roundtrip (C\,$\leftrightarrow$\,Rust)'),
        ('rs_dispatch_us',       r'Rust backend dispatch'),
        ('pure_embedding_us',    r'EmbedAnything inference'),
        ('rs_to_c_return_us',    r'Rust\,$\to$\,C return'),
        ('post_ffi_overhead_us', r'Post-FFI C overhead'),
    ]

    # Aggregate per size
    agg = {}  # size -> {col: (mean, std)}
    all_cols = [c for c, _ in components] + ['wall_time_us']
    for sz in sizes:
        sub = df[df['size'] == sz]
        agg[sz] = {}
        for col in all_cols:
            vals = sub[col].dropna().values if col in sub.columns else np.array([])
            agg[sz][col] = (
                float(np.mean(vals)) if len(vals) else 0.0,
                float(np.std(vals))  if len(vals) else 0.0,
            )

    col_spec = 'l' + 'r' * len(sizes)
    size_headers = ' & '.join(f'$n={s}$' for s in sizes)

    lines = [
        r'\begin{table}[t]',
        r'  \centering',
        r'  \small',
        r'  \caption{Gembed stack overhead breakdown (mean\,$\pm$\,std, in \textmu{}s)'
        r'           across batch sizes. Backend: \texttt{embed\_anything},'
        r'           model: \texttt{MiniLM\mbox{-}L6\mbox{-}v2} (warm model cache).}',
        r'  \label{tab:gembed_overhead}',
        f'  \\begin{{tabular}}{{{col_spec}}}',
        r'    \toprule',
        f'    \\textbf{{Component}} & {size_headers} \\\\',
        r'    \midrule',
    ]

    for col, label in components:
        parts = []
        for sz in sizes:
            mean_us, std_us = agg[sz].get(col, (0.0, 0.0))
            parts.append(f'${mean_us:.0f}\\pm{std_us:.0f}$')
        lines.append(f'    {label} & {" & ".join(parts)} \\\\')

    # Wall-time row
    lines.append(r'    \midrule')
    wall_parts = []
    for sz in sizes:
        m, s = agg[sz].get('wall_time_us', (0.0, 0.0))
        wall_parts.append(f'${m:.0f}\\pm{s:.0f}$')
    lines.append(
        r'    \textbf{Wall time (Python)} & '
        + ' & '.join(wall_parts)
        + r' \\'
    )

    # Stack overhead % row
    lines.append(r'    \midrule')
    pct_parts = []
    for sz in sizes:
        wall_mean, _ = agg[sz].get('wall_time_us', (0.0, 0.0))
        emb_mean, emb_std = agg[sz].get('pure_embedding_us', (0.0, 0.0))
        if wall_mean > 0:
            pct = 100.0 * (wall_mean - emb_mean) / wall_mean
            pct_std = 100.0 * emb_std / wall_mean
            pct_parts.append(f'${pct:.1f}\\%\\pm{pct_std:.1f}\\%$')
        else:
            pct_parts.append('--')
    lines.append(
        r'    \textbf{Stack overhead} & '
        + ' & '.join(pct_parts)
        + r' \\'
    )

    lines += [
        r'    \bottomrule',
        r'  \end{tabular}',
        r'\end{table}',
    ]

    tex_path = output_path / f'overhead_table_{timestamp}.tex'
    tex_path.write_text('\n'.join(lines) + '\n')
    print(f"LaTeX overhead table saved to {tex_path}")

