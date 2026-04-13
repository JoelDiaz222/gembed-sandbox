import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

COLOR_PG_MAIN = '#003f5c'  # Navy Blue (Internal/Mono-Store)
COLOR_PG_ALT = '#444e86'  # Slate (Indexed/Optimized)
COLOR_VECTOR_QD = '#ff1f5b'  # Crimson (Qdrant)
COLOR_VECTOR_CH = '#ffa600'  # Amber (Chroma)
COLOR_DIRECT = '#00af91'  # Teal (In-Process Local)
COLOR_REMOTE_GRPC = '#845ec2'  # Light Indigo (gRPC Remote)
COLOR_REMOTE_HTTP = '#d65db1'  # Light Magenta (HTTP Remote)
COLOR_B6_EA = '#003f5c'  # Navy  – EmbedAnything (Candle + CUDA)
COLOR_B6_ORT = '#ffa600'  # Amber – ONNX Runtime (CPU)
COLOR_B7_MYSQL = '#00af91'  # Teal  – MySQL (mysql_gembed)
COLOR_B7_REDIS = '#ef5675'  # Coral – Redis (redis_gembed)

STYLE_MAP = {
    # PostgreSQL / Internal Group
    'pg_local': (COLOR_PG_MAIN, '-', 'o'),
    'pg_mono_store': (COLOR_PG_MAIN, '-', 'o'),
    'mono_store': (COLOR_PG_MAIN, '-', 'o'),
    'internal': (COLOR_PG_MAIN, '-', 'o'),
    'pg': (COLOR_PG_MAIN, '-', 'o'),
    'mysql': (COLOR_B7_MYSQL, '-', 's'),
    'redis': (COLOR_B7_REDIS, '-', 'D'),

    # PG Variants (Indexed/Deferred/gRPC)
    'pg_indexed': (COLOR_PG_ALT, '-', 's'),
    'pg_deferred': (COLOR_PG_ALT, '--', 's'),
    'pg_local_indexed': (COLOR_PG_ALT, '-', 's'),
    'pg_local_deferred': (COLOR_PG_ALT, '--', 's'),
    'pg_mono_indexed': (COLOR_PG_ALT, '-', 's'),
    'pg_mono_deferred': (COLOR_PG_ALT, '--', 's'),
    'pg_grpc_indexed': (COLOR_PG_ALT, '-', 'D'),
    'pg_grpc_deferred': (COLOR_PG_ALT, '--', 'D'),
    'pg_grpc': (COLOR_PG_ALT, ':', 'D'),
    'pg_http': (COLOR_PG_ALT, ':', 'X'),

    # Vector Databases
    'qdrant': (COLOR_VECTOR_QD, '-.', 'D'),
    'poly_qdrant': (COLOR_VECTOR_QD, '-.', 'D'),
    'qd_indexed': (COLOR_VECTOR_QD, '-.', 'D'),
    'qd_deferred': (COLOR_VECTOR_QD, ':', 'v'),
    'chroma': (COLOR_VECTOR_CH, '-.', '^'),
    'poly_chroma': (COLOR_VECTOR_CH, '-.', '^'),

    # Application Clients (Local/In-Process)
    'ext_direct': (COLOR_DIRECT, '--', '*'),
    'external': (COLOR_DIRECT, '--', '*'),
    'ext_direct_indexed': (COLOR_DIRECT, '-', '*'),
    'ext_direct_deferred': (COLOR_DIRECT, '--', '*'),
    'mono_pg_direct_deferred': (COLOR_DIRECT, '--', 'o'),
    'mono_ext_direct_deferred': (COLOR_DIRECT, '--', '*'),
    'mono_pg_unified_deferred': (COLOR_PG_ALT, '--', 's'),
    'poly_qdrant_deferred': (COLOR_VECTOR_QD, ':', 'D'),
    'ext_grpc': (COLOR_REMOTE_GRPC, ':', 'P'),
    'ext_http': (COLOR_REMOTE_HTTP, ':', 'X'),
}

LABEL_MAP = {
    'pg': 'PostgreSQL',
    'pg_local': 'PG Local',
    'pg_mono_store': 'Mono-Store (PG, Local)',
    'mono_store': 'Mono-Store (PG, Local)',
    'internal': 'PG Local',
    'pg_grpc': 'PG gRPC',
    'pg_http': 'PG HTTP',
    'pg_indexed': 'PG Local Immediate',
    'pg_deferred': 'PG Local Deferred',
    'pg_local_indexed': 'PG Local Immediate',
    'pg_local_deferred': 'PG Local Deferred',
    'pg_mono_indexed': 'Mono-Store (PG, Local, Immediate Index)',
    'pg_mono_deferred': 'Mono-Store (PG, Local, Deferred Index)',
    'pg_grpc_indexed': 'PG (gRPC, Immediate Index)',
    'pg_grpc_deferred': 'PG (gRPC, Deferred Index)',
    'ext_direct': 'App. Local',
    'external': 'PG External Client',
    'ext_direct_indexed': 'App. Local Immediate',
    'ext_direct_deferred': 'App. Local Deferred',
    'pg_unified': 'PG Local',
    'pg_direct': 'App. Local',
    'pg_gembed_unified': 'PG Local',
    'ext_grpc': 'App. gRPC',
    'ext_http': 'App. HTTP',
    'chroma': 'ChromaDB',
    'qdrant': 'Qdrant',
    'poly_chroma': 'Poly-Store (with ChromaDB)',
    'poly_qdrant': 'Poly-Store (with Qdrant)',
    'two_step_chroma': 'Poly-Store (with ChromaDB)',
    'two_step_qdrant': 'Poly-Store (with Qdrant)',
    'qd_indexed': 'Qdrant Immediate',
    'qd_deferred': 'Qdrant Deferred',
    'mono_pg_unified_deferred': 'Mono-Store (PG Local Deferred)',
    'mono_pg_direct_deferred': 'Mono-Store (App. Local Deferred)',
    'mono_ext_direct_deferred': 'Mono-Store (App. Local Deferred)',
    'poly_qdrant_deferred': 'Poly-Store (with Qdrant Deferred)',
    'embed_anything': 'EmbedAnything (Candle, CUDA)',
    'ort': 'ONNX Runtime (CPU)',
    # Benchmark 7 adapters
    'pg': 'PostgreSQL (pg\\_gembed)',
    'mysql': 'MySQL (mysql\\_gembed)',
    'redis': 'Redis (redis\\_gembed)',
}


def get_style(method_name: str):
    """Return (color, linestyle, marker, label) for a given method."""
    import re
    # Check if a dynamic modifier was appended (e.g., pg_gembed_unified_f0.05)
    m = re.match(r'(.*)_f([0-9.]+)$', method_name)
    if m:
        base_method = m.group(1)
        frac = m.group(2)
        style = STYLE_MAP.get(base_method, ('#333333', '-', 'x'))
        label = LABEL_MAP.get(base_method, base_method) + f" (α={frac})"
        return style[0], style[1], style[2], label

    style = STYLE_MAP.get(method_name, ('#333333', '-', 'x'))
    label = LABEL_MAP.get(method_name, method_name)
    return style[0], style[1], style[2], label


def configure_latex_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 16,
        "font.size": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 2.0,
        "figure.autolayout": True,
        'text.usetex': True
    })


def save_results_csv(all_results: List[dict], output_dir: Path, timestamp: str, methods: List[str]):
    """Save aggregated results with statistics (used after all runs are concatenated)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"benchmark_{timestamp}.csv"

    # Data-driven component detection
    has_pg = any(m in r and 'pg_cpu' in r[m] for r in all_results for m in methods)
    has_qd = any(m in r and 'qd_cpu' in r[m] for r in all_results for m in methods)

    metrics = ['throughput', 'time_s', 'py_cpu', 'py_mem_delta', 'py_mem_peak']
    if has_pg:
        metrics += ['pg_cpu', 'pg_mem_delta', 'pg_mem_peak']
    if has_qd:
        metrics += ['qd_cpu', 'qd_mem_delta', 'qd_mem_peak']
    metrics += ['sys_cpu', 'sys_mem']

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['size']
        for method in methods:
            for metric in metrics:
                header.append(f"{method}_{metric}")
                header.append(f"{method}_{metric}_std")
                header.append(f"{method}_{metric}_median")
                header.append(f"{method}_{metric}_q1")
                header.append(f"{method}_{metric}_q3")
        writer.writerow(header)

        for r in all_results:
            row = [r['size']]
            for method in methods:
                if method in r:
                    for metric in metrics:
                        row.append(r[method].get(metric, ''))
                        row.append(r[method].get(f"{metric}_std", ''))
                        row.append(r[method].get(f"{metric}_median", ''))
                        row.append(r[method].get(f"{metric}_q1", ''))
                        row.append(r[method].get(f"{metric}_q3", ''))
                else:
                    row.extend([''] * (len(metrics) * 5))
            writer.writerow(row)
    print(f"\nResults saved to {path}")
    return str(path)


def save_single_run_csv(all_results: List[dict], output_dir: Path, run_id: str, methods: List[str]):
    """Save a single run's raw results (without aggregation) for later concatenation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"benchmark_{run_id}_run.csv"

    # Data-driven component detection
    has_pg = any(m in r and 'pg_cpu' in r[m] for r in all_results for m in methods)
    has_qd = any(m in r and 'qd_cpu' in r[m] for r in all_results for m in methods)

    metrics = ['throughput', 'time_s', 'py_cpu', 'py_mem_delta', 'py_mem_peak']
    if has_pg:
        metrics += ['pg_cpu', 'pg_mem_delta', 'pg_mem_peak']
    if has_qd:
        metrics += ['qd_cpu', 'qd_mem_delta', 'qd_mem_peak']
    metrics += ['sys_cpu', 'sys_mem']

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['size']
        for method in methods:
            for metric in metrics:
                header.append(f"{method}_{metric}")
        writer.writerow(header)

        for r in all_results:
            row = [r['size']]
            for method in methods:
                if method in r:
                    for metric in metrics:
                        row.append(r[method].get(metric, ''))
                else:
                    row.extend([''] * len(metrics))
            writer.writerow(row)

    print(f"CSV saved to: {path}")
    return str(path)


def generate_plots(all_results: List[dict], output_dir: Path, timestamp: str, methods: List[str],
                   throughput_unit: str = 'rows/s'):
    configure_latex_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = [r['size'] for r in all_results]

    # Data-driven component detection
    has_pg_data = any(m in r and r[m].get('pg_mem_peak', 0) > 0 for r in all_results for m in methods)
    has_qd_data = any(m in r and r[m].get('qd_mem_peak', 0) > 0 for r in all_results for m in methods)

    def plot_single(key, y_label, metric_type, filename_suffix):
        fig, ax = plt.subplots(figsize=(7, 5))

        for method in methods:
            color, ls, marker, label = get_style(method)
            y_vals = [r[method].get(key, 0) for r in all_results if method in r]
            std_vals = [r[method].get(f'{key}_std', 0) for r in all_results if method in r]
            y_errs = std_vals
            if not any(y_vals): continue
            ax.errorbar(sizes, y_vals, yerr=y_errs, fmt=marker, linestyle=ls, color=color, label=label,
                        linewidth=2.0, capsize=3, markersize=6, alpha=0.9)

        ax.set_xlabel('Input Size (Log Scale)')
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xscale('log', base=2)

        ax.legend(loc='best', frameon=True, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(output_dir / f"{filename_suffix}_{metric_type}_{timestamp}.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(output_dir / f"{filename_suffix}_{metric_type}_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_throughput():
        plt.figure(figsize=(7, 5))
        for method in methods:
            color, ls, marker, label = get_style(method)
            y_vals = [r[method].get('throughput', 0) for r in all_results if
                      method in r]
            std_vals = [r[method].get('throughput_std', 0) for r in all_results if
                        method in r]
            # Symmetric error bars tracking std
            y_errs = std_vals
            if not any(y_vals): continue
            plt.errorbar(sizes, y_vals, yerr=y_errs, fmt=marker, linestyle=ls, color=color, label=label,
                         linewidth=1.5, capsize=3, markersize=5, alpha=0.9)

        plt.xlabel('Input Size (Log Scale)')
        plt.ylabel(f'Throughput ({throughput_unit})')

        plt.legend(loc='best', frameon=True, framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xscale('log', base=2)
        plt.savefig(output_dir / f"throughput_{timestamp}.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(output_dir / f"throughput_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_normalized_throughput():
        baseline_candidates = ['ext_direct', 'pg_direct', 'ext_direct_indexed', 'mono_pg_direct_deferred',
                               'mono_ext_direct_deferred', 'pg']
        baseline_method = None
        for candidate in baseline_candidates:
            if candidate in methods:
                baseline_method = candidate
                break

        if not baseline_method:
            return

        plt.figure(figsize=(8, 6))

        baseline_y_vals = [r[baseline_method].get('throughput', 0) for r in
                           all_results if baseline_method in r]
        if not any(baseline_y_vals):
            plt.close()
            return

        for method in methods:
            color, ls, marker, label = get_style(method)
            y_vals = [r[method].get('throughput', 0) for r in all_results if
                      method in r]
            std_vals = [r[method].get('throughput_std', 0) for r in all_results if
                        method in r]

            if not any(y_vals): continue

            norm_y_vals, norm_std_vals, valid_sizes = [], [], []
            for sz, y, std, b in zip(sizes, y_vals, std_vals, baseline_y_vals):
                if y and b and b > 0:
                    norm_y_vals.append(y / b)
                    norm_std_vals.append(std / b)
                    valid_sizes.append(sz)

            if not norm_y_vals: continue

            y_errs = norm_std_vals

            plt.errorbar(valid_sizes, norm_y_vals, yerr=y_errs, fmt=marker, linestyle=ls, color=color, label=label,
                         linewidth=1.5, capsize=3, markersize=5, alpha=0.9)

        plt.xscale('log', base=2)
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        plt.xlabel('Input Size (Log Scale)')

        if baseline_method in ['mono_pg_direct_deferred', 'mono_ext_direct_deferred']:
            y_axis_baseline_name = 'App. Local Deferred'
        else:
            y_axis_baseline_name = LABEL_MAP.get(baseline_method, baseline_method)

        plt.ylabel(f'Relative Throughput (vs {y_axis_baseline_name})')

        plt.legend(loc='best', frameon=True, framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.3)

        plt.savefig(output_dir / f"normalized_throughput_{timestamp}.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(output_dir / f"normalized_throughput_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

    plot_throughput()
    plot_normalized_throughput()
    plot_single('py_cpu', r'CPU (\%)', 'cpu', 'python_resources')
    plot_single('py_mem_peak', 'Memory (MB)', 'memory', 'python_resources')
    if has_pg_data:
        plot_single('pg_cpu', r'CPU (\%)', 'cpu', 'postgres_resources')
        plot_single('pg_mem_peak', 'Memory (MB)', 'memory', 'postgres_resources')
    if has_qd_data:
        plot_single('qd_cpu', r'CPU (\%)', 'cpu', 'qdrant_resources')
        plot_single('qd_mem_peak', 'Memory (MB)', 'memory', 'qdrant_resources')
    plot_single('sys_cpu', r'CPU (\%)', 'cpu', 'system_resources')
    plot_single('sys_mem', 'Memory (MB)', 'memory', 'system_resources')
    print(f"Plots saved to {output_dir} (PDF + PNG)")


def _is_extensibility_methods(methods: List[str]) -> bool:
    """Return True when every method follows the <backend>_<model_name> pattern
    used by Benchmark 6 (e.g. 'embed_anything_openai/clip-vit-base-patch32')."""
    b6_backends = {'embed_anything', 'ort', 'candle'}
    return bool(methods) and all(
        any(m.startswith(f"{b}_") for b in b6_backends) for m in methods
    )


def _is_portability_methods(methods: List[str]) -> bool:
    """Return True when every method follows the <adapter>_<model_name> pattern
    used by Benchmark 7 (e.g. 'pg_sentence-transformers/all-MiniLM-L6-v2')."""
    b7_adapters = {'pg', 'mysql', 'redis'}
    return bool(methods) and all(
        any(m.startswith(f"{a}_") for a in b7_adapters) for m in methods
    ) and not _is_extensibility_methods(methods)


def generate_plots_b6(
        all_results: List[dict],
        output_dir: Path,
        timestamp: str,
        methods: List[str],
):
    """Grouped bar chart for Benchmark 6 (image embedding backend comparison).

    Layout: one group of bars per model, one bar per backend within each group.
    Only a single size is expected (the benchmark is run at one fixed scale).
    IQR-based asymmetric error bars are drawn when available (multi-run CSV),
    falling back to the raw value when only a single run is present.
    """
    configure_latex_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- parse methods into (backend, model_name) pairs --------------------
    b6_backends = ['embed_anything', 'ort', 'candle']
    backend_color = {
        'embed_anything': COLOR_B6_EA,
        'ort': COLOR_B6_ORT,
        'candle': COLOR_PG_ALT,
    }
    backend_label = {
        'embed_anything': LABEL_MAP.get('embed_anything', 'EmbedAnything (Candle, CUDA)'),
        'ort': LABEL_MAP.get('ort', 'ONNX Runtime (CPU)'),
        'candle': 'Candle (CPU)',
    }
    backend_hatch = {
        'embed_anything': '',
        'ort': '//',
        'candle': 'xx',
    }

    # Preserve the order in which backends/models first appear
    seen_backends = []
    seen_models = []
    method_map: dict[str, tuple[str, str]] = {}  # method_name -> (backend, model_name)
    for m in methods:
        for b in b6_backends:
            if m.startswith(f"{b}_"):
                model_name = m[len(b) + 1:]
                method_map[m] = (b, model_name)
                if b not in seen_backends:
                    seen_backends.append(b)
                if model_name not in seen_models:
                    seen_models.append(model_name)
                break

    n_models = len(seen_models)
    n_backends = len(seen_backends)
    if n_models == 0 or n_backends == 0:
        return

    # Aggregate across runs (or just take the single value if only one run)
    # all_results has one entry per size; we expect a single size here.
    aggregated: dict[str, dict] = {}  # method -> metric dict (with _median, _q1, _q3)
    for r in all_results:
        for m in methods:
            if m not in r:
                continue
            existing = aggregated.get(m, {})
            # Merge: prefer _median keys (set by generate_plots_from_csv aggregation),
            # fall back to the raw scalar stored by the benchmark's own all_results list.
            for key, val in r[m].items():
                if key not in existing:
                    existing[key] = val
            aggregated[m] = existing

    def _get(m, metric):
        d = aggregated.get(m, {})
        return (
            d.get(metric, 0) or 0,
            d.get(f'{metric}_std', 0) or 0,
        )

    # Throughput grouped bar chart
    bar_width = 0.8 / n_backends
    x = range(n_models)

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * n_models), 4.5))

    for bi, backend in enumerate(seen_backends):
        offsets = [xi + (bi - (n_backends - 1) / 2) * bar_width for xi in x]
        heights, err_vals = [], []
        for model_name in seen_models:
            method_name = f"{backend}_{model_name}"
            mean_val, std_val = _get(method_name, 'throughput')
            heights.append(mean_val)
            err_vals.append(std_val)

        ax.bar(
            offsets, heights,
            width=bar_width,
            label=backend_label[backend],
            color=backend_color[backend],
            hatch=backend_hatch[backend],
            edgecolor='white',
            linewidth=0.5,
            yerr=err_vals,
            capsize=3,
            error_kw={'linewidth': 1.0, 'ecolor': '#555555'},
            zorder=3,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(seen_models, rotation=15, ha='center')
    ax.set_xlabel('Model')
    ax.set_ylabel(r'Throughput (img/s)')

    ax.legend(loc='best', frameon=True, framealpha=0.9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_dir / f"throughput_{timestamp}.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(output_dir / f"throughput_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Speedup bar chart  (embed_anything / ort, per model)
    if 'embed_anything' in seen_backends and 'ort' in seen_backends:
        fig, ax = plt.subplots(figsize=(max(5, 2.0 * n_models), 4.0))
        speedups, err_sp_vals = [], []
        for model_name in seen_models:
            ea_mean, ea_std = _get(f'embed_anything_{model_name}', 'throughput')
            ort_mean, ort_std = _get(f'ort_{model_name}', 'throughput')
            sp = (ea_mean / ort_mean) if ort_mean > 0 else 0.0
            # Simple error propagation for division
            sp_rel_err = (ea_std / ea_mean if ea_mean > 0 else 0.0) + (ort_std / ort_mean if ort_mean > 0 else 0.0)
            sp_std = sp * sp_rel_err
            speedups.append(sp)
            err_sp_vals.append(sp_std)

        ax.bar(
            list(x), speedups,
            color=COLOR_B6_EA,
            edgecolor='white',
            linewidth=0.5,
            yerr=err_sp_vals,
            capsize=3,
            error_kw={'linewidth': 1.0, 'ecolor': '#555555'},
            zorder=3,
        )
        ax.axhline(y=1.0, color='#aaaaaa', linestyle='--', linewidth=1.0, zorder=2)
        ax.set_xticks(list(x))
        ax.set_xticklabels(seen_models, rotation=15, ha='right')
        ax.set_xlabel('Model')
        ax.set_ylabel(r'Speedup vs.\ CPU')

        ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(output_dir / f"speedup_{timestamp}.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(output_dir / f"speedup_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Resource plots: CPU + memory, one dual chart per component.
    # Same grouped-bar layout as the throughput chart.
    def plot_resource_single(key, y_label, metric_type, filename_suffix):
        fig, ax = plt.subplots(figsize=(max(5, 2.0 * n_models), 4.5))

        for bi, backend in enumerate(seen_backends):
            offsets = [xi + (bi - (n_backends - 1) / 2) * bar_width for xi in x]
            heights, err_vals = [], []
            for model_name in seen_models:
                method_name = f"{backend}_{model_name}"
                mean_val, std_val = _get(method_name, key)
                heights.append(mean_val)
                err_vals.append(std_val)

            bar_kwargs = dict(
                width=bar_width,
                label=backend_label[backend],
                color=backend_color[backend],
                hatch=backend_hatch[backend],
                edgecolor='white',
                linewidth=0.5,
                capsize=3,
                error_kw={'linewidth': 1.0, 'ecolor': '#555555'},
                zorder=3,
            )
            ax.bar(offsets, heights, yerr=err_vals, **bar_kwargs)

        ax.set_xticks(list(x))
        ax.set_xticklabels(seen_models, rotation=15, ha='right')
        ax.set_xlabel('Model')
        ax.set_ylabel(y_label)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

        ax.legend(loc='best', frameon=True, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(output_dir / f"{filename_suffix}_{metric_type}_{timestamp}.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(output_dir / f"{filename_suffix}_{metric_type}_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

    plot_resource_single('py_cpu', r'CPU (\%)', 'cpu', 'python_resources')
    plot_resource_single('py_mem_peak', 'Memory (MB)', 'memory', 'python_resources')
    plot_resource_single('pg_cpu', r'CPU (\%)', 'cpu', 'postgres_resources')
    plot_resource_single('pg_mem_peak', 'Memory (MB)', 'memory', 'postgres_resources')
    plot_resource_single('sys_cpu', r'CPU (\%)', 'cpu', 'system_resources')
    plot_resource_single('sys_mem', 'Memory (MB)', 'memory', 'system_resources')

    print(f"Benchmark 6 plots saved to {output_dir} (PDF + PNG)")


def generate_plots_b7(
        all_results: List[dict],
        output_dir: Path,
        timestamp: str,
        methods: List[str],
):
    """
    Generate plots for Benchmark 7 (portability).
    For each model tested, extracts the adapter-specific results and generates
    line plots across batch sizes (including normalized throughput), storing
    them in a subdirectory per model.
    """
    b7_adapters = ['pg', 'mysql', 'redis']

    # Find all unique models and adapters
    seen_models = []
    seen_adapters = []
    for m in methods:
        for a in b7_adapters:
            if m.startswith(f"{a}_"):
                model_name = m[len(a) + 1:]
                if model_name not in seen_models:
                    seen_models.append(model_name)
                if a not in seen_adapters:
                    seen_adapters.append(a)
                break

    if not seen_models or not seen_adapters:
        return

    for model_name in seen_models:
        model_methods = []
        for a in seen_adapters:
            model_methods.append(a)

        # Build model_results for this specific model
        model_results = []
        for r in all_results:
            model_entry = {'size': r['size']}
            has_data = False
            for a in seen_adapters:
                full_method = f"{a}_{model_name}"
                if full_method in r:
                    model_entry[a] = r[full_method]
                    has_data = True
            if has_data:
                model_results.append(model_entry)

        if not model_results:
            continue

        safe_model_name = model_name.replace('/', '--')
        model_dir = output_dir / safe_model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        generate_plots(model_results, model_dir, timestamp, model_methods, 'texts/s')

    print(f"Benchmark 7 per-model plots saved in subdirectories of {output_dir}")


def generate_plots_from_csv(csv_file: str, output_dir: str, timestamp: str):
    """Generate plots from a concatenated CSV file.
    
    This function reads a CSV with multiple runs, aggregates the data,
    and generates plots.
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path

    df = pd.read_csv(csv_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if _is_overhead_csv(df):
        generate_plots_b8(df, output_dir, timestamp)
        print(f"Overhead plots generated from CSV: {csv_file}")
        return

    # Extract method names from columns
    methods = []
    for col in df.columns:
        if col.endswith('_time_s'):
            method = col.replace('_time_s', '')
            methods.append(method)

    if not methods:
        print(f"No methods found in CSV: {csv_file}")
        return

    # Place ChromaDB before Qdrant
    if 'chroma' in methods and 'qd_indexed' in methods:
        methods.remove('chroma')
        methods.insert(methods.index('qd_indexed'), 'chroma')

    # Get unique sizes
    sizes = sorted(df['size'].unique())

    # Aggregate data for each size and method
    all_results = []
    for size in sizes:
        size_data = df[df['size'] == size]
        result_entry = {'size': size}

        for method in methods:
            method_data = {}

            # Collect all metrics for this method
            for col in df.columns:
                if col.startswith(f'{method}_') and not col.endswith(('_std', '_median', '_q1', '_q3')):
                    metric_name = col.replace(f'{method}_', '')
                    values = size_data[col].dropna().values

                    if len(values) > 0:
                        method_data[metric_name] = np.mean(values)
                        method_data[f'{metric_name}_std'] = np.std(values)
                        method_data[f'{metric_name}_median'] = np.median(values)
                        method_data[f'{metric_name}_q1'] = np.percentile(values, 25)
                        method_data[f'{metric_name}_q3'] = np.percentile(values, 75)

            result_entry[method] = method_data

        all_results.append(result_entry)

    # Route to the appropriate plot function
    if _is_portability_methods(methods):
        generate_plots_b7(all_results, output_dir, timestamp, methods)
    elif _is_extensibility_methods(methods):
        generate_plots_b6(all_results, output_dir, timestamp, methods)
    else:
        generate_plots(all_results, output_dir, timestamp, methods)
    print(f"Plots generated from CSV: {csv_file}")


def _is_overhead_csv(df) -> bool:
    """Return True when the CSV has the benchmark-8 overhead column schema."""
    return 'wall_time_us' in df.columns and 'ffi_roundtrip_us' in df.columns


def generate_plots_b8(df, output_dir, timestamp: str):
    """
    Generate the overhead breakdown stacked-bar chart for Benchmark 8.
    """
    import numpy as np

    configure_latex_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    sizes = sorted(df['size'].unique().tolist())

    # Overhead components with distinct colors and hatch patterns for paper readability
    components = [
        ('validate_backend_us', r'validate\_backend()', '#003f5c', ''),
        ('validate_model_us', r'validate\_model()', '#444e86', '//'),
        ('pre_ffi_overhead_us', r'Pre-FFI C overhead', '#955196', '\\\\'),
        ('rs_dispatch_us', r'Rust dispatch', '#dd5182', '..'),
        ('pure_embedding_us', r'EmbedAnything inference', '#ff6e54', 'xx'),
        ('rs_to_c_return_us', r'Rs\,$\to$\,C return', '#ffa600', '++'),
        ('post_ffi_overhead_us', r'Post-FFI C overhead', '#00af91', '||'),
    ]

    # Aggregate per size
    agg = {}  # size -> {col: (mean, std)}
    for sz in sizes:
        sub = df[df['size'] == sz]
        agg[sz] = {}
        for col, _, _, _ in components:
            vals = sub[col].dropna().values if col in sub.columns else np.array([])
            agg[sz][col] = (float(np.mean(vals)) if len(vals) else 0.0,
                            float(np.std(vals)) if len(vals) else 0.0)
        for col in ('wall_time_us', 'total_c_ext_us'):
            vals = sub[col].dropna().values if col in sub.columns else np.array([])
            agg[sz][col] = (float(np.mean(vals)) if len(vals) else 0.0,
                            float(np.std(vals)) if len(vals) else 0.0)

    x = list(range(len(sizes)))
    xlabels = [str(s) for s in sizes]

    # Standardized figure size for paper plots
    fig, ax = plt.subplots(figsize=(7, 5))

    bottoms = [0.0] * len(sizes)
    for col, label, color, hatch in components:
        heights = [agg[s][col][0] / 1_000.0 for s in sizes]  # µs → ms
        ax.bar(x, heights, bottom=bottoms, label=label, color=color,
               hatch=hatch, edgecolor='white', linewidth=0.5, zorder=3)
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    # ±1σ error bars on total wall time
    wall_means = [agg[s]['wall_time_us'][0] / 1_000.0 for s in sizes]
    wall_stds = [agg[s]['wall_time_us'][1] / 1_000.0 for s in sizes]
    ax.errorbar(x, wall_means, yerr=wall_stds, fmt='none', color='#333333',
                capsize=4, linewidth=1.5, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel(r'Batch size (\# texts)')
    ax.set_ylabel(r'Time (ms)')

    # Paper style: Use best legend placement and remove plot title
    ax.legend(frameon=True, framealpha=0.9, loc='best')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_dir / f'overhead_breakdown_{timestamp}.pdf',
                format='pdf', bbox_inches='tight')
    plt.savefig(output_dir / f'overhead_breakdown_{timestamp}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Benchmark 8 plot saved to {output_dir}")
