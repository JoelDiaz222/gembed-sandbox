import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

COLOR_PG_MAIN = '#003f5c'  # Navy Blue (Internal/Mono-Store)
COLOR_PG_ALT = '#444e86'  # Slate (Indexed/Optimized)
COLOR_VECTOR_QD = '#ff1f5b'  # Crimson (Qdrant)
COLOR_VECTOR_CH = '#ffa600'  # Amber (Chroma)
COLOR_DIRECT = '#00af91'  # Teal (In-Process Direct)
COLOR_REMOTE_GRPC = '#845ec2'  # Light Indigo (gRPC Remote)
COLOR_REMOTE_HTTP = '#d65db1'  # Light Magenta (HTTP Remote)
COLOR_B6_EA = '#003f5c'  # Navy  – EmbedAnything (Candle + CUDA)
COLOR_B6_ORT = '#ffa600'  # Amber – ONNX Runtime (CPU)

# Benchmark 7 – Portability (one colour per DB adapter)
COLOR_B7_PG = '#003f5c'  # Navy   – PostgreSQL (pg_gembed)
COLOR_B7_MYSQL = '#ef5675'  # Coral  – MySQL (mysql_gembed)
COLOR_B7_REDIS = '#ffa600'  # Amber  – Redis (redis_gembed)

STYLE_MAP = {
    # PostgreSQL / Internal Group
    'pg_local': (COLOR_PG_MAIN, '-', 'o'),
    'pg_mono_store': (COLOR_PG_MAIN, '-', 'o'),
    'mono_store': (COLOR_PG_MAIN, '-', 'o'),
    'internal': (COLOR_PG_MAIN, '-', 'o'),
    'pg': (COLOR_PG_MAIN, '-', 'o'),

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

    # Application Clients (Direct/In-Process)
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
    'pg_local': 'PG (Local)',
    'pg_mono_store': 'Mono-Store (PG, Local)',
    'mono_store': 'Mono-Store (PG, Local)',
    'internal': 'PG (Local)',
    'pg_grpc': 'PG (gRPC)',
    'pg_http': 'PG (HTTP)',
    'pg_indexed': 'PG (Local, Indexed)',
    'pg_deferred': 'PG (Local, Deferred Index)',
    'pg_local_indexed': 'PG (Local, Indexed)',
    'pg_local_deferred': 'PG (Local, Deferred Index)',
    'pg_mono_indexed': 'Mono-Store (PG, Local, Indexed)',
    'pg_mono_deferred': 'Mono-Store (PG, Local, Deferred Index)',
    'pg_grpc_indexed': 'PG (gRPC, Indexed)',
    'pg_grpc_deferred': 'PG (gRPC, Deferred Index)',
    'ext_direct': 'Ext Direct',
    'external': 'PG External Client',
    'ext_direct_indexed': 'Ext Direct (Indexed)',
    'ext_direct_deferred': 'Ext Direct (Deferred Index)',
    'pg_unified': 'PG Local',
    'pg_direct': 'Ext Direct',
    'pg_gembed_unified': 'PG Local',
    'ext_grpc': 'External gRPC',
    'ext_http': 'External HTTP',
    'chroma': 'ChromaDB',
    'qdrant': 'Qdrant',
    'poly_chroma': 'Poly-Store (PG, ChromaDB)',
    'poly_qdrant': 'Poly-Store (PG, Qdrant)',
    'two_step_chroma': 'Poly-Store (PG, ChromaDB)',
    'two_step_qdrant': 'Poly-Store (PG, Qdrant)',
    'qd_indexed': 'Qdrant (Indexed)',
    'qd_deferred': 'Qdrant (Deferred Index)',
    'mono_pg_unified_deferred': 'Mono-Store (PG Local)',
    'mono_ext_direct_deferred': 'Mono-Store (Ext Direct)',
    'poly_qdrant_deferred': 'Poly-Store (PG, Qdrant Deferred)',
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
        "axes.labelsize": 11,
        "font.size": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.5,
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


def generate_plots(all_results: List[dict], output_dir: Path, timestamp: str, methods: List[str]):
    configure_latex_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = [r['size'] for r in all_results]

    # Data-driven component detection
    has_pg_data = any(m in r and r[m].get('pg_mem_peak', 0) > 0 for r in all_results for m in methods)
    has_qd_data = any(m in r and r[m].get('qd_mem_peak', 0) > 0 for r in all_results for m in methods)

    def plot_dual(cpu_key, mem_key, component_name, filename_suffix):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        for method in methods:
            color, ls, marker, label = get_style(method)
            y_vals = [r[method].get(f'{cpu_key}_median', r[method].get(cpu_key, 0)) for r in all_results if method in r]
            q1_vals = [r[method].get(f'{cpu_key}_q1', r[method].get(f'{cpu_key}_median', 0)) for r in all_results if
                       method in r]
            q3_vals = [r[method].get(f'{cpu_key}_q3', r[method].get(f'{cpu_key}_median', 0)) for r in all_results if
                       method in r]
            # Asymmetric error bars: lower = median - Q1, upper = Q3 - median
            y_errs = [[y - q1 for y, q1 in zip(y_vals, q1_vals)], [q3 - y for y, q3 in zip(y_vals, q3_vals)]]
            if not any(y_vals): continue
            ax1.errorbar(sizes, y_vals, yerr=y_errs, fmt=marker, linestyle=ls, color=color, label=label,
                         linewidth=1.5, capsize=3, markersize=5, alpha=0.9)

        ax1.set_xlabel('Scale (Log Scale)')
        ax1.set_ylabel(r'CPU (\%)')
        ax1.set_title(f'{component_name} CPU Usage')
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.set_xscale('log', base=2)

        for method in methods:
            color, ls, marker, label = get_style(method)
            y_vals = [r[method].get(f'{mem_key}_median', r[method].get(mem_key, 0)) for r in all_results if method in r]
            q1_vals = [r[method].get(f'{mem_key}_q1', r[method].get(f'{mem_key}_median', 0)) for r in all_results if
                       method in r]
            q3_vals = [r[method].get(f'{mem_key}_q3', r[method].get(f'{mem_key}_median', 0)) for r in all_results if
                       method in r]
            # Asymmetric error bars: lower = median - Q1, upper = Q3 - median
            y_errs = [[y - q1 for y, q1 in zip(y_vals, q1_vals)], [q3 - y for y, q3 in zip(y_vals, q3_vals)]]
            if not any(y_vals): continue
            ax2.errorbar(sizes, y_vals, yerr=y_errs, fmt=marker, linestyle=ls, color=color, label=label,
                         linewidth=1.5, capsize=3, markersize=5, alpha=0.9)

        ax2.set_xlabel('Scale (Log Scale)')
        ax2.set_ylabel('Memory (MB)')
        ax2.set_title(f'{component_name} Memory Usage')
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.set_xscale('log', base=2)

        handles, labels = ax2.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(output_dir / f"{filename_suffix}_{timestamp}.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(output_dir / f"{filename_suffix}_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_throughput():
        plt.figure(figsize=(7, 5))
        for method in methods:
            color, ls, marker, label = get_style(method)
            y_vals = [r[method].get('throughput_median', r[method].get('throughput', 0)) for r in all_results if
                      method in r]
            q1_vals = [r[method].get('throughput_q1', r[method].get('throughput_median', 0)) for r in all_results if
                       method in r]
            q3_vals = [r[method].get('throughput_q3', r[method].get('throughput_median', 0)) for r in all_results if
                       method in r]
            # Asymmetric error bars: lower = median - Q1, upper = Q3 - median
            y_errs = [[y - q1 for y, q1 in zip(y_vals, q1_vals)], [q3 - y for y, q3 in zip(y_vals, q3_vals)]]
            if not any(y_vals): continue
            plt.errorbar(sizes, y_vals, yerr=y_errs, fmt=marker, linestyle=ls, color=color, label=label,
                         linewidth=1.5, capsize=3, markersize=5, alpha=0.9)

        plt.xlabel('Scale (Log Scale)')
        plt.ylabel('Throughput (ops/sec)')
        plt.title('System Throughput')
        plt.legend(frameon=True, framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xscale('log', base=2)
        plt.savefig(output_dir / f"throughput_{timestamp}.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(output_dir / f"throughput_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_normalized_throughput():
        baseline_candidates = ['ext_direct', 'pg_direct', 'ext_direct_indexed', 'mono_pg_direct_deferred',
                               'mono_ext_direct_deferred']
        baseline_method = None
        for candidate in baseline_candidates:
            if candidate in methods:
                baseline_method = candidate
                break

        if not baseline_method:
            return

        plt.figure(figsize=(8, 6))

        baseline_y_vals = [r[baseline_method].get('throughput_median', r[baseline_method].get('throughput', 0)) for r in
                           all_results if baseline_method in r]
        if not any(baseline_y_vals):
            plt.close()
            return

        for method in methods:
            color, ls, marker, label = get_style(method)
            y_vals = [r[method].get('throughput_median', r[method].get('throughput', 0)) for r in all_results if
                      method in r]

            if not any(y_vals): continue

            norm_y_vals = []
            valid_sizes = []
            for sz, y, b in zip(sizes, y_vals, baseline_y_vals):
                if y and b and b > 0:
                    norm_y_vals.append(y / b)
                    valid_sizes.append(sz)

            if not norm_y_vals: continue

            plt.plot(valid_sizes, norm_y_vals, marker=marker, linestyle=ls, color=color, label=label, alpha=0.9)

        plt.xscale('log', base=2)
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        plt.xlabel('Batch Size (Log Scale)')
        baseline_label = LABEL_MAP.get(baseline_method, baseline_method)
        plt.ylabel(f'Relative Throughput (vs {baseline_label})')
        plt.title(f'Relative Throughput Normalized to {baseline_label}')
        plt.legend(frameon=True, framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.3)

        plt.savefig(output_dir / f"normalized_throughput_{timestamp}.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(output_dir / f"normalized_throughput_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

    plot_throughput()
    plot_normalized_throughput()
    plot_dual('py_cpu', 'py_mem_peak', 'Python Process', 'python_resources')
    if has_pg_data:
        plot_dual('pg_cpu', 'pg_mem_peak', 'PostgreSQL Connection', 'postgres_resources')
    if has_qd_data:
        plot_dual('qd_cpu', 'qd_mem_peak', 'Qdrant Container', 'qdrant_resources')
    plot_dual('sys_cpu', 'sys_mem', 'System', 'system_resources')
    print(f"Plots saved to {output_dir} (PDF + PNG)")


def _is_extensibility_methods(methods: List[str]) -> bool:
    """Return True when every method follows the <backend>_<model_name> pattern
    used by Benchmark 6 (e.g. 'embed_anything_openai--clip-vit-base-patch32', 'ort_openai--clip-vit-base-patch32')."""
    b6_backends = {'embed_anything', 'ort', 'candle'}
    return bool(methods) and all(
        any(m.startswith(f"{b}_") for b in b6_backends) for m in methods
    )


def _is_portability_methods(methods: List[str]) -> bool:
    """Return True when every method follows the <adapter>_<model_name> pattern
    used by Benchmark 7 (e.g. 'pg_sentence-transformers--all-MiniLM-L6-v2')."""
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
            d.get(f'{metric}_median', d.get(metric, 0)) or 0,
            d.get(f'{metric}_q1', d.get(f'{metric}_median', d.get(metric, 0))) or 0,
            d.get(f'{metric}_q3', d.get(f'{metric}_median', d.get(metric, 0))) or 0,
        )

    # Throughput grouped bar chart
    bar_width = 0.8 / n_backends
    x = range(n_models)

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * n_models), 4.5))

    for bi, backend in enumerate(seen_backends):
        offsets = [xi + (bi - (n_backends - 1) / 2) * bar_width for xi in x]
        heights, err_lo, err_hi = [], [], []
        for model_name in seen_models:
            method_name = f"{backend}_{model_name}"
            med, q1, q3 = _get(method_name, 'throughput')
            heights.append(med)
            err_lo.append(max(0.0, med - q1))
            err_hi.append(max(0.0, q3 - med))

        ax.bar(
            offsets, heights,
            width=bar_width,
            label=backend_label[backend],
            color=backend_color[backend],
            hatch=backend_hatch[backend],
            edgecolor='white',
            linewidth=0.5,
            yerr=[err_lo, err_hi],
            capsize=3,
            error_kw={'linewidth': 1.0, 'ecolor': '#555555'},
            zorder=3,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(seen_models, rotation=15, ha='right')
    ax.set_xlabel('Model')
    ax.set_ylabel(r'Throughput (img/s)')
    ax.set_title(r'Image Embedding Throughput by Backend \& Model (PG Local)')
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_dir / f"throughput_{timestamp}.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(output_dir / f"throughput_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Speedup bar chart  (embed_anything / ort, per model)
    if 'embed_anything' in seen_backends and 'ort' in seen_backends:
        fig, ax = plt.subplots(figsize=(max(5, 2.0 * n_models), 4.0))
        speedups, err_lo_sp, err_hi_sp = [], [], []
        for model_name in seen_models:
            ea_med, ea_q1, ea_q3 = _get(f'embed_anything_{model_name}', 'throughput')
            ort_med, ort_q1, ort_q3 = _get(f'ort_{model_name}', 'throughput')
            sp = (ea_med / ort_med) if ort_med > 0 else 0.0
            # Simple propagation: speedup bounds from the bar extremes
            sp_lo = (ea_q1 / ort_q3) if ort_q3 > 0 else 0.0
            sp_hi = (ea_q3 / ort_q1) if ort_q1 > 0 else 0.0
            speedups.append(sp)
            err_lo_sp.append(max(0.0, sp - sp_lo))
            err_hi_sp.append(max(0.0, sp_hi - sp))

        ax.bar(
            list(x), speedups,
            color=COLOR_B6_EA,
            edgecolor='white',
            linewidth=0.5,
            yerr=[err_lo_sp, err_hi_sp],
            capsize=3,
            error_kw={'linewidth': 1.0, 'ecolor': '#555555'},
            zorder=3,
        )
        ax.axhline(y=1.0, color='#aaaaaa', linestyle='--', linewidth=1.0, zorder=2)
        ax.set_xticks(list(x))
        ax.set_xticklabels(seen_models, rotation=15, ha='right')
        ax.set_xlabel('Model')
        ax.set_ylabel(r'Speedup vs.\ CPU')
        ax.set_title(r'EmbedAnything (Candle, CUDA) Speedup over CPU')
        ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(output_dir / f"speedup_{timestamp}.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(output_dir / f"speedup_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Resource plots: CPU + memory, one dual chart per component.
    # Same grouped-bar layout as the throughput chart.
    def plot_resource_dual(cpu_key, mem_key, component_name, filename_suffix):
        fig, (ax_cpu, ax_mem) = plt.subplots(1, 2, figsize=(max(10, 4.0 * n_models), 4.5))

        for bi, backend in enumerate(seen_backends):
            offsets = [xi + (bi - (n_backends - 1) / 2) * bar_width for xi in x]
            cpu_heights, cpu_lo, cpu_hi = [], [], []
            mem_heights, mem_lo, mem_hi = [], [], []
            for model_name in seen_models:
                method_name = f"{backend}_{model_name}"
                c_med, c_q1, c_q3 = _get(method_name, cpu_key)
                m_med, m_q1, m_q3 = _get(method_name, mem_key)
                cpu_heights.append(c_med)
                cpu_lo.append(max(0.0, c_med - c_q1))
                cpu_hi.append(max(0.0, c_q3 - c_med))
                mem_heights.append(m_med)
                mem_lo.append(max(0.0, m_med - m_q1))
                mem_hi.append(max(0.0, m_q3 - m_med))

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
            ax_cpu.bar(offsets, cpu_heights, yerr=[cpu_lo, cpu_hi], **bar_kwargs)
            ax_mem.bar(offsets, mem_heights, yerr=[mem_lo, mem_hi], **bar_kwargs)

        for ax, ylabel, title in [
            (ax_cpu, r'CPU (\%)', f'{component_name} CPU Usage'),
            (ax_mem, 'Memory (MB)', f'{component_name} Memory Usage'),
        ]:
            ax.set_xticks(list(x))
            ax.set_xticklabels(seen_models, rotation=15, ha='right')
            ax.set_xlabel('Model')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
            ax.set_axisbelow(True)

        handles, labels = ax_mem.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                   ncol=n_backends, frameon=False)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(output_dir / f"{filename_suffix}_{timestamp}.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(output_dir / f"{filename_suffix}_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

    plot_resource_dual('py_cpu', 'py_mem_peak', 'Python Process', 'python_resources')
    plot_resource_dual('pg_cpu', 'pg_mem_peak', 'PostgreSQL Connection', 'postgres_resources')
    plot_resource_dual('sys_cpu', 'sys_mem', 'System', 'system_resources')

    print(f"Benchmark 6 plots saved to {output_dir} (PDF + PNG)")


def generate_plots_b7(
        all_results: List[dict],
        output_dir: Path,
        timestamp: str,
        methods: List[str],
):
    """Grouped bar chart for Benchmark 7 (portability: pg vs mysql vs redis).

    Layout: one group of bars per model, one bar per DB adapter within the group.
    Only a single size is expected (the benchmark runs at one fixed scale).
    IQR-based asymmetric error bars are drawn when available.
    """
    configure_latex_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    b7_adapters = ['pg', 'mysql', 'redis']
    adapter_color = {
        'pg': COLOR_B7_PG,
        'mysql': COLOR_B7_MYSQL,
        'redis': COLOR_B7_REDIS,
    }
    adapter_label = {
        'pg': LABEL_MAP.get('pg', 'PostgreSQL (pg\\_gembed)'),
        'mysql': LABEL_MAP.get('mysql', 'MySQL (mysql\\_gembed)'),
        'redis': LABEL_MAP.get('redis', 'Redis (redis\\_gembed)'),
    }
    adapter_hatch = {
        'pg': '',
        'mysql': '//',
        'redis': 'xx',
    }

    # Preserve order in which adapters and models first appear
    seen_adapters = []
    seen_models = []
    method_map: dict = {}  # method_name -> (adapter, model_slug)
    for m in methods:
        for a in b7_adapters:
            if m.startswith(f"{a}_"):
                model_name = m[len(a) + 1:]  # everything after "<adapter>_"
                method_map[m] = (a, model_name)
                if a not in seen_adapters:
                    seen_adapters.append(a)
                if model_name not in seen_models:
                    seen_models.append(model_name)
                break

    n_models = len(seen_models)
    n_adapters = len(seen_adapters)
    if n_models == 0 or n_adapters == 0:
        return

    # Aggregate across sizes (expect a single size, but handle multiple gracefully)
    aggregated: dict = {}
    for r in all_results:
        for m in methods:
            if m not in r:
                continue
            existing = aggregated.get(m, {})
            for key, val in r[m].items():
                if key not in existing:
                    existing[key] = val
            aggregated[m] = existing

    def _get(m, metric):
        d = aggregated.get(m, {})
        return (
            d.get(f'{metric}_median', d.get(metric, 0)) or 0,
            d.get(f'{metric}_q1', d.get(f'{metric}_median', d.get(metric, 0))) or 0,
            d.get(f'{metric}_q3', d.get(f'{metric}_median', d.get(metric, 0))) or 0,
        )

    bar_width = 0.8 / n_adapters
    x = range(n_models)

    # ── Throughput grouped bar chart ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, 2.8 * n_models), 4.5))

    for ai, adapter in enumerate(seen_adapters):
        offsets = [xi + (ai - (n_adapters - 1) / 2) * bar_width for xi in x]
        heights, err_lo, err_hi = [], [], []
        for model_slug in seen_models:
            method_name = f"{adapter}_{model_slug}"
            med, q1, q3 = _get(method_name, 'throughput')
            heights.append(med)
            err_lo.append(max(0.0, med - q1))
            err_hi.append(max(0.0, q3 - med))

        ax.bar(
            offsets, heights,
            width=bar_width,
            label=adapter_label[adapter],
            color=adapter_color[adapter],
            hatch=adapter_hatch[adapter],
            edgecolor='white',
            linewidth=0.5,
            yerr=[err_lo, err_hi],
            capsize=3,
            error_kw={'linewidth': 1.0, 'ecolor': '#555555'},
            zorder=3,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(seen_models, rotation=20, ha='right')
    ax.set_xlabel('Model')
    ax.set_ylabel(r'Throughput (texts/s)')
    ax.set_title(r'Text Embedding Throughput by Adapter \& Model (EmbedAnything)')
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_dir / f"throughput_{timestamp}.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(output_dir / f"throughput_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Resource plots ────────────────────────────────────────────────────────
    def plot_resource_dual(cpu_key, mem_key, component_name, filename_suffix):
        fig, (ax_cpu, ax_mem) = plt.subplots(1, 2, figsize=(max(10, 4.0 * n_models), 4.5))

        for ai, adapter in enumerate(seen_adapters):
            offsets = [xi + (ai - (n_adapters - 1) / 2) * bar_width for xi in x]
            cpu_heights, cpu_lo, cpu_hi = [], [], []
            mem_heights, mem_lo, mem_hi = [], [], []
            for model_slug in seen_models:
                method_name = f"{adapter}_{model_slug}"
                c_med, c_q1, c_q3 = _get(method_name, cpu_key)
                m_med, m_q1, m_q3 = _get(method_name, mem_key)
                cpu_heights.append(c_med)
                cpu_lo.append(max(0.0, c_med - c_q1))
                cpu_hi.append(max(0.0, c_q3 - c_med))
                mem_heights.append(m_med)
                mem_lo.append(max(0.0, m_med - m_q1))
                mem_hi.append(max(0.0, m_q3 - m_med))

            bar_kwargs = dict(
                width=bar_width,
                label=adapter_label[adapter],
                color=adapter_color[adapter],
                hatch=adapter_hatch[adapter],
                edgecolor='white',
                linewidth=0.5,
                capsize=3,
                error_kw={'linewidth': 1.0, 'ecolor': '#555555'},
                zorder=3,
            )
            ax_cpu.bar(offsets, cpu_heights, yerr=[cpu_lo, cpu_hi], **bar_kwargs)
            ax_mem.bar(offsets, mem_heights, yerr=[mem_lo, mem_hi], **bar_kwargs)

        for ax, ylabel, title in [
            (ax_cpu, r'CPU (\%)', f'{component_name} CPU Usage'),
            (ax_mem, 'Memory (MiB)', f'{component_name} Memory Usage'),
        ]:
            ax.set_xticks(list(x))
            ax.set_xticklabels(seen_models, rotation=20, ha='right')
            ax.set_xlabel('Model')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
            ax.set_axisbelow(True)

        handles, labels = ax_mem.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                   ncol=n_adapters, frameon=False)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(output_dir / f"{filename_suffix}_{timestamp}.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(output_dir / f"{filename_suffix}_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

    plot_resource_dual('py_cpu', 'py_mem_peak', 'Python Process', 'python_resources')
    plot_resource_dual('pg_cpu', 'pg_mem_peak', 'DB Process', 'db_resources')
    plot_resource_dual('sys_cpu', 'sys_mem', 'System', 'system_resources')

    print(f"Benchmark 7 plots saved to {output_dir} (PDF + PNG)")


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
