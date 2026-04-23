import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ─── Shared Helpers ────────────────────────────────────────────────────────────

def load_data(file):
    data = pd.read_csv(file.name, usecols=['time_s', 'voltage_V'])
    data.columns = ['time', 'voltage']
    data['cycle'] = ((data['time'] - data['time'].iloc[0]) // 10).astype(int)
    return data

def compute_tau_discharge(time, voltage):
    voltage = np.clip(voltage, 1e-6, None)
    log_v = np.log(voltage)
    slope, _ = np.polyfit(time, log_v, 1)
    return -1 / slope

# ─── Raw Signal Viewer (start → end cycle range) ──────────────────────────────

def plot_raw_signal(file, start_raw, end_raw):
    if file is None:
        return None

    if start_raw > end_raw:
        start_raw, end_raw = end_raw, start_raw

    data = load_data(file)
    filtered_df = data[(data['cycle'] >= start_raw) & (data['cycle'] <= end_raw)]

    if filtered_df.empty:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.text(0.5, 0.5, "No data found in selected cycle range.",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        return fig

    num_shown = end_raw - start_raw + 1
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(filtered_df['time'], filtered_df['voltage'],
            color='#1f77b4', linewidth=0.8)
    ax.set_title(f"Raw DAQ Signal — Cycle {start_raw} to {end_raw}  ({num_shown} cycles)",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# ─── Aligned Cycles Plot (start/end cycle range) ──────────────────────────────

def plot_aligned_cycles(file, start_cycle, end_cycle):
    if file is None:
        return None

    if start_cycle > end_cycle:
        start_cycle, end_cycle = end_cycle, start_cycle

    data = load_data(file)
    grouped = data.groupby('cycle')
    cycle_ids = sorted([cid for cid in grouped.groups if start_cycle <= cid <= end_cycle])
    num_cycles = len(cycle_ids)

    if num_cycles == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No cycles found in selected range.",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        return fig

    colors = cm.viridis(np.linspace(0, 1, num_cycles))

    fig, ax = plt.subplots(figsize=(12, 5))
    for color, cycle_id in zip(colors, cycle_ids):
        cycle_df = grouped.get_group(cycle_id)
        t = cycle_df['time'].values - cycle_df['time'].iloc[0]
        v = cycle_df['voltage'].values
        ax.plot(t, v, color=color, linewidth=1, label=f'Cycle {cycle_id}')

    ax.set_title(
        f"Aligned Voltage Cycles — Cycle {start_cycle} to {end_cycle}  ({num_cycles} cycles)",
        fontsize=13, fontweight='bold')
    ax.set_xlabel("Time within Cycle (s)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.25)

    if num_cycles <= 30:
        ax.legend(fontsize=6, ncol=max(1, num_cycles // 10 + 1),
                  loc='upper right', framealpha=0.6)
    else:
        sm = plt.cm.ScalarMappable(cmap='viridis',
                                    norm=plt.Normalize(vmin=start_cycle, vmax=end_cycle))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label("Cycle Number", fontsize=9)

    plt.tight_layout()
    return fig

# ─── Degradation Analysis ──────────────────────────────────────────────────────

def analyze_data(file, resistor_value):
    if file is None:
        return None, None, None

    data = load_data(file)
    cycle_results = []

    for cycle_id, cycle_df in data.groupby('cycle'):
        t = cycle_df['time'].values - cycle_df['time'].iloc[0]
        v = cycle_df['voltage'].values
        mask = t >= 5
        t_d = t[mask] - 5
        v_d = v[mask]
        if len(t_d) > 1:
            try:
                tau = compute_tau_discharge(t_d, v_d)
                cap = tau / resistor_value
                cycle_results.append({'cycle': cycle_id, 'tau': tau, 'capacitance': cap})
            except Exception:
                pass

    results_df = pd.DataFrame(cycle_results)
    if results_df.empty:
        return pd.DataFrame(), None, None

    fig_cap, ax_cap = plt.subplots(figsize=(10, 4))
    ax_cap.plot(results_df['cycle'], results_df['capacitance'],
                color='#2196F3', linewidth=1.5, marker='o', markersize=2)
    ax_cap.set_title("Capacitance Degradation Over Cycles", fontsize=13, fontweight='bold')
    ax_cap.set_xlabel("Cycle Number")
    ax_cap.set_ylabel("Capacitance (F)")
    ax_cap.grid(True, alpha=0.25)
    plt.tight_layout()

    fig_tau, ax_tau = plt.subplots(figsize=(10, 4))
    ax_tau.plot(results_df['cycle'], results_df['tau'],
                color='#E53935', linewidth=1.5, marker='o', markersize=2)
    ax_tau.set_title("Time Constant (τ) vs Cycle", fontsize=13, fontweight='bold')
    ax_tau.set_xlabel("Cycle Number")
    ax_tau.set_ylabel("τ (s)")
    ax_tau.grid(True, alpha=0.25)
    plt.tight_layout()

    return results_df.head(20), fig_cap, fig_tau

# ─── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Capacitor DAQ Dashboard", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🔋 Capacitor DAQ Analysis Dashboard
    Upload your `daq_data14.csv` to explore raw waveforms and compute degradation metrics.
    """)

    # ── Shared Inputs ──────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(label="📂 Upload CSV Data (`time_s`, `voltage_V` columns)")
        with gr.Column(scale=1):
            r_input = gr.Number(label="Resistor Value (Ω)", value=1000)

    with gr.Row():
        btn_analyze = gr.Button("🔬 Run Degradation Analysis", variant="primary")

    gr.Markdown("---")

    # ── Section 1: Raw Signal Viewer ──────────────────────────────────────────
    gr.Markdown("## ⚡ Raw DAQ Signal Viewer")
    gr.Markdown("Select **start** and **end** cycle to view any window of the raw waveform.")

    with gr.Row():
        with gr.Column(scale=1):
            raw_start_slider = gr.Slider(
                minimum=0, maximum=2880, step=1, value=0,
                label="Start Cycle"
            )
            raw_end_slider = gr.Slider(
                minimum=0, maximum=2880, step=1, value=10,
                label="End Cycle"
            )
            btn_raw = gr.Button("📈 Show Raw Signal", variant="secondary")
        with gr.Column(scale=3):
            plot_raw_out = gr.Plot(label="Continuous Waveform")

    gr.Markdown("---")

    # ── Section 2: Aligned Cycles Viewer ──────────────────────────────────────
    gr.Markdown("## 🔄 Aligned Voltage Cycles Viewer")
    gr.Markdown("Select the **start** and **end** cycle numbers to overlay all cycles in that range.")

    with gr.Row():
        with gr.Column(scale=1):
            start_cycle_slider = gr.Slider(
                minimum=0, maximum=2880, step=1, value=0,
                label="Start Cycle"
            )
            end_cycle_slider = gr.Slider(
                minimum=0, maximum=2880, step=1, value=24,
                label="End Cycle"
            )
            btn_aligned = gr.Button("🔍 Update Aligned Plot", variant="primary")
        with gr.Column(scale=3):
            plot_aligned = gr.Plot(label="Aligned Voltage Cycles")

    gr.Markdown("---")

    # ── Section 3: Degradation Analysis ───────────────────────────────────────
    gr.Markdown("## 📊 Degradation Analysis")

    with gr.Row():
        table_out = gr.Dataframe(label="Cycle-wise Results (First 20 Cycles)", wrap=True)

    with gr.Row():
        plot_cap = gr.Plot(label="Capacitance Trend")
        plot_tau = gr.Plot(label="τ (Tau) Trend")

    # ── Interactions ──────────────────────────────────────────────────────────

    # Raw signal — button + slider release
    btn_raw.click(fn=plot_raw_signal,
                  inputs=[file_input, raw_start_slider, raw_end_slider],
                  outputs=plot_raw_out)
    raw_start_slider.release(fn=plot_raw_signal,
                              inputs=[file_input, raw_start_slider, raw_end_slider],
                              outputs=plot_raw_out)
    raw_end_slider.release(fn=plot_raw_signal,
                            inputs=[file_input, raw_start_slider, raw_end_slider],
                            outputs=plot_raw_out)

    # Aligned cycles
    btn_aligned.click(fn=plot_aligned_cycles,
                      inputs=[file_input, start_cycle_slider, end_cycle_slider],
                      outputs=plot_aligned)
    start_cycle_slider.release(fn=plot_aligned_cycles,
                                inputs=[file_input, start_cycle_slider, end_cycle_slider],
                                outputs=plot_aligned)
    end_cycle_slider.release(fn=plot_aligned_cycles,
                              inputs=[file_input, start_cycle_slider, end_cycle_slider],
                              outputs=plot_aligned)

    # Degradation analysis
    btn_analyze.click(fn=analyze_data,
                      inputs=[file_input, r_input],
                      outputs=[table_out, plot_cap, plot_tau])

if __name__ == "__main__":
    demo.launch()