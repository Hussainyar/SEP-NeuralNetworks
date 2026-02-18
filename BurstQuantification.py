import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.gridspec import GridSpec

# --- Publication-Quality Style Settings (Arial Font, Consistent with Figure 1) ---
plt.rcParams.update({
    'font.family': 'sans-serif',  # Changed to sans-serif for Arial
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,  # Changed to 300 DPI for better quality
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02
})

# --- Scientific Color Palette (Consistent with Figure 1) ---
colors = {
    'SEP1000': '#1f77b4',   # Professional blue for SEP=1000
    'SEP500': '#ff7f0e',    # Professional orange for SEP=500
    'excitatory': '#d62728',  # Red for excitatory neurons
    'inhibitory': '#0055a4'   # Darker blue for inhibitory neurons
}

# --- Data Functions ---
def read_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            data.append((float(parts[0]), int(parts[1])))
    return data

def process_data(data):
    neuron_fires = {}
    for t, nid in data:
        neuron_fires.setdefault(nid, []).append(t)
    for nid in neuron_fires:
        neuron_fires[nid].sort()
    max_fires = max(len(v) for v in neuron_fires.values())
    matrix = np.full((5000, max_fires), np.nan)
    for nid, times in neuron_fires.items():
        matrix[nid - 1, :len(times)] = times
    return matrix

def count_bursts(spike_times, isi_thresh, min_spikes):
    durations = []
    spike_times = np.array(spike_times)
    if len(spike_times) < min_spikes:
        return 0, durations
    isi = np.diff(spike_times)
    burst_mask = isi <= isi_thresh
    boundaries = np.where(np.concatenate(([burst_mask[0]], burst_mask[:-1] != burst_mask[1:], [True])))[0]
    lengths = np.diff(boundaries)[::2]
    indices = np.where(lengths >= min_spikes)[0] * 2
    for idx in indices:
        start = boundaries[idx]
        end = boundaries[idx + 1]
        durations.append(spike_times[end] - spike_times[start])
    return len(durations), durations

def analyze_bursts(matrix, isi_thresh, min_spikes):
    counts = np.zeros(matrix.shape[0])
    durations = []
    for i in range(matrix.shape[0]):
        spike_times = matrix[i, ~np.isnan(matrix[i])]
        count, d = count_bursts(spike_times, isi_thresh, min_spikes)
        counts[i] = count
        durations.extend(d)
    return counts, durations

def load_firing_dict(file_path):
    data = np.genfromtxt(file_path, dtype=None, encoding=None)
    fires = {}
    for t, n in data:
        fires.setdefault(int(n), []).append(float(t))
    for k in fires:
        fires[k].sort()
    return fires

def filter_neurons(fires, start, end):
    return {nid: [t for t in ts if start <= t <= end] for nid, ts in fires.items() if any(start <= t <= end for t in ts)}

# --- Raster Plot Function ---
def plot_spike_raster(ax, neuron_fires, ex_neurons, inh_neurons, start, end, panel_label, condition_label, ylabel=False):
    total_neurons = len(ex_neurons) + len(inh_neurons)

    # Plot excitatory neurons
    for i, nid in enumerate(ex_neurons):
        if nid in neuron_fires:
            ax.eventplot([t - start for t in neuron_fires[nid]], 
                        lineoffsets=i + 1, 
                        linelengths=0.9, 
                        linewidths=1.2,  # Slightly thicker for better visibility
                        color=colors['excitatory'])
    
    # Plot inhibitory neurons
    for i, nid in enumerate(inh_neurons):
        if nid in neuron_fires:
            ax.eventplot([t - start for t in neuron_fires[nid]], 
                        lineoffsets=len(ex_neurons) + i + 1, 
                        linelengths=0.9, 
                        linewidths=1.2,  # Slightly thicker for better visibility
                        color=colors['inhibitory'])

    # Add light grid lines for better readability
    for y in range(1, total_neurons + 1):
        ax.hlines(y, xmin=0, xmax=end - start, color='#CCCCCC', linewidth=0.3, linestyles='-', zorder=0, alpha=0.5)

    ax.set_xlim(0, end - start)
    ax.set_ylim(0.5, total_neurons + 0.5)
    ax.set_yticks(range(0, total_neurons + 1, 5))

    if ylabel:
        ax.set_yticklabels([str(i) for i in range(0, total_neurons + 1, 5)])
        ax.set_ylabel('Neuron ID', fontweight='normal', fontsize=10)
    else:
        ax.set_yticklabels([])

    ax.set_xlabel('Time (ms)', fontweight='normal', fontsize=10)
    
    # Panel label outside the plot (same size as before)
    ax.text(-0.15, 1.08, panel_label, transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='right')
    
    # Condition label as title
    ax.set_title(condition_label, loc='center', fontweight='normal', fontsize=10)
    
    # Professional spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('black')
    
    # Keep all spines for professional look
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_linewidth(0.8)
    ax.spines['right'].set_linewidth(0.8)
    
    # Enhanced tick parameters
    ax.tick_params(direction='out', length=3, width=0.8, colors='black', which='major')
    ax.tick_params(direction='out', length=1.5, width=0.6, colors='black', which='minor')
    
    # Set background to white
    ax.set_facecolor('white')

# --- Load Data ---
file1 = 'firingpattern1000-2000.dat'
file2 = 'firingpattern1500-2000.dat'
data1 = read_data(file1)
data2 = read_data(file2)
matrix1 = process_data(data1)
matrix2 = process_data(data2)
burst_counts1, durations1 = analyze_bursts(matrix1, 7.5, 3)
burst_counts2, durations2 = analyze_bursts(matrix2, 7.5, 3)

fires1 = filter_neurons(load_firing_dict(file1), 2000, 10000)
fires2 = filter_neurons(load_firing_dict(file2), 2000, 10000)

ex_neurons = list(np.arange(2989, 2989 + 16))
inh_neurons = list(np.arange(4000, 4000 + 4))

# --- Create Main Figure Optimized for A4 Journal Layout ---
fig = plt.figure(figsize=(11, 8))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1], 
              hspace=0.30, wspace=0.30)

# --- Define the four axes ---
ax_a = fig.add_subplot(gs[0, 0])  # Panel A (top-left)
ax_b = fig.add_subplot(gs[0, 1])  # Panel B (top-right)
ax_c = fig.add_subplot(gs[1, 0])  # Panel C (bottom-left)
ax_d = fig.add_subplot(gs[1, 1])  # Panel D (bottom-right)

# Top row: Raster plots with firing patterns
plot_spike_raster(ax_a, fires1, ex_neurons, inh_neurons, 2000, 10000, '(A)', 'SEP = 1000', ylabel=True)
plot_spike_raster(ax_b, fires2, ex_neurons, inh_neurons, 2000, 10000, '(B)', 'SEP = 500', ylabel=False)
ax_a.set_xticks([0, 2000, 4000, 6000, 8000])
ax_b.set_xticks([0, 2000, 4000, 6000, 8000])

# Bottom row: Panel C – Burst Counts with consistent styling
bins_counts = np.arange(0.5, 30.5, 1)
ax_c.hist(burst_counts1, bins=bins_counts, alpha=0.65, 
          color=colors['SEP1000'], edgecolor='black', linewidth=0.5, label='SEP = 1000')
ax_c.hist(burst_counts2, bins=bins_counts, alpha=0.65, 
          color=colors['SEP500'], edgecolor='black', linewidth=0.5, label='SEP = 500')

ax_c.set_xlabel('Bursts per neuron', fontweight='normal', fontsize=10)
ax_c.set_ylabel('Neuron count (log)', fontweight='normal', fontsize=10)
ax_c.set_yscale('log')
ax_c.set_xlim(0, 30)
ax_c.set_yticks([1, 10, 100, 1000])

ax_c.get_yaxis().set_major_formatter(ScalarFormatter())

# Panel C label (same size as before)
ax_c.text(-0.15, 1.08, '(C)', transform=ax_c.transAxes,
          fontsize=14, fontweight='bold', va='top', ha='right')

# Professional spine styling for panel C
for spine in ax_c.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color('black')

ax_c.spines['top'].set_visible(True)
ax_c.spines['right'].set_visible(True)
ax_c.spines['top'].set_linewidth(0.8)
ax_c.spines['right'].set_linewidth(0.8)

# Grid for better readability
ax_c.grid(True, linestyle='-', linewidth=0.4, color='#CCCCCC', alpha=0.5, zorder=0)
ax_c.set_facecolor('white')

# Enhanced tick parameters for panel C
ax_c.tick_params(direction='out', length=3, width=0.8, colors='black', which='major')
ax_c.tick_params(direction='out', length=1.5, width=0.6, colors='black', which='minor')

# Bottom row: Panel D – Burst Durations with consistent styling
duration_bins = np.linspace(0, 1000, 200)
ax_d.hist(durations1, bins=duration_bins, alpha=0.65, 
          color=colors['SEP1000'], edgecolor='black', linewidth=0.5,
          label='SEP = 1000')
ax_d.hist(durations2, bins=duration_bins, alpha=0.65, 
          color=colors['SEP500'], edgecolor='black', linewidth=0.5,
          label='SEP = 500')

ax_d.set_xlabel('Burst duration (ms)', fontweight='normal', fontsize=10)
ax_d.set_ylabel('Count (log)', fontweight='normal', fontsize=10)
ax_d.set_yscale('log')
ax_d.set_xlim(0, 1000)
ax_d.set_yticks([1, 10, 100, 1000, 2500])
ax_d.get_yaxis().set_major_formatter(ScalarFormatter())

# Panel D label (same size as before)
ax_d.text(-0.15, 1.08, '(D)', transform=ax_d.transAxes,
          fontsize=14, fontweight='bold', va='top', ha='right')

# Legend only in panel D (consistent with Figure 1 style)
legend = ax_d.legend(frameon=True, framealpha=1.0, 
                    edgecolor='black', fancybox=False, 
                    loc='upper right', fontsize=9)
legend.get_frame().set_linewidth(0.8)
legend.get_frame().set_facecolor('white')

# Professional spine styling for panel D
for spine in ax_d.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color('black')

ax_d.spines['top'].set_visible(True)
ax_d.spines['right'].set_visible(True)
ax_d.spines['top'].set_linewidth(0.8)
ax_d.spines['right'].set_linewidth(0.8)

# Grid for better readability
ax_d.grid(True, linestyle='-', linewidth=0.4, color='#CCCCCC', alpha=0.5, zorder=0)
ax_d.set_facecolor('white')

# Enhanced tick parameters for panel D
ax_d.tick_params(direction='out', length=3, width=0.8, colors='black', which='major')
ax_d.tick_params(direction='out', length=1.5, width=0.6, colors='black', which='minor')

# Final layout adjustments
plt.tight_layout()
plt.savefig("FIG3.pdf", dpi=300, bbox_inches='tight', facecolor='white')
#plt.savefig("Enhanced_Bursts_Figure.png", dpi=300, bbox_inches='tight', facecolor='white')  # Also save as PNG
plt.show()