import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.gridspec import GridSpec

# --- Enhanced Publication-Quality Style Settings ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Verdana', 'sans-serif'],
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',  # Changed from '#F8F8F8' to 'white'
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05
})

# --- Enhanced Scientific Color Palette ---
# Keep red/blue for excitatory/inhibitory neurons as standard in neuroscience
# But enhance the colors for SPEC comparison
colors = {
    'SDC1000': '#00A4B4',  # Brighter teal that stands out better
    'SDC500': '#E83E8C',    # Richer magenta for better contrast
    'excitatory': '#FF4136', # Bright red for excitatory neurons
    'inhibitory': '#0074D9'  # Bright blue for inhibitory neurons
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
def plot_spike_raster(ax, neuron_fires, ex_neurons, inh_neurons, start, end, title_label, ylabel=False):
    total_neurons = len(ex_neurons) + len(inh_neurons)

    # Plot excitatory neurons with enhanced styling
    for i, nid in enumerate(ex_neurons):
        if nid in neuron_fires:
            ax.eventplot([t - start for t in neuron_fires[nid]], 
                        lineoffsets=i + 1, 
                        linelengths=0.9, 
                        linewidths=1.0,
                        color=colors['excitatory'])
    
    # Plot inhibitory neurons with enhanced styling
    for i, nid in enumerate(inh_neurons):
        if nid in neuron_fires:
            ax.eventplot([t - start for t in neuron_fires[nid]], 
                        lineoffsets=len(ex_neurons) + i + 1, 
                        linelengths=0.9, 
                        linewidths=1.0,
                        color=colors['inhibitory'])

    # Add light grid lines for better readability
    for y in range(1, total_neurons + 1):
        ax.hlines(y, xmin=0, xmax=end - start, color='gray', linewidth=0.3, linestyles='--', zorder=0, alpha=0.5)

    ax.set_xlim(0, end - start)
    ax.set_ylim(0.5, total_neurons + 0.5)
    ax.set_yticks(range(0, total_neurons + 1, 5))

    if ylabel:
        ax.set_yticklabels([str(i) for i in range(0, total_neurons + 1, 5)])
        ax.set_ylabel('Neurons', fontweight='bold')
    else:
        ax.set_yticklabels([])

    ax.set_xlabel('Time (ms)', fontweight='bold')
    
    # Enhanced title appearance
    title_parts = title_label.split('|')
    if len(title_parts) > 1:
        panel = title_parts[0].strip()
        condition = title_parts[1].strip()
        ax.text(-0.15, 1.1, panel, transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='center', ha='center')
        ax.set_title(condition, loc='center', fontweight='bold', fontsize=13)
    else:
        ax.set_title(title_label, loc='center', fontweight='bold', fontsize=13)
    
    # Enhance spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('black')
    
    # Hide top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Enhanced tick parameters
    ax.tick_params(direction='out', length=4, width=0.8, colors='black')
    
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

# --- Create Main Figure with GridSpec for Precise Layout Control ---
fig = plt.figure(figsize=(10, 8.5))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1], 
              hspace=0.25, wspace=0.25)

# --- Define the four axes ---
ax_a = fig.add_subplot(gs[0, 0])  # Panel A (top-left)
ax_b = fig.add_subplot(gs[0, 1])  # Panel B (top-right)
ax_c = fig.add_subplot(gs[1, 0])  # Panel C (bottom-left)
ax_d = fig.add_subplot(gs[1, 1])  # Panel D (bottom-right)

# Top row: Raster plots
plot_spike_raster(ax_a, fires1, ex_neurons, inh_neurons, 2000, 10000, 'A | SPEC = 1000', ylabel=True)
plot_spike_raster(ax_b, fires2, ex_neurons, inh_neurons, 2000, 10000, 'B | SPEC = 500', ylabel=False)

# Bottom row: Panel C – Burst Counts with enhanced styling
bins_counts = np.arange(0.5, 30.5, 1)
ax_c.hist(burst_counts1, bins=bins_counts, alpha=0.7, 
          color=colors['SDC1000'], edgecolor='black', linewidth=0.8,
          label='SPEC = 1000')
ax_c.hist(burst_counts2, bins=bins_counts, alpha=0.7, 
          color=colors['SDC500'], edgecolor='black', linewidth=0.8,
          label='SPEC = 500')

ax_c.set_xlabel('Number of Bursts', fontweight='bold')
ax_c.set_ylabel('Neuron Count (log scale)', fontweight='bold')
ax_c.set_yscale('log')
ax_c.set_xlim(0, 30)
ax_c.set_yticks([1, 10, 100, 1000, 4000])
ax_c.get_yaxis().set_major_formatter(ScalarFormatter())

# Panel C label and styling
ax_c.text(-0.15, 1.1, 'C', transform=ax_c.transAxes,
          fontsize=18, fontweight='bold', va='center', ha='center')

# Enhance spines for panel C
for spine in ax_c.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color('black')

# Hide top and right spines
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)

# Enhanced grid for better readability
ax_c.grid(True, linestyle='--', linewidth=0.6, color='#CCCCCC', alpha=0.6)
ax_c.set_facecolor('white')  # Changed from '#F8F8F8' to 'white'

# Bottom row: Panel D – Burst Durations with enhanced styling
duration_bins = np.linspace(0, 1000, 200)
ax_d.hist(durations1, bins=duration_bins, alpha=0.7, 
          color=colors['SDC1000'], edgecolor='black', linewidth=0.8,
          label='SPEC = 1000')
ax_d.hist(durations2, bins=duration_bins, alpha=0.7, 
          color=colors['SDC500'], edgecolor='black', linewidth=0.8,
          label='SPEC = 500')

ax_d.set_xlabel('Burst Duration (ms)', fontweight='bold')
ax_d.set_ylabel('Count (log scale)', fontweight='bold')
ax_d.set_yscale('log')
ax_d.set_xlim(0, 1000)
ax_d.set_yticks([1, 10, 100, 1000, 2500])
ax_d.get_yaxis().set_major_formatter(ScalarFormatter())

# Panel D label and styling
ax_d.text(-0.15, 1.1, 'D', transform=ax_d.transAxes,
          fontsize=18, fontweight='bold', va='center', ha='center')

# Enhance legend for panel D
legend = ax_d.legend(frameon=True, framealpha=0.9, 
                    edgecolor='gray', fancybox=True, 
                    loc='upper right', fontsize=11)
legend.get_frame().set_linewidth(0.8)
for text in legend.get_texts():
    text.set_fontweight('bold')

# Enhance spines for panel D
for spine in ax_d.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color('black')

# Hide top and right spines
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)

# Enhanced grid for better readability
ax_d.grid(True, linestyle='--', linewidth=0.6, color='#CCCCCC', alpha=0.6)
ax_d.set_facecolor('white')  # Changed from '#F8F8F8' to 'white'

# Improve export quality
plt.tight_layout()
plt.savefig("Bursts_Enhanced.png", dpi=200, bbox_inches='tight')
plt.show()