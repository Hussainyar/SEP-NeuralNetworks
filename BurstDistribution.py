import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
from matplotlib.ticker import ScalarFormatter
from matplotlib.gridspec import GridSpec

# --- Publication-Quality Style Settings (Consistent with Figures 1 & 2) ---
plt.rcParams.update({
    'font.family': 'sans-serif',  # Arial for consistency
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
    'savefig.dpi': 300,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02
})

# --- Scientific Color Palette (Consistent with Figures 1 & 2) ---
colors = {
    'SEP1000': '#1f77b4',      # Professional blue for SEP=1000
    'SEP500': '#ff7f0e',       # Professional orange for SEP=500
    'non_sep_neurons': '#999999',  # Dark gray for non-SEP neurons
    'sep_neurons': '#1f77b4',      # Blue for SEP neurons
}

def read_firing_data(file_path):
    """Read firing pattern data from file"""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            time = float(parts[0])
            neuron_id = int(parts[1])
            data.append((time, neuron_id))
    return data

def read_position_data(file_path):
    """Read neuron position data from file"""
    positions = {}
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file):
            parts = line.split()
            if len(parts) >= 2:
                x = float(parts[0])
                y = float(parts[1])
                neuron_id = line_num + 1
                positions[neuron_id] = (x, y)
    return positions

def process_firing_data(data):
    """Convert firing data to a matrix format"""
    neuron_fires = {}
    for time, neuron_id in data:
        if neuron_id not in neuron_fires:
            neuron_fires[neuron_id] = []
        neuron_fires[neuron_id].append(time)

    for neuron_id in neuron_fires:
        neuron_fires[neuron_id].sort()

    max_fires = max(len(fires) for fires in neuron_fires.values()) if neuron_fires else 1
    matrix = np.zeros((5000, max_fires))

    for neuron_id, fires in neuron_fires.items():
        matrix[neuron_id - 1, :len(fires)] = fires

    return matrix

def count_bursts(spike_times, isi_threshold, min_spikes):
    """Count bursts in a sequence of spike times and calculate durations"""
    if len(spike_times) < min_spikes:
        return 0, []
    
    bursts = 0
    durations = []
    spike_times = np.array(spike_times)
    isi = np.diff(spike_times)
    burst_mask = isi <= isi_threshold
    burst_boundaries = np.where(np.concatenate(([burst_mask[0]], burst_mask[:-1] != burst_mask[1:], [True])))[0]
    burst_lengths = np.diff(burst_boundaries)[::2]
    burst_indices = np.where(burst_lengths >= (min_spikes))[0] * 2
    bursts = len(burst_indices)
    
    for idx in burst_indices:
        start = burst_boundaries[idx]
        end = burst_boundaries[idx + 1]
        duration = spike_times[end] - spike_times[start]
        durations.append(duration)
    
    return bursts, durations

def analyze_bursts_spatial(matrix, positions, isi_threshold, min_spikes):
    """Analyze bursts and their spatial distribution"""
    burst_data = {}
    
    for i in range(5000):
        neuron_id = i + 1
        if neuron_id in positions:
            spike_times = matrix[i, matrix[i] != 0]
            if len(spike_times) > 0:
                count, durations = count_bursts(spike_times, isi_threshold, min_spikes)
                x, y = positions[neuron_id]
                burst_data[neuron_id] = {
                    'x': x,
                    'y': y,
                    'burst_count': count,
                    'burst_durations': durations,
                    'total_spikes': len(spike_times)
                }
    
    return burst_data

def create_burst_heatmap(burst_data):
    """Create burst count heatmap data"""
    x_coords = np.array([data['x'] for data in burst_data.values()])
    y_coords = np.array([data['y'] for data in burst_data.values()])
    burst_counts = np.array([data['burst_count'] for data in burst_data.values()])
    
    xi = np.linspace(min(x_coords), max(x_coords), 100)
    yi = np.linspace(min(y_coords), max(y_coords), 100)
    xi, yi = np.meshgrid(xi, yi)
    
    zi = griddata((x_coords, y_coords), burst_counts, (xi, yi), 
                  method='linear', fill_value=0)
    zi = np.maximum(zi, 0)
    
    return xi, yi, zi, x_coords, y_coords, burst_counts

def main():
    # Parameters
    isi_threshold = 7.5
    min_spikes = 3
    
    # File paths
    position_file = 'position.dat'
    firing_files = ['firingpattern1000-2000.dat', 'firingpattern1500-2000.dat']
    labels = ['SEP=1000', 'SEP=500']
    
    # Load position data
    print("Loading position data...")
    positions = read_position_data(position_file)
    print(f"Loaded positions for {len(positions)} neurons")
    
    # Process both datasets
    all_burst_data = {}
    for firing_file, label in zip(firing_files, labels):
        print(f"\nProcessing {firing_file} ({label})...")
        firing_data = read_firing_data(firing_file)
        firing_matrix = process_firing_data(firing_data)
        burst_data = analyze_bursts_spatial(firing_matrix, positions, isi_threshold, min_spikes)
        all_burst_data[label] = burst_data
        
        # Print statistics
        burst_counts = np.array([data['burst_count'] for data in burst_data.values()])
        print(f"Statistics for {label}:")
        print(f"  Total Neurons: {len(burst_data)}")
        print(f"  Neurons with Bursts: {sum(1 for bc in burst_counts if bc > 0)}")
        print(f"  Mean Burst Count: {np.mean(burst_counts):.2f}")
        print(f"  Max Burst Count: {np.max(burst_counts)}")
    
    # Create the main figure with GridSpec for precise layout control
    fig = plt.figure(figsize=(11, 8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], 
                  hspace=0.35, wspace=0.35)
    
    # Calculate global min/max for consistent colorbar
    all_zi_values = []
    heatmap_data = {}
    
    for label, burst_data in all_burst_data.items():
        xi, yi, zi, x_coords, y_coords, burst_counts = create_burst_heatmap(burst_data)
        heatmap_data[label] = (xi, yi, zi, x_coords, y_coords, burst_counts)
        all_zi_values.extend(zi.flatten())
    
    vmin, vmax = np.min(all_zi_values), np.max(all_zi_values)
    
    # Plot A & B: SEP vs Non-SEP scatter plots (TOP ROW)
    # Define SEP ranges for each condition
    sep_ranges = {'SEP=1000': (1000, 2000), 'SEP=500': (1500, 2000)}
    
    for i, label in enumerate(labels):
        ax = fig.add_subplot(gs[0, i])
        
        # Get SEP range for this condition
        sep_start, sep_end = sep_ranges[label]
        
        # Separate SEP and non-SEP neurons based on neuron ID from positions
        sep_neurons = {}
        non_sep_neurons = {}
        
        for neuron_id, (x, y) in positions.items():
            if sep_start <= neuron_id <= sep_end:
                sep_neurons[neuron_id] = {'x': x, 'y': y}
            else:
                non_sep_neurons[neuron_id] = {'x': x, 'y': y}
        
        # Extract coordinates
        sep_x = [data['x'] for data in sep_neurons.values()]
        sep_y = [data['y'] for data in sep_neurons.values()]
        
        non_sep_x = [data['x'] for data in non_sep_neurons.values()]
        non_sep_y = [data['y'] for data in non_sep_neurons.values()]
        
        # Plot non-SEP neurons first (so they appear behind SEP neurons)
        scatter_non_sep = ax.scatter(non_sep_x, non_sep_y, s=8, 
                                   color=colors['non_sep_neurons'], alpha=0.2, 
                                   label='Non-SEP neurons',
                                   edgecolors='none')
        
        # Determine color for SEP neurons based on panel
        sep_color = colors['SEP1000'] if i == 0 else colors['SEP500']
        sep_label = 'SEP neurons'
        
        scatter_sep = ax.scatter(sep_x, sep_y, s=12, 
                                color=sep_color, alpha=0.35, 
                                edgecolors='none',
                                label=sep_label)
        
        # Panel label
        panel_label = '(A)' if i == 0 else '(B)'
        ax.text(-0.15, 1.08, panel_label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')
        
        # Add SEP titles
        title_text = 'SEP = 1000' if i == 0 else 'SEP = 500'
        ax.set_title(title_text, fontsize=10, fontweight='normal', pad=10)
        
        # Y-LABEL: Only on LEFT panel (A)
        if i == 0:
            ax.set_ylabel('Position (a.u.)', fontweight='normal', fontsize=10)
        else:
            ax.set_ylabel('')
        
        # NO x-labels on top row (will be at bottom)
        ax.set_xlabel('')
        
        # Set explicit tick locations for consistency
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        
        if i == 0:
            ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        else:
            ax.set_yticklabels([])
        
        ax.tick_params(labelsize=9, direction='out', length=3, width=0.8)
        
        # Legend only in panel A with improved styling
        if i == 0:
            legend = ax.legend(loc='upper left', fontsize=9, 
                             frameon=True, framealpha=0.9,
                             edgecolor='black', fancybox=False)
            legend.get_frame().set_linewidth(0.8)
            legend.get_frame().set_facecolor('white')
            # Make legend markers larger and more visible
            for handle in legend.legend_handles:
                handle.set_sizes([80])
        
        # Professional styling
        ax.grid(True, linestyle='-', linewidth=0.4, color='#CCCCCC', alpha=0.5, zorder=0)
        ax.set_facecolor('white')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        
        # Spine styling
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color('black')
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_linewidth(0.8)
        ax.spines['right'].set_linewidth(0.8)
        
        # Print statistics
        print(f"\n--- {label} SEP vs Non-SEP Statistics ---")
        print(f"SEP neuron range: {sep_start}-{sep_end}")
        print(f"SEP neurons: {len(sep_neurons)}")
        print(f"Non-SEP neurons: {len(non_sep_neurons)}")
    
    # Plot C: Heatmap for SEP=1000 (BOTTOM LEFT)
    ax_c = fig.add_subplot(gs[1, 0])
    xi, yi, zi, x_coords, y_coords, burst_counts = heatmap_data['SEP=1000']
    im_c = ax_c.imshow(zi, extent=[min(x_coords), max(x_coords), 
                              min(y_coords), max(y_coords)], 
                  origin='lower', cmap='hot', aspect='auto',
                  vmin=vmin, vmax=vmax, interpolation='bilinear')
    
    ax_c.text(-0.15, 1.08, '(C)', transform=ax_c.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='right')
    
    # X and Y labels at bottom row
    ax_c.set_xlabel('Position (a.u.)', fontweight='normal', fontsize=10)
    ax_c.set_ylabel('Position (a.u.)', fontweight='normal', fontsize=10)
    
    # Set explicit tick locations for consistency
    ax_c.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_c.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_c.set_xticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax_c.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax_c.tick_params(labelsize=9, direction='out', length=3, width=0.8)
    ax_c.set_facecolor('white')
    ax_c.set_xlim(-0.02, 1.02)
    ax_c.set_ylim(-0.02, 1.02)
    
    # Spine styling
    for spine in ax_c.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('black')
    ax_c.spines['top'].set_visible(True)
    ax_c.spines['right'].set_visible(True)
    ax_c.spines['top'].set_linewidth(0.8)
    ax_c.spines['right'].set_linewidth(0.8)
    
    # Plot D: Heatmap for SEP=500 (BOTTOM RIGHT)
    ax_d = fig.add_subplot(gs[1, 1])
    xi, yi, zi, x_coords, y_coords, burst_counts = heatmap_data['SEP=500']
    im_d = ax_d.imshow(zi, extent=[min(x_coords), max(x_coords), 
                              min(y_coords), max(y_coords)], 
                  origin='lower', cmap='hot', aspect='auto',
                  vmin=vmin, vmax=vmax, interpolation='bilinear')
    
    ax_d.text(-0.15, 1.08, '(D)', transform=ax_d.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='right')
    
    # X-label at bottom (shared with C)
    ax_d.set_xlabel('Position (a.u.)', fontweight='normal', fontsize=10)
    # NO y-label (already shown in C)
    ax_d.set_ylabel('')
    
    ax_d.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_d.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_d.set_xticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax_d.set_yticklabels([])
    ax_d.tick_params(labelsize=9, direction='out', length=3, width=0.8)
    ax_d.set_facecolor('white')
    ax_d.set_xlim(-0.02, 1.02)
    ax_d.set_ylim(-0.02, 1.02)
    
    # Spine styling
    for spine in ax_d.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('black')
    ax_d.spines['top'].set_visible(True)
    ax_d.spines['right'].set_visible(True)
    ax_d.spines['top'].set_linewidth(0.8)
    ax_d.spines['right'].set_linewidth(0.8)
    
    # Add colorbar with proper styling (only once, on the right side)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    divider = make_axes_locatable(ax_d)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im_d, cax=cax)
    cbar.set_label('Burst Count', rotation=90, labelpad=15, fontsize=10, fontweight='normal')
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(0.8)
    
    # Final layout adjustments
    plt.tight_layout()
    
    # Save with high quality for publication
    plt.savefig('Enhanced_Burst_Spatial_Distribution.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('FIG4.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    plt.show()
    
    print("\nHigh-quality publication figures saved!")
    print("Analysis complete!")

if __name__ == "__main__":
    main()