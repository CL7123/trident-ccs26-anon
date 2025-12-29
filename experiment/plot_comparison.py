import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set font to Times Roman with larger size
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # Increased base font size
plt.rcParams['font.weight'] = 'bold'  # Make text bolder
# For systems where Times New Roman is not available, use serif as fallback
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'serif']
# Make axes labels and ticks bolder
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# Data from the comparison analysis
# Format: {dataset: {method: {ef_search: (latency_ms, mrr/recall)}}}
data = {
    'SIFTSMALL': {
        'FAISS HNSW': {
            16: (0.001, 1.0000),   # From line 24-25: QPS=90511.5, MRR@10=1.0000
            32: (0.001, 1.0000),   # From line 42-43: QPS=68189.0, MRR@10=1.0000
            64: (0.001, 1.0000),   # From line 60-61: QPS=120456.7, MRR@10=1.0000
            128: (0.002, 1.0000)   # From line 78-79: QPS=53842.2, MRR@10=1.0000
        },
        'TridentSearcher': {
            16: (3.31, 0.910),
            32: (4.60, 0.950),
            64: (6.51, 0.980),
            128: (9.98, 0.980)
        }
    },
    'LAION': {
        'FAISS HNSW': {
            16: (0.077, 1.0000),   # From line 192-193: QPS=13026.7, MRR@10=1.0000
            32: (0.117, 1.0000),   # From line 204-205: QPS=8514.4, MRR@10=1.0000
            64: (0.203, 1.0000),   # From line 216-217: QPS=4937.8, MRR@10=1.0000
            128: (0.319, 1.0000)   # From line 228-229: QPS=3135.7, MRR@10=1.0000
        },
        'TridentSearcher': {
            16: (5.15, 0.980),
            32: (7.15, 0.990),
            64: (11.58, 1.000),
            128: (18.74, 1.000)
        }
    },
    'TRIPCLICK': {
        'FAISS HNSW': {
            18: (0.335, 0.8701),   # From line 252-254: QPS=3503.8, MRR@10=0.8701
            36: (0.562, 0.8743),   # From line 264-265: QPS=2089.1, MRR@10=0.8743
            72: (0.986, 0.8750),   # From line 276-277: QPS=1191.5, MRR@10=0.8750
            144: (1.815, 0.8750)   # From line 288-289: QPS=647.5, MRR@10=0.8750
        },
        'TridentSearcher': {
            18: (11.71, 0.9333),
            36: (20.26, 0.9333),
            72: (34.97, 0.9333),
            144: (59.58, 0.9333)
        }
    },
    'NFCORPUS': {
        'FAISS HNSW': {
            16: (0.002, 0.4751),   # From line 108-109: QPS=203754.0, MRR@10=0.4751
            32: (0.007, 0.4888),   # From line 126-127: QPS=47709.5, MRR@10=0.4888
            64: (0.006, 0.4940),   # From line 144-145: QPS=57507.4, MRR@10=0.4940
            128: (0.009, 0.4969)   # From line 162-163: QPS=36135.6, MRR@10=0.4969
        },
        'TridentSearcher': {
            16: (2.65, 0.2582),
            32: (4.22, 0.2687),
            64: (6.40, 0.2682),
            128: (10.56, 0.2882)
        }
    }
}

# Create the figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

# Flatten axes for easier iteration
axes = axes.flatten()

# Define colors and markers using the specified color scheme
# Convert RGB to hex
colors = {
    'FAISS HNSW': '#386641',      # rgb(56, 102, 65) - 深绿色
    'TridentSearcher': '#F97A00'   # rgb(249, 122, 0) - 橙色
}
markers = {'FAISS HNSW': 'o', 'TridentSearcher': 's'}

# Plot each dataset
for idx, (dataset_name, dataset_data) in enumerate(data.items()):
    ax = axes[idx]
    
    for method_name, method_data in dataset_data.items():
        # Extract latencies and MRR values
        ef_searches = sorted(method_data.keys())
        latencies = [method_data[ef][0] for ef in ef_searches]
        mrr_values = [method_data[ef][1] for ef in ef_searches]
        
        # Plot the line
        ax.plot(latencies, mrr_values, 
                color=colors[method_name], 
                marker=markers[method_name], 
                markersize=8,
                linewidth=2,
                label=method_name,
                alpha=0.8)
        
        # Add ef_search labels next to points with better visibility
        for i, ef in enumerate(ef_searches):
            # Alternate label positions to avoid overlap
            if i % 2 == 0:
                offset = (8, 8)
            else:
                offset = (8, -15)
            
            ax.annotate(f'ef={ef}', 
                       (latencies[i], mrr_values[i]), 
                       textcoords="offset points", 
                       xytext=offset, 
                       fontsize=10,  # Increased to 10
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white', 
                                edgecolor=colors[method_name],
                                alpha=1.0),  # Fully opaque
                       color=colors[method_name])
    
    # Set logarithmic scale for x-axis due to large difference in latencies
    ax.set_xscale('log')
    
    # Set labels and title
    ax.set_xlabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('MRR@10', fontsize=12, fontweight='bold')
    ax.set_title(f'{dataset_name}', fontsize=13, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set y-axis limits based on data range
    y_min = min([v[1] for method in dataset_data.values() for v in method.values()]) * 0.95
    y_max = min(1.0, max([v[1] for method in dataset_data.values() for v in method.values()]) * 1.05)
    
    # If max value is 1.0, add some padding but don't show ticks above 1.0
    if y_max == 1.0:
        ax.set_ylim(y_min, 1.05)  # Add 5% padding above 1.0
        # Set y-ticks to not exceed 1.0
        current_ticks = ax.get_yticks()
        new_ticks = [tick for tick in current_ticks if tick <= 1.0]
        ax.set_yticks(new_ticks)
    else:
        ax.set_ylim(y_min, y_max)
    
    # Add legend
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)

# Adjust layout
plt.tight_layout()

# Save as PDF
output_path = '~/trident/experiment/hnsw_vs_trident_comparison.pdf'
plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')

# Also save as PNG for preview
png_path = '~/trident/experiment/hnsw_vs_trident_comparison.png'
plt.savefig(png_path, format='png', dpi=150, bbox_inches='tight')

print(f"Comparison plot saved to:")
print(f"  PDF: {output_path}")
print(f"  PNG: {png_path}")

# Print summary statistics
print("\nSummary Statistics:")
print("="*60)
for dataset in data:
    print(f"\n{dataset}:")
    faiss_latencies = [v[0] for v in data[dataset]['FAISS HNSW'].values()]
    trident_latencies = [v[0] for v in data[dataset]['TridentSearcher'].values()]
    speedup = np.mean(trident_latencies) / np.mean(faiss_latencies)
    
    faiss_mrr = np.mean([v[1] for v in data[dataset]['FAISS HNSW'].values()])
    trident_mrr = np.mean([v[1] for v in data[dataset]['TridentSearcher'].values()])
    
    print(f"  Average speedup of FAISS over Trident: {speedup:.1f}x")
    print(f"  Average MRR - FAISS: {faiss_mrr:.4f}, Trident: {trident_mrr:.4f}")
    print(f"  MRR improvement of Trident: {(trident_mrr/faiss_mrr - 1)*100:.1f}%")