import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# Fixed GATcut parameter sensitivity heatmap
def plot_gatcut_heatmap_fixed():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Data definition
    epsilon_values = [0.0, 0.1, 0.15, 0.2]
    top_k_values = [5, 10, 15]
    epsilon_labels = ['0.0', '0.1', '0.15', '0.2']
    top_k_labels = ['5', '10', '15']
    
    # Test accuracy data
    cora_data = np.array([
        [0.8150, 0.8000, 0.6250],  # epsilon=0.0
        [0.8210, 0.8140, 0.7190],  # epsilon=0.1
        [0.8250, 0.8090, 0.7760],  # epsilon=0.15
        [0.8120, 0.7950, 0.7740],  # epsilon=0.2
    ])
    
    citeseer_data = np.array([
        [0.6820, 0.5920, 0.5100],  # epsilon=0.0
        [0.7000, 0.6430, 0.6310],  # epsilon=0.1
        [0.7120, 0.6560, 0.5860],  # epsilon=0.15
        [0.6590, 0.6900, 0.6080],  # epsilon=0.2
    ])
    
    # Use seaborn heatmap for better stability
    sns.heatmap(cora_data, 
                annot=True, 
                fmt='.3f', 
                cmap='YlOrRd',
                xticklabels=top_k_labels,
                yticklabels=epsilon_labels,
                ax=ax1,
                cbar_kws={'label': 'Test Accuracy'},
                annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    
    ax1.set_title('GATcut Performance on Cora Dataset', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Top-k Value', fontsize=12)
    ax1.set_ylabel('Epsilon Threshold', fontsize=12)
    
    sns.heatmap(citeseer_data, 
                annot=True, 
                fmt='.3f', 
                cmap='YlOrRd',
                xticklabels=top_k_labels,
                yticklabels=epsilon_labels,
                ax=ax2,
                cbar_kws={'label': 'Test Accuracy'},
                annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    
    ax2.set_title('GATcut Performance on Citeseer Dataset', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Top-k Value', fontsize=12)
    ax2.set_ylabel('Epsilon Threshold', fontsize=12)
    
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15, wspace=0.4)
    plt.savefig('gatcut_sensitivity_heatmap_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()

# Fixed GATbfs parameter trend plot
def plot_gatbfs_trends_fixed():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Cora dataset
    epsilon_values = [0.1, 0.15, 0.2, 0.25, 0.3]
    hops_2 = [0.8120, 0.8110, 0.8110, 0.8190, 0.8100]
    hops_3 = [0.8160, 0.7960, 0.8140, 0.8280, 0.8210]
    
    ax1.plot(epsilon_values, hops_2, 'o-', label='max_hops=2', linewidth=2.5, markersize=8, color='#2E86AB')
    ax1.plot(epsilon_values, hops_3, 's-', label='max_hops=3', linewidth=2.5, markersize=8, color='#A23B72')
    ax1.set_title('GATbfs Performance Trends on Cora Dataset', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epsilon Threshold', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.78, 0.83)
    
    # Add value annotations
    for i, (eps, v2, v3) in enumerate(zip(epsilon_values, hops_2, hops_3)):
        ax1.annotate(f'{v2:.3f}', (eps, v2), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        ax1.annotate(f'{v3:.3f}', (eps, v3), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    # Citeseer dataset
    citeseer_hops_2 = [0.7170, 0.7320, 0.7170, 0.7310, 0.7150]
    citeseer_hops_3 = [0.7250, 0.7210, 0.7160, 0.7240, 0.7270]
    
    ax2.plot(epsilon_values, citeseer_hops_2, 'o-', label='max_hops=2', linewidth=2.5, markersize=8, color='#2E86AB')
    ax2.plot(epsilon_values, citeseer_hops_3, 's-', label='max_hops=3', linewidth=2.5, markersize=8, color='#A23B72')
    ax2.set_title('GATbfs Performance Trends on Citeseer Dataset', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epsilon Threshold', fontsize=12)
    ax2.set_ylabel('Test Accuracy', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.71, 0.73)
    
    # Add value annotations
    for i, (eps, v2, v3) in enumerate(zip(epsilon_values, citeseer_hops_2, citeseer_hops_3)):
        ax2.annotate(f'{v2:.3f}', (eps, v2), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        ax2.annotate(f'{v3:.3f}', (eps, v3), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15, wspace=0.3)
    plt.savefig('gatbfs_trends_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()

# Fixed method performance comparison
def plot_method_comparison_fixed():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    methods = ['GAT', 'GAT++(cut)', 'GAT++(bfs)', 'GAT++(dsu)', 'GATv2', 'SuperGAT-MX', 'SuperGAT-SD']
    methods_short = ['GAT', 'Cut', 'BFS', 'DSU', 'GATv2', 'MX', 'SD']
    
    # Performance data
    cora_scores = [0.8180, 0.8250, 0.8190, 0.8280, 0.8460, 0.8160, 0.8210]
    citeseer_scores = [0.7270, 0.7120, 0.7170, 0.7450, 0.7830, 0.7280, 0.7170]
    
    x = np.arange(len(methods))
    width = 0.35
    
    # Better colors
    colors_cora = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6', '#34495e', '#1abc9c']
    colors_citeseer = ['#5dade2', '#ec7063', '#f7dc6f', '#58d68d', '#bb8fce', '#85929e', '#76d7c4']
    
    # Bar chart
    bars1 = ax1.bar(x - width/2, cora_scores, width, label='Cora', alpha=0.8, color=colors_cora)
    bars2 = ax1.bar(x + width/2, citeseer_scores, width, label='Citeseer', alpha=0.8, color=colors_citeseer)
    
    ax1.set_title('Performance Comparison Across Methods and Datasets', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_xlabel('Methods', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods_short, rotation=0)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0.65, 0.87)
    
    # Add value annotations
    for bar, score in zip(bars1, cora_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar, score in zip(bars2, citeseer_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # GAT++ series comparison
    gat_plus_methods = ['Cut', 'BFS', 'DSU']
    gat_plus_cora = [0.8250, 0.8190, 0.8280]
    gat_plus_citeseer = [0.7120, 0.7170, 0.7450]
    
    x2 = np.arange(len(gat_plus_methods))
    bars3 = ax2.bar(x2 - width/2, gat_plus_cora, width, label='Cora', alpha=0.8, 
                   color=['#e74c3c', '#f39c12', '#2ecc71'])
    bars4 = ax2.bar(x2 + width/2, gat_plus_citeseer, width, label='Citeseer', alpha=0.8,
                   color=['#ec7063', '#f7dc6f', '#58d68d'])
    
    # Add GAT baseline
    ax2.axhline(y=0.8180, color='#3498db', linestyle='--', alpha=0.8, linewidth=2, label='GAT (Cora)')
    ax2.axhline(y=0.7270, color='#5dade2', linestyle='--', alpha=0.8, linewidth=2, label='GAT (Citeseer)')
    
    ax2.set_title('GAT++ Variants Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Test Accuracy', fontsize=12)
    ax2.set_xlabel('GAT++ Variants', fontsize=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(gat_plus_methods)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0.70, 0.85)
    
    # Add value annotations
    for bar, score in zip(bars3, gat_plus_cora):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar, score in zip(bars4, gat_plus_citeseer):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.15, wspace=0.25)
    plt.savefig('method_comparison_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()

# Fixed improvement analysis
def plot_improvement_analysis_fixed():
    fig = plt.figure(figsize=(16, 7))
    
    # Create subplot layout
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Improvement percentage relative to GAT
    methods = ['GAT++(cut)', 'GAT++(bfs)', 'GAT++(dsu)']
    methods_short = ['Cut', 'BFS', 'DSU']
    cora_improvements = [
        (0.8250 - 0.8180) / 0.8180 * 100,  # +0.86%
        (0.8190 - 0.8180) / 0.8180 * 100,  # +0.12%
        (0.8280 - 0.8180) / 0.8180 * 100,  # +1.22%
    ]
    citeseer_improvements = [
        (0.7120 - 0.7270) / 0.7270 * 100,  # -2.06%
        (0.7170 - 0.7270) / 0.7270 * 100,  # -1.38%
        (0.7450 - 0.7270) / 0.7270 * 100,  # +2.47%
    ]
    
    x = np.arange(len(methods))
    width = 0.35
    
    # Set colors based on positive/negative values
    cora_colors = ['green' if i > 0 else 'red' for i in cora_improvements]
    citeseer_colors = ['green' if i > 0 else 'red' for i in citeseer_improvements]
    
    bars1 = ax1.bar(x - width/2, cora_improvements, width, label='Cora', 
                   color=cora_colors, alpha=0.7)
    bars2 = ax1.bar(x + width/2, citeseer_improvements, width, label='Citeseer',
                   color=citeseer_colors, alpha=0.7)
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_title('GAT++ Performance Improvement over Original GAT', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Improvement Percentage (%)', fontsize=12)
    ax1.set_xlabel('GAT++ Variants', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods_short)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value annotations
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax1.text(bar1.get_x() + bar1.get_width()/2., height1 + (0.2 if height1 > 0 else -0.2),
                f'{height1:.2f}%', ha='center', va='bottom' if height1 > 0 else 'top', 
                fontweight='bold', fontsize=10)
        ax1.text(bar2.get_x() + bar2.get_width()/2., height2 + (0.2 if height2 > 0 else -0.2),
                f'{height2:.2f}%', ha='center', va='bottom' if height2 > 0 else 'top', 
                fontweight='bold', fontsize=10)
    
    # Fusion method comparison (GATdsu specific analysis)
    fusion_methods = ['add', 'concat', 'gate']
    epsilon_05_cora = [0.8250, 0.8180, 0.8350]
    epsilon_1_cora = [0.7850, 0.8160, 0.8280]
    epsilon_15_cora = [0.8190, 0.8060, 0.8400]
    
    x2 = np.arange(len(fusion_methods))
    ax2.plot(x2, epsilon_05_cora, 'o-', label='ε=0.05', linewidth=2.5, markersize=8, color='#e74c3c')
    ax2.plot(x2, epsilon_1_cora, 's-', label='ε=0.1', linewidth=2.5, markersize=8, color='#3498db')
    ax2.plot(x2, epsilon_15_cora, '^-', label='ε=0.15', linewidth=2.5, markersize=8, color='#2ecc71')
    
    ax2.set_title('GATdsu Fusion Strategy Comparison (Cora)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Test Accuracy', fontsize=12)
    ax2.set_xlabel('Fusion Method', fontsize=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(fusion_methods)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.78, 0.85)
    
    # Add value annotations
    for i, method in enumerate(fusion_methods):
        ax2.annotate(f'{epsilon_05_cora[i]:.3f}', (i, epsilon_05_cora[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        ax2.annotate(f'{epsilon_1_cora[i]:.3f}', (i, epsilon_1_cora[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        ax2.annotate(f'{epsilon_15_cora[i]:.3f}', (i, epsilon_15_cora[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    plt.savefig('improvement_analysis_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()

# Fixed radar chart
def plot_radar_comparison_fixed():
    from math import pi
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Methods and datasets
    methods = ['GAT', 'GAT++(dsu)', 'GATv2']
    datasets = ['Cora', 'Citeseer', 'Pubmed']
    
    # Performance data (normalized to 0-1)
    data = {
        'GAT': [0.8180, 0.7270, 0.7750],
        'GAT++(dsu)': [0.8280, 0.7450, 0.7750],  # Pubmed uses GAT value
        'GATv2': [0.8460, 0.7830, 0.7830]
    }
    
    # Normalize
    max_vals = [max(data[method][i] for method in methods) for i in range(3)]
    min_vals = [min(data[method][i] for method in methods) for i in range(3)]
    
    normalized_data = {}
    for method in methods:
        normalized_data[method] = [(data[method][i] - min_vals[i]) / (max_vals[i] - min_vals[i]) 
                                  for i in range(3)]
    
    # Angles
    angles = [n / float(len(datasets)) * 2 * pi for n in range(len(datasets))]
    angles += angles[:1]  # Close the plot
    
    colors = ['blue', 'red', 'green']
    
    for i, method in enumerate(methods):
        values = normalized_data[method]
        values += values[:1]  # Close the plot
        
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('Method Performance Radar Chart\n(Normalized)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True)
    
    plt.savefig('radar_comparison_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()

# Execute all plotting functions
if __name__ == "__main__":
    print("Generating GATcut parameter sensitivity heatmap...")
    plot_gatcut_heatmap_fixed()
    
    print("Generating GATbfs parameter trend plots...")
    plot_gatbfs_trends_fixed()
    
    print("Generating method performance comparison...")
    plot_method_comparison_fixed()
    
    print("Generating improvement analysis...")
    plot_improvement_analysis_fixed()
    
    print("Generating radar comparison chart...")
    plot_radar_comparison_fixed()
    
    print("All plots generated successfully!")