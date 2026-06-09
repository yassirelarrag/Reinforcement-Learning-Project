import numpy as np
import matplotlib.pyplot as plt
import os

# This script uses the data saved during training of the episodic rewards of the 3 models,
# REINFORCE without a baseline, REINFORCE with a baseline, and Actor Critic.
# The data is loaded from the npz files ("results_<ALGORITHM>.npz")
def load_metrics(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing data file: '{filepath}'. Please run training for this variant first.")
    data = np.load(filepath)
    return data['rewards'], float(data['time'])

def main():
    try:
        r_no_b, t_no_b = load_metrics("results_REINFORCE_0.0.npz")
        r_with_b, t_with_b = load_metrics("results_REINFORCE_20.0.npz")
        r_ac, t_ac = load_metrics("results_AC.npz")
    except FileNotFoundError as e:
        print(e)
        return

    window_size = 20 
    
    colors = {'no_b': '#dc3545', 'with_b': '#28a745', 'ac': '#007bff'}
    
    datasets = [r_no_b, r_with_b, r_ac]
    labels = ['REINFORCE (No Baseline)', 'REINFORCE (Baseline b=20)', 'Actor-Critic']
    color_list = [colors['no_b'], colors['with_b'], colors['ac']]


    # FIGURE 1: All Learning Curves
    plt.figure(figsize=(10, 6))
    
    for data, label, color in zip(datasets, labels, color_list):
        plt.plot(data, alpha=0.15, color=color, linestyle='-')
        if len(data) >= window_size:
            mv_avg = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
            padded_mv = np.concatenate((np.full(window_size-1, np.nan), mv_avg))
            plt.plot(padded_mv, label=label, color=color, linewidth=2)
        else:
            plt.plot(data, label=label, color=color, linewidth=2)

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Return', fontsize=12)
    plt.title('Algorithmic Reward Evolution on Hopper-v4', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    curves_filename = 'hopper_all_learning_curves.png'
    plt.savefig(curves_filename, dpi=300)
    print(f"Learning curves plot saved successfully as: '{curves_filename}'")
    plt.close()

    # FIGURE 2: Cost Bar Chart
    plt.figure(figsize=(8, 6))
    
    bars = plt.bar(['REINFORCE\n(b=0)', 'REINFORCE\n(b=20)', 'Actor-Critic'], 
                   [t_no_b, t_with_b, t_ac], 
                   color=color_list, width=0.5, edgecolor='black', alpha=0.85)
    
    plt.ylabel('Total Execution Time (seconds)', fontsize=12)
    plt.title('Computational Time Consumption Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    max_time = max(t_no_b, t_with_b, t_ac)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + (max_time * 0.01),
                 f'{height:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    time_filename = 'hopper_time_comparison.png'
    plt.savefig(time_filename, dpi=300)
    print(f"Time comparison plot saved successfully as: '{time_filename}'")
    
    plt.show()

if __name__ == '__main__':
    main()