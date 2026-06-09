import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# This file uses the folder of the tensorflow logs during training to plot the learning 
# curves of the PPO on the source and the SAC model on the source using no Domain Randomization 
# techniques, using UDR, and using ADR as well as the sac model trained on the target domain.

def smooth_data(scalars, weight=0.85):
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def extract_tb_data(log_dir, scalar_name='rollout/success_rate'):
    event_acc = EventAccumulator(log_dir, size_guidance={'scalars': 0})
    event_acc.Reload()

    if scalar_name not in event_acc.Tags()['scalars']:
        print(f"Warning: '{scalar_name}' not found in {log_dir}")
        return None, None

    events = event_acc.Scalars(scalar_name)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    
    return steps, values

def main():
    base_log_dir = './tb_logs/' 
    
    runs_to_plot = {
        'PPO_source' : {'label': 'Training PPO on the source env', 'color': '#12b5cb'},
        'SAC_target': {'label': 'Training SAC on the target env', 'color': '#e83e8c'}, 
        'SAC_source_none': {'label': 'Training  SAC on the source env', 'color': '#6f42c1'}, 
        'SAC_source_udr': {'label': 'UDR SAC (Source + UDR)', 'color': '#fd7e14'},
        'SAC_source_adr': {'label': 'ADR SAC (Source + ADR)', 'color': '#28a745'}, 
    }

    plt.figure(figsize=(10, 6))

    for run_folder, config in runs_to_plot.items():
        full_path = os.path.join(base_log_dir, run_folder)
        
        if not os.path.exists(full_path):
            print(f"Directory not found: {full_path}. Skipping...")
            continue
            
        steps, values = extract_tb_data(full_path, scalar_name='rollout/success_rate')
        
        if steps is not None:
            plt.plot(steps, values, alpha=0.3, color=config['color'])
            
            smoothed_values = smooth_data(values, weight=0.85)
            plt.plot(steps, smoothed_values, label=config['label'], color=config['color'], linewidth=2)

    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title('Sim-to-Real Transfer Success Rate (PandaGym PushTask)', fontsize=14, fontweight='bold')
    
    plt.ylim(-0.05, 1.05)
    
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}k".format(int(x/1000))))
    
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    output_filename = 'sim_to_real_success_rates.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Plot successfully saved to {output_filename}")
    plt.show()

if __name__ == '__main__':
    main()