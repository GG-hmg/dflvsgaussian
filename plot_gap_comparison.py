"""Plot CIFAR10 DFL comparison chart for 3 Gap values."""
import os, re
import matplotlib.pyplot as plt
import numpy as np

def parse_output(output_path):
    epochs, accs, antis = [], [], []
    if not os.path.exists(output_path):
        return epochs, accs, antis
    with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    acc_match = re.search(r'平均准确率历程:\s*\[([^\]]+)\]', content)
    if acc_match:
        s = acc_match.group(1).replace('%','').replace("'","")
        accs = [float(x.strip()) for x in s.split(',') if x.strip()]
        epochs = list(range(1, len(accs)+1))
    anti_match = re.search(r'抗梯度反演能力历程:\s*\[([^\]]+)\]', content)
    if anti_match:
        s = anti_match.group(1).replace('%','').replace("'","")
        antis = [float(x.strip()) for x in s.split(',') if x.strip()]
    final_acc = re.search(r'最终准确率:\s*([\d.]+)%', content)
    final_anti = re.search(r'最终抗梯度反演能力:\s*([\d.]+)', content)
    return epochs, accs, antis, float(final_acc.group(1)) if final_acc else 0, float(final_anti.group(1)) if final_anti else 0

gaps = [6, 12, 24]
colors = {6: '#e74c3c', 12: '#2ecc71', 24: '#3498db'}
results = {}

for gap in gaps:
    path = f'experiment_results/CIFAR10_dfl_run{gap}_output.txt'
    epochs, accs, antis, final_acc, final_anti = parse_output(path)
    results[gap] = (epochs, accs, antis, final_acc, final_anti)
    print(f"Gap={gap}: acc={final_acc:.2f}%, anti={final_anti:.4f}, epochs={len(accs)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
for gap in gaps:
    ep, acc, _, fa, _ = results[gap]
    if ep:
        ax1.plot(ep, acc, color=colors[gap], linewidth=1.2, label=f'Gap={gap} (final={fa:.1f}%)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('CIFAR10 DFL - Accuracy vs Gap')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
for gap in gaps:
    ep, _, anti, _, fanti = results[gap]
    if ep:
        ax2.plot(ep, anti, color=colors[gap], linewidth=1.2, label=f'Gap={gap} (final={fanti:.4f})')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Anti-Inversion Ability')
ax2.set_title('CIFAR10 DFL - Privacy vs Gap')
ax2.legend()
ax2.grid(True, alpha=0.3)

param_text = "Params: sigma=0.10, clip=2.0, alpha=0.85, burn_in=2048  |  CIFAR10 DFL (chaotic_factor=1.0)"
fig.text(0.5, 0.02, param_text, ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout(rect=[0, 0.05, 1, 1])
out_path = 'experiment_results/cifar10_gap_comparison.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nChart saved: {out_path}")
plt.close()
