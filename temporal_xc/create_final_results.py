"""Create comprehensive final results visualization with actual token recovery data."""

import matplotlib.pyplot as plt
import numpy as np
import json

# Actual results from our experiments
results = {
    1: {
        'k': 1,
        'train_r2': 0.968,
        'test_r2': 0.846,
        'mean_cosine': 0.977,
        'actual_accuracy': 0.35,  # 35%
        'estimated_accuracy': 0.86,
        'correct': 7,
        'total': 20
    },
    2: {
        'k': 2,
        'train_r2': 0.941,
        'test_r2': 0.797,
        'mean_cosine': 0.939,
        'actual_accuracy': 0.30,  # 30%
        'estimated_accuracy': 0.78,
        'correct': 6,
        'total': 20
    },
    4: {
        'k': 4,
        'train_r2': 0.876,
        'test_r2': 0.687,
        'mean_cosine': 0.908,
        'actual_accuracy': 0.05,  # 5%
        'estimated_accuracy': 0.72,
        'correct': 1,
        'total': 20
    }
}

# Create comprehensive figure
fig = plt.figure(figsize=(16, 10))

k_values = sorted(results.keys())
train_r2 = [results[k]['train_r2'] for k in k_values]
test_r2 = [results[k]['test_r2'] for k in k_values]
cosine_sim = [results[k]['mean_cosine'] for k in k_values]
actual_acc = [results[k]['actual_accuracy'] for k in k_values]
estimated_acc = [results[k]['estimated_accuracy'] for k in k_values]

# Plot 1: R² Scores
ax1 = plt.subplot(2, 3, 1)
ax1.plot(k_values, train_r2, 'o-', label='Train R²', linewidth=2, markersize=8, color='blue')
ax1.plot(k_values, test_r2, 's-', label='Test R²', linewidth=2, markersize=8, color='red')
ax1.set_xlabel('k (tokens ahead)', fontsize=12)
ax1.set_ylabel('R² Score', fontsize=12)
ax1.set_title('Train vs Test R² Scores', fontsize=14)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.6, 1.0])
ax1.set_xticks(k_values)

# Add annotations
for k, test in zip(k_values, test_r2):
    ax1.annotate(f'{test:.3f}', xy=(k, test), xytext=(5, -10),
                textcoords='offset points', fontsize=9, color='red')

# Plot 2: Prediction Quality
ax2 = plt.subplot(2, 3, 2)
ax2.plot(k_values, test_r2, 'o-', label='Test R²', linewidth=2, markersize=8, color='red')
ax2.plot(k_values, cosine_sim, '^-', label='Cosine Similarity', linewidth=2, markersize=8, color='purple')
ax2.set_xlabel('k (tokens ahead)', fontsize=12)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Prediction Quality Metrics', fontsize=14)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.6, 1.0])
ax2.set_xticks(k_values)

# Plot 3: Accuracy Comparison
ax3 = plt.subplot(2, 3, 3)
ax3.plot(k_values, estimated_acc, 'd-', label='Estimated Accuracy', linewidth=2, markersize=10, color='orange')
ax3.plot(k_values, actual_acc, 'o-', label='ACTUAL Accuracy', linewidth=2.5, markersize=12, color='green')
ax3.set_xlabel('k (tokens ahead)', fontsize=12)
ax3.set_ylabel('Accuracy', fontsize=12)
ax3.set_title('⭐ Actual vs Estimated Token Recovery', fontsize=14, fontweight='bold')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, 1.0])
ax3.set_xticks(k_values)

# Add annotations with emphasis on actual accuracy
for k, acc in zip(k_values, actual_acc):
    ax3.annotate(f'{acc:.0%}', xy=(k, acc), xytext=(5, 10),
                textcoords='offset points', fontsize=12, color='green', fontweight='bold')
for k, est in zip(k_values, estimated_acc):
    ax3.annotate(f'{est:.0%}', xy=(k, est), xytext=(5, -15),
                textcoords='offset points', fontsize=9, color='orange', alpha=0.7)

# Plot 4: Summary Table
ax4 = plt.subplot(2, 3, 4)
ax4.axis('tight')
ax4.axis('off')

# Create table data
table_data = [
    ['Metric', 'k=1', 'k=2', 'k=4'],
    ['Train R²', f'{results[1]["train_r2"]:.3f}', f'{results[2]["train_r2"]:.3f}', f'{results[4]["train_r2"]:.3f}'],
    ['Test R²', f'{results[1]["test_r2"]:.3f}', f'{results[2]["test_r2"]:.3f}', f'{results[4]["test_r2"]:.3f}'],
    ['Cosine Sim', f'{results[1]["mean_cosine"]:.3f}', f'{results[2]["mean_cosine"]:.3f}', f'{results[4]["mean_cosine"]:.3f}'],
    ['Est. Accuracy', f'{results[1]["estimated_accuracy"]:.0%}', f'{results[2]["estimated_accuracy"]:.0%}', f'{results[4]["estimated_accuracy"]:.0%}'],
    ['ACTUAL Acc.', f'{results[1]["actual_accuracy"]:.0%}', f'{results[2]["actual_accuracy"]:.0%}', f'{results[4]["actual_accuracy"]:.0%}'],
    ['Correct/Total', f'{results[1]["correct"]}/{results[1]["total"]}', f'{results[2]["correct"]}/{results[2]["total"]}', f'{results[4]["correct"]}/{results[4]["total"]}']
]

table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)

# Style the header row
for i in range(4):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight actual accuracy row
for i in range(4):
    table[(5, i)].set_facecolor('#FFE4B5')
    table[(5, i)].set_text_props(weight='bold')

ax4.set_title('Summary Results Table', fontsize=14, pad=20)

# Plot 5: R² vs Accuracy Gap Analysis
ax5 = plt.subplot(2, 3, 5)
width = 0.35
x = np.arange(len(k_values))
ax5.bar(x - width/2, test_r2, width, label='Test R²', color='steelblue', alpha=0.8)
ax5.bar(x + width/2, actual_acc, width, label='Actual Accuracy', color='forestgreen', alpha=0.8)
ax5.set_xlabel('k (tokens ahead)', fontsize=12)
ax5.set_ylabel('Score', fontsize=12)
ax5.set_title('R² vs Actual Accuracy Gap', fontsize=14)
ax5.set_xticks(x)
ax5.set_xticklabels(k_values)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Add gap annotations
for i, k in enumerate(k_values):
    gap = test_r2[i] - actual_acc[i]
    ax5.annotate(f'Gap: {gap:.1%}', xy=(i, (test_r2[i] + actual_acc[i])/2),
                ha='center', fontsize=10, fontweight='bold', color='red')

# Plot 6: Key Insights Text
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

insights_text = """KEY INSIGHTS

✅ ACTUAL TOKEN RECOVERY RESULTS:
• k=1: 35% accuracy (7/20 correct)
• k=2: 30% accuracy (6/20 correct)
• k=4: 5% accuracy (1/20 correct)

📊 R² vs Accuracy Gap:
• High R² (69-85%) but lower token accuracy
• Gap shows activation prediction ≠ token recovery
• Small errors in 4096-dim space → wrong tokens

🎯 Significance:
• 35% accuracy is ~35,000x better than random!
• Random baseline: ~0.001% (100k vocab)
• Probes capture significant temporal structure

💡 Implications:
• Temporal structure exists in residual streams
• Structure degrades with distance (k↑ → accuracy↓)
• SAE features may better preserve this structure"""

ax6.text(0.05, 0.95, insights_text, transform=ax6.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Temporal Probe: Complete Token Recovery Results\n'
             'Actual Accuracy Evaluation on DeepSeek-R1-Distill-Llama-8B',
             fontsize=16, y=1.02, fontweight='bold')

plt.tight_layout()

# Save the figure
plt.savefig('large_files/viz/final_actual_token_recovery_results.png', dpi=150, bbox_inches='tight')
print("📊 Saved comprehensive results to large_files/viz/final_actual_token_recovery_results.png")

# Save results to JSON
with open('large_files/viz/final_actual_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("💾 Saved results to large_files/viz/final_actual_results.json")

# Print summary to console
print("\n" + "="*60)
print("FINAL ACTUAL TOKEN RECOVERY RESULTS")
print("="*60)
print(f"{'k':<5} {'Train R²':<10} {'Test R²':<10} {'ACTUAL Accuracy':<20}")
print("-" * 45)
for k in k_values:
    r = results[k]
    acc_str = f"{r['actual_accuracy']:.0%} ({r['correct']}/{r['total']})"
    print(f"{k:<5} {r['train_r2']:<10.3f} {r['test_r2']:<10.3f} {acc_str:<20}")

print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)
print("1. Successfully evaluated ACTUAL token recovery accuracy")
print("2. k=1: 35% accuracy - significantly above random baseline")
print("3. k=2: 30% accuracy - slight degradation from k=1")
print("4. k=4: 5% accuracy - substantial drop for longer distances")
print("5. Gap between R² and accuracy confirms hypothesis:")
print("   - Probes learn good representations (high R²)")
print("   - But token prediction requires extreme precision")

plt.show()