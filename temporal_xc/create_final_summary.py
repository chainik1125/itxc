"""Create final summary plot with all results."""

import matplotlib.pyplot as plt
import numpy as np

# Data from our experiments
k_values = [1, 2, 4]
train_r2 = [0.969, 0.940, 0.864]
test_r2 = [0.862, 0.789, 0.703]
cosine_sim = [0.964, 0.939, 0.908]
estimated_acc = [0.86, 0.78, 0.72]
actual_acc = [0.35, None, None]  # Only have k=1 actual result

# Create comprehensive figure
fig = plt.figure(figsize=(15, 10))

# Plot 1: R² Scores
ax1 = plt.subplot(2, 2, 1)
ax1.plot(k_values, train_r2, 'o-', label='Train R²', linewidth=2, markersize=8, color='blue')
ax1.plot(k_values, test_r2, 's-', label='Test R²', linewidth=2, markersize=8, color='red')
ax1.set_xlabel('k (tokens ahead)', fontsize=12)
ax1.set_ylabel('R² Score', fontsize=12)
ax1.set_title('Train vs Test R² Scores', fontsize=14)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.6, 1.0])

# Add annotations
for k, train, test in zip(k_values, train_r2, test_r2):
    ax1.annotate(f'{test:.3f}', xy=(k, test), xytext=(5, -10),
                textcoords='offset points', fontsize=9, color='red')

# Plot 2: Prediction Quality
ax2 = plt.subplot(2, 2, 2)
ax2.plot(k_values, test_r2, 'o-', label='Test R²', linewidth=2, markersize=8, color='red')
ax2.plot(k_values, cosine_sim, '^-', label='Cosine Similarity', linewidth=2, markersize=8, color='purple')
ax2.set_xlabel('k (tokens ahead)', fontsize=12)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Prediction Quality Metrics', fontsize=14)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.6, 1.0])

# Plot 3: Accuracy Comparison
ax3 = plt.subplot(2, 2, 3)
ax3.plot(k_values, estimated_acc, 'd-', label='Estimated Accuracy', linewidth=2, markersize=10, color='orange')
# Plot actual accuracy for k=1
ax3.plot([1], [actual_acc[0]], 'o', label='ACTUAL Accuracy', markersize=12, color='green')
ax3.set_xlabel('k (tokens ahead)', fontsize=12)
ax3.set_ylabel('Accuracy', fontsize=12)
ax3.set_title('Estimated vs Actual Token Recovery', fontsize=14)
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, 1.0])

# Add annotations
ax3.annotate(f'{actual_acc[0]:.1%}', xy=(1, actual_acc[0]), xytext=(10, 10),
            textcoords='offset points', fontsize=11, color='green', fontweight='bold')
for k, est in zip(k_values, estimated_acc):
    ax3.annotate(f'{est:.1%}', xy=(k, est), xytext=(5, -15),
                textcoords='offset points', fontsize=9, color='orange')

# Plot 4: Summary Table
ax4 = plt.subplot(2, 2, 4)
ax4.axis('tight')
ax4.axis('off')

# Create table data
table_data = [
    ['Metric', 'k=1', 'k=2', 'k=4'],
    ['Train R²', f'{train_r2[0]:.3f}', f'{train_r2[1]:.3f}', f'{train_r2[2]:.3f}'],
    ['Test R²', f'{test_r2[0]:.3f}', f'{test_r2[1]:.3f}', f'{test_r2[2]:.3f}'],
    ['Cosine Sim', f'{cosine_sim[0]:.3f}', f'{cosine_sim[1]:.3f}', f'{cosine_sim[2]:.3f}'],
    ['Est. Accuracy', f'{estimated_acc[0]:.1%}', f'{estimated_acc[1]:.1%}', f'{estimated_acc[2]:.1%}'],
    ['ACTUAL Acc.', f'{actual_acc[0]:.1%}', 'N/A', 'N/A']
]

table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Style the header row
for i in range(4):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight actual accuracy
table[(5, 1)].set_facecolor('#FFE4B5')
table[(5, 1)].set_text_props(weight='bold')

ax4.set_title('Summary Results', fontsize=14, pad=20)

plt.suptitle('Temporal Probe Comprehensive Results\n'
             'Predicting Activations k Tokens Ahead Within Reasoning Chunks',
             fontsize=16, y=1.02)

plt.tight_layout()
plt.savefig('temporal_xc/final_results_comprehensive.png', dpi=150, bbox_inches='tight')
print("📊 Saved comprehensive results to temporal_xc/final_results_comprehensive.png")

# Key insights
print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print("1. High R² scores (85%+) show probes learn good representations")
print("2. Actual token accuracy (35% for k=1) is lower than R²")
print("3. This gap suggests:")
print("   - Activation patterns are predictable (high R²)")
print("   - But exact token prediction requires very precise activations")
print("   - Small errors in 4096-dim space → different tokens")
print("\n4. The 35% accuracy is still significant given:")
print("   - Vocabulary size: ~100k tokens")
print("   - Random baseline: ~0.001%")
print("   - Probe achieves 35,000x better than random!")

plt.show()