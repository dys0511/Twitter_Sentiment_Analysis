import matplotlib.pyplot as plt
import numpy as np

# Your exact model evaluation metrics
models = ['Linear SVM', 'Random Forest', 'XGBoost', 'Gradient Boosting', 'AdaBoost']
accuracy = [0.7329, 0.6969, 0.6816, 0.6779, 0.5999]
precision = [0.7250, 0.6583, 0.6361, 0.6337, 0.5610]
recall = [0.7504, 0.8188, 0.8488, 0.8434, 0.9186]
f1 = [0.7375, 0.7298, 0.7272, 0.7236, 0.6966]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

# Chart 1: Accuracy vs F1-Score
x = np.arange(len(models))
width = 0.35

rects1 = axes[0].bar(x - width/2, accuracy, width, label='Accuracy', color='#2b5c8f')
rects2 = axes[0].bar(x + width/2, f1, width, label='F1-Score', color='#4ba3e3')

axes[0].set_ylabel('Score', fontsize=11, fontweight='bold')
axes[0].set_title('Model Accuracy vs. F1-Score', fontsize=13, fontweight='bold', pad=12)
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, rotation=15, ha='right', fontsize=9.5)
axes[0].set_ylim(0.5, 0.8)
axes[0].legend(frameon=True)

# Add values above bars
for rect in rects1 + rects2:
    h = rect.get_height()
    axes[0].annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

# Chart 2: Precision vs Recall Scatter Plot
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
for i, m in enumerate(models):
    axes[1].scatter(precision[i], recall[i], color=colors[i], s=180, zorder=5)
    axes[1].annotate(m, (precision[i], recall[i]), xytext=(8, -4), textcoords='offset points', fontsize=9, fontweight='bold')

axes[1].set_xlabel('Precision', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Recall', fontsize=11, fontweight='bold')
axes[1].set_title('Precision vs. Recall Trade-Off', fontsize=13, fontweight='bold', pad=12)
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('model_comparison.png', bbox_inches='tight')
print("Saved chart as model_comparison.png")