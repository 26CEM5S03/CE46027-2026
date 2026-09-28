import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 1. Load data and clean column names
df = pd.read_csv('heart.csv')
df.columns = [c.replace('ï»¿', '') for c in df.columns]

X = df.drop(columns=['target'])
y = df['target']

# 2. Standardize features before applying t-SNE
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply t-SNE (reducing 13 features to 2 dimensions)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# 4. Generate Scatter Plot
plt.figure(figsize=(8, 6))

# Plot Class 0: No Disease
plt.scatter(
    X_tsne[y == 0, 0], X_tsne[y == 0, 1], 
    label='No Disease (0)', 
    alpha=0.85, 
    c='#1f77b4', 
    edgecolors='k', 
    linewidths=0.7, 
    s=50
)

# Plot Class 1: Heart Disease
plt.scatter(
    X_tsne[y == 1, 0], X_tsne[y == 1, 1], 
    label='Heart Disease (1)', 
    alpha=0.85, 
    c='#ff7f0e', 
    edgecolors='k', 
    linewidths=0.7, 
    s=50
)

# Customize title and axis labels
plt.title('2D t-SNE Projection of Heart Disease Feature Space', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('t-SNE Dimension 1', fontsize=10)
plt.ylabel('t-SNE Dimension 2', fontsize=10)

# Add Legend and Grid
plt.legend(title='Target Class', frameon=True, facecolor='white', framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

# Save image file for assignment submission
plt.savefig('tsne_plot.png', dpi=300)
plt.show()