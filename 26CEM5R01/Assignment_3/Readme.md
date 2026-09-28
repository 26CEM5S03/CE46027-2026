# CE46027 Assignment-3 Classification

**Roll No:** 26CEM5R01
**Dataset:** gender_classification_v7.csv (Kaggle) - 5001 samples, 7 features [long_hair, forehead_width_cm, forehead_height_cm, nose_wide, nose_long, lips_thin, distance_nose_to_lip_long], target=gender. Chosen because it's balanced (48% Female, 52% Male) and unique from Iris dataset.

### 1. Data Preprocessing
- Loaded CSV, checked no null values
- LabelEncoded target: Female=0, Male=1
- Train-Test split: 70-30, random_state=42
- StandardScaler for normalization

### 2. t-SNE Visualization
- Used StandardScaler + TSNE(n_components=2, perplexity=30, max_iter=500)
- Plot saved as `tsne_plot.png` color-coded by gender
- Observation: Two distinct clusters visible with slight overlap. Features show good separability, indicating classification will perform well. Long_hair and forehead features are strong discriminators.

### 3. SVM Results (with Execution Time & Memory)
| Model | Accuracy | Precision | Recall | F1 | Time (s) | Memory (MB) |
|-------|----------|-----------|--------|----|----------|-------------|
| SVM_linear | 96.33% | 96.33% | 96.33% | 96.33% | 0.057s | 2.18 |
| SVM_poly | 96.33% | 96.34% | 96.33% | 96.33% | 0.070s | 2.39 |
| **SVM_rbf** | **96.66% - BEST** | **96.69%** | **96.66%** | **96.66%** | 0.075s | 2.71 |

SVM RBF kernel performed best because it handles non-linear boundaries in facial features.

### 4. Decision Tree Results (varying depth & min_samples_split)
| Model | Accuracy | Precision | Recall | F1 | Time (s) |
|-------|----------|-----------|--------|----|----------|
| DT_depth=3_minSplit=2/5/10 | 95.40% | 95.40% | 95.40% | 95.40% | 0.005s |
| DT_depth=5_minSplit=2/5/10 | 95.73% | 95.80% | 95.73% | 95.73% | 0.006s |
| DT_depth=10_minSplit=2 | 96.00% | 96.01% | 96.00% | 96.00% | 0.008s |
| DT_depth=10_minSplit=5 | 96.26% | 96.27% | 96.26% | 96.26% | 0.008s |
| **DT_depth=10_minSplit=10** | **96.53% - BEST DT** | **96.53%** | **96.53%** | **96.53%** | 0.006s |
| DT_depth=None_minSplit=2 | 95.66% | 95.67% | 95.66% | 95.66% | 0.007s |
| DT_depth=None_minSplit=5 | 96.33% | 96.34% | 96.33% | 96.33% | 0.006s |
| DT_depth=None_minSplit=10 | 96.53% | 96.53% | 96.53% | 96.53% | 0.007s |

Observation: Depth 3 underfits. Depth 10 gives optimal balance. Depth None slightly overfits, reducing accuracy.

### 5. Final Comparison & Conclusion
| Metric | Best SVM (RBF) | Best DT (Depth 10, Split 10) |
|--------|----------------|------------------------------|
| Accuracy | **96.66%** | 96.53% |
| Precision | **96.69%** | 96.53% |
| Recall | **96.66%** | 96.53% |
| F1 | **96.69%** | 96.53% |
| Time | 0.075s (slower) | **0.006s (10x faster)** |
| Memory | 2.71 MB | ~0 MB |
| Confusion Matrix | [[722, 17], [33, 729]] | - |

**Conclusion:**
1. Overall best model is **SVM with RBF kernel (96.66% accuracy)**.
2. Decision Tree is 10x faster and uses less memory, with very close accuracy (96.53%), so it's better for real-time use.
3. Both models perform well, confirming t-SNE observation of separable clusters.
4. Increasing depth improves DT up to 10, beyond that risk of overfitting.

### Files
- Gender_classification.py - main code
- gender_classification_v7.csv - dataset
- tsne_plot.png - 2D t-SNE plot
- comparison_results.csv - detailed results table
- requirements.txt - dependencies

### How to Run