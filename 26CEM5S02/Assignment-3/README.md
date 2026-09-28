Heart Disease Classification using SVM and Decision Tree

About the Project

This project is about classifying whether a person has heart disease or not using machine learning.

I used the Heart Disease dataset ("heart.csv") and compared different machine learning models, mainly Support Vector Machine (SVM) and Decision Tree classifiers.

Along with classification, I also used t-SNE to visualize the dataset in two dimensions. The performance of each model was compared using accuracy, precision, recall, F1-score, execution time, and confusion matrix.

The main purpose of this project is to understand how different machine learning models perform on the same dataset and how their parameters affect the results.

---

Dataset

The dataset used in this project is:

File: "heart.csv"

The target column is:

- "0" → No heart disease
- "1" → Heart disease

All the other columns are used as input features for classification.

---

Libraries Used

The following Python libraries were used:

- Pandas – for loading and handling the dataset
- NumPy – for numerical operations
- Matplotlib – for plotting the t-SNE visualization
- Scikit-learn – for preprocessing, t-SNE, model training and evaluation
- Time – for measuring model execution time
- Tracemalloc – for tracking peak memory usage

---

Steps Followed

1. Loading the Dataset

The dataset is loaded using Pandas:

df = pd.read_csv('heart.csv')

The code also removes a possible UTF-8 BOM character from the column names.

The target column is separated from the input features:

X = df.drop(columns=['target'])
y = df['target']

---

2. Feature Standardization

Before applying the machine learning models, the input features are standardized using "StandardScaler".

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Standardization converts the features to a similar scale. This is particularly important for SVM because SVM is sensitive to the scale of the input features.

---

3. t-SNE Visualization

t-SNE (t-Distributed Stochastic Neighbor Embedding) is used to reduce the feature space to two dimensions for visualization.

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

The resulting plot shows the two target classes:

- 0 – No Disease
- 1 – Heart Disease

The generated figure is saved as:

tsne_plot.png

The t-SNE plot is mainly used for visualization and is not used as the input to the classifiers.

---

4. Train-Test Split

The dataset is divided into training and testing sets.

- 80% → Training data
- 20% → Testing data

The split is stratified so that the proportion of the two target classes is maintained as much as possible in both sets.

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

---

5. Machine Learning Models

Six different models/configurations are compared.

Support Vector Machine

SVM with Linear Kernel

SVC(kernel='linear', random_state=42)

This uses a linear decision boundary to separate the two classes.

SVM with Polynomial Kernel

SVC(kernel='poly', degree=3, random_state=42)

A third-degree polynomial kernel is used to create a non-linear decision boundary.

SVM with RBF Kernel

SVC(kernel='rbf', random_state=42)

The RBF kernel can model more complex non-linear relationships between the features.

---

Decision Tree

Shallow Decision Tree

DecisionTreeClassifier(max_depth=2, random_state=42)

The maximum depth is limited to 2. This produces a simple tree and helps examine the effect of restricting model complexity.

Deep Decision Tree

DecisionTreeClassifier(max_depth=8, random_state=42)

A deeper tree is allowed to learn more complex patterns in the training data.

Decision Tree with Minimum Split Requirement

DecisionTreeClassifier(
    min_samples_split=5,
    random_state=42
)

Here, a node must have at least 5 samples before it can be split.

---

6. Evaluation Metrics

Each model is evaluated using the following metrics.

Accuracy

Accuracy measures the overall proportion of correctly classified samples.

Accuracy = Correct Predictions / Total Predictions

Precision

Precision tells us how many of the samples predicted as positive were actually positive.

Precision = TP / (TP + FP)

Recall

Recall tells us how many of the actual positive samples were correctly identified.

Recall = TP / (TP + FN)

F1-Score

F1-score combines precision and recall into a single measure.

F1 = 2 × (Precision × Recall) / (Precision + Recall)

Confusion Matrix

The confusion matrix is represented as:

[TN, FP, FN, TP]

where:

- TN = True Negative
- FP = False Positive
- FN = False Negative
- TP = True Positive

---

7. Execution Time and Memory

The code also measures the time required for each model to train and make predictions.

Python's "time.perf_counter()" is used for execution-time measurement.

"tracemalloc" is used to monitor peak memory allocated by Python during the model training and prediction process.

This allows the models to be compared not only based on classification performance but also based on computational requirements.

«Note: The current code measures peak memory internally, but it does not include the memory value in the final printed table.»

---

Output

The program prints a comparison table similar to:

Model                | Acc    | Prec   | Rec    | F1     | Time(ms) | CM [TN,FP,FN,TP]
SVM (Linear)         | 0.XX   | 0.XX   | 0.XX   | 0.XX   | XX.XX    | [XX, XX, XX, XX]
SVM (Polynomial)     | 0.XX   | 0.XX   | 0.XX   | 0.XX   | XX.XX    | [XX, XX, XX, XX]
SVM (RBF)            | 0.XX   | 0.XX   | 0.XX   | 0.XX   | XX.XX    | [XX, XX, XX, XX]
DT (Shallow, d=2)    | 0.XX   | 0.XX   | 0.XX   | 0.XX   | XX.XX    | [XX, XX, XX, XX]
DT (Deep, d=8)       | 0.XX   | 0.XX   | 0.XX   | 0.XX   | XX.XX    | [XX, XX, XX, XX]
DT (min_split=5)     | 0.XX   | 0.XX   | 0.XX   | 0.XX   | XX.XX    | [XX, XX, XX, XX]

The actual values depend on the contents of the "heart.csv" dataset and the machine on which the code is executed.

---

Project Files

The project can have the following structure:

Heart-Disease-Classification/

heart.csv
heart_disease_classification.py
tsne_plot.png
README.md

---

Requirements

Python 3.x is required.

Install the required libraries using:

pip install pandas numpy matplotlib scikit-learn

---

How to Run

1. Download or place the "heart.csv" dataset in the same folder as the Python program.
2. Make sure the target column is named "target".
3. Install the required Python libraries.
4. Run the Python file.

For example:

python heart_disease_classification.py

After running the program:

- A t-SNE visualization will be displayed.
- The t-SNE plot will be saved as "tsne_plot.png".
- The performance of all six models will be printed in the terminal.

---

Conclusion

In this project, I compared different SVM and Decision Tree configurations for heart disease classification.

The experiment shows how changing the kernel or tree parameters can affect classification performance and computational time. The evaluation is based on multiple metrics rather than accuracy alone.

The t-SNE visualization also provides a way to look at how the samples are distributed in a two-dimensional representation of the original feature space.

Overall, this project helped me understand the practical implementation and comparison of SVM, Decision Trees, feature standardization, t-SNE, and classification evaluation metrics.
