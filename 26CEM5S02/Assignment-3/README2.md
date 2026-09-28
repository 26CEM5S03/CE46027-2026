Heart Disease Dataset - t-SNE Visualization

About the Project

This project uses the Heart Disease dataset to visualize the data in two dimensions using t-SNE (t-Distributed Stochastic Neighbor Embedding).

The original dataset contains multiple features, which are difficult to visualize directly. t-SNE is used to reduce these features to just two dimensions so that the data points can be shown in a scatter plot.

The plot uses different colors for the two target classes:

- 0 - No Disease
- 1 - Heart Disease

The final plot is saved as "tsne_plot.png".

---

Dataset

The dataset used in this project is:

heart.csv

The "target" column is used as the class label.

The remaining columns are used as input features.

---

Libraries Used

The following Python libraries are used:

- Pandas - for loading and handling the dataset
- Matplotlib - for creating the scatter plot
- Scikit-learn - for feature standardization and t-SNE

---

Steps Performed

1. Load the Dataset

The dataset is loaded using Pandas.

df = pd.read_csv('heart.csv')

The code also removes a possible UTF-8 BOM character from the column names.

---

2. Separate Features and Target

The "target" column is separated from the other columns.

X = df.drop(columns=['target'])
y = df['target']

Here:

- "X" contains the input features.
- "y" contains the target class.

---

3. Standardize the Features

Before applying t-SNE, the features are standardized using "StandardScaler".

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Standardization puts the different features on a similar scale. This is useful because the features in a dataset can have different ranges.

---

4. Apply t-SNE

t-SNE is applied to reduce the feature space to two dimensions.

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

In this project:

- "n_components=2" means the data is reduced to two dimensions.
- "perplexity=30" controls the balance between local and broader relationships in the data.
- "random_state=42" is used so that the result can be reproduced.

---

5. Create the Scatter Plot

After applying t-SNE, the resulting two dimensions are plotted using Matplotlib.

The samples belonging to class "0" are shown as No Disease, while samples belonging to class "1" are shown as Heart Disease.

The plot contains:

- t-SNE Dimension 1 on the X-axis
- t-SNE Dimension 2 on the Y-axis
- Different colors for the two target classes
- A legend showing the target classes
- Grid lines for easier visualization

---

Output

The generated visualization is saved as:

tsne_plot.png

The program also displays the plot on the screen using:

plt.show()

The output can be used for assignment submission and to visually understand how the two classes are distributed after dimensionality reduction.

---

Project Structure

The files can be arranged like this:

Heart-Disease-tSNE/

 heart.csv
 tsne_plot.py
 tsne_plot.png
 README.md

---

Requirements

Python 3.x is required.

Install the required libraries using:

pip install pandas matplotlib scikit-learn

---

How to Run

1. Keep "heart.csv" in the same folder as the Python program.
2. Install the required Python libraries.
3. Run the Python program.

For example:

python tsne_plot.py

The program will generate and save:

tsne_plot.png

---

Conclusion

In this project, t-SNE was used to convert the original multi-feature Heart Disease dataset into a two-dimensional representation.

The resulting scatter plot makes it easier to visually inspect the distribution of the two target classes. It also gives a simple way to understand the structure of the dataset before applying machine learning classification methods.

This project helped me understand the basic use of feature standardization, dimensionality reduction, t-SNE, and data visualization.