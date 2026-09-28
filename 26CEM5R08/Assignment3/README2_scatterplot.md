Car Dataset - t-SNE Visualization

About the Project

In this project, I used a car dataset to visualize the relationship between different car features using t-SNE (t-Distributed Stochastic Neighbor Embedding).

The dataset contains information about cars such as horsepower, volume, speed and weight. I created a binary target based on the car's MPG value and then used t-SNE to reduce the four features into two dimensions.

The final result is a 2D scatter plot showing the two MPG classes.

---

Dataset

The dataset used for this project is:

Cars.csv

The following features are used:

- HP - Horsepower
- VOL - Volume
- SP - Speed
- WT - Weight
- MPG - Miles Per Gallon

I used MPG to create the target class.

Target Classes

The median MPG value of 35 is used as the cutoff.

MPG >= 35  →  High MPG (1)
MPG < 35   →  Low MPG (0)

This target is created using:

df['Target'] = (df['MPG'] >= 35).astype(int)

---

Libraries Used

I used the following Python libraries:

- Pandas - for reading and handling the dataset
- NumPy - for numerical operations
- Matplotlib - for creating the plot
- Seaborn - imported for visualization support
- Scikit-learn - for standardization and t-SNE

---

Steps Performed

1. Load the Dataset

First, I loaded the "Cars.csv" file using Pandas.

df = pd.read_csv('Cars.csv')

---

2. Create the Target Variable

Since the dataset does not have the required binary classification target, I created one using MPG.

Cars with MPG greater than or equal to 35 are assigned class "1", and the remaining cars are assigned class "0".

df['Target'] = (df['MPG'] >= 35).astype(int)

---

3. Select Features

I selected four car features for the analysis:

X = df[['HP', 'VOL', 'SP', 'WT']]
y = df['Target']

So, the input features are:

HP
VOL
SP
WT

and "Target" is the output class.

---

4. Standardize the Features

The features have different ranges, so I standardized them before applying t-SNE.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

This puts the features on a comparable scale.

---

5. Apply t-SNE

I used t-SNE to reduce the four-dimensional feature space into two dimensions.

tsne = TSNE(
    n_components=2,
    perplexity=15,
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

Here:

- "n_components=2" reduces the data to two dimensions.
- "perplexity=15" is used for the t-SNE calculation.
- "random_state=42" makes the result reproducible.

---

Visualization

After applying t-SNE, I created a scatter plot.

The two classes are shown separately:

- Low MPG (0)
- High MPG (1)

The X-axis represents t-SNE Dimension 1, and the Y-axis represents t-SNE Dimension 2.

The graph is saved as:

tsne_visualization.png

 Structure

Cars-tSNE-visualization

Cars.csv
scatter_plot.py
tsne_plot.png
README2.md

---

Requirements

Python 3.x is required.

The required libraries can be installed using:

pip install pandas numpy matplotlib seaborn scikit-learn

---

How to Run

First, keep "Cars.csv" in the same folder as the Python file.

Then run:

python tsne.py

The program will display the t-SNE plot and also save it as:

tsne_visualization.png

---

Conclusion

In this project, I used t-SNE to visualize car data in two dimensions.

I first selected HP, VOL, SP and WT as the features, standardized them and then applied t-SNE. I also created a binary target using MPG, where cars with MPG greater than or equal to 35 were considered High MPG.

The final visualization helps to see how the two MPG groups are distributed in the reduced feature space.

Through this work, I got a better understanding of feature selection, standardization, binary classification targets, dimensionality reduction and t-SNE visualization.