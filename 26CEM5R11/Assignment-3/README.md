ASSIGNMENT 3: ASSESSING THE PERFORMANCE OF SUPPORT VECTOR MACHINE AND DECISION TREE CLASSIFIER


DATASET USED:
Heart Disease Prediction Data (https://www.kaggle.com/datasets/mfarhaannazirkhan/heart-dataset)

Size: 1888 rows X 14 columns
After removing duplicates: 602 rows X 14 columns

Features:
1. Age
2. Sex
3. cp (Chest pain type ---> 0: Typical angina, 1-Atypical angina, 2-Non-anginal pain, 3-Asymptomatic)
4. trestbps (Resting Blood Pressure)
5. chol (Serum Cholestrol Level)
6. fbs (Fasting Blood Sugar > 120 mg/dl ---> 1-true, 0-false)
7. restecg (Resting electrocardiographic results ---> 0-Normal, 1-ST-T wave abnormality, 2-Left ventricular hypertropy)
8. thalach (Maximum heart rate)
9. exang (Exercise induced angina ---> 1-yes, 0-No)
10. oldpeak (ST depression induced by exercise relative to rest)
11. slope (slope of the peak exercise ---> 0-Upsloping, 1-Flat, 2-Downsloping)
12. ca (Number of maor blood vessels)
13. thal (Thalassemia types ---> 1-Normal, 2-Fixed, 3-Reversible defect)

Target (Outcome variable ---> 1-More chance of heart attack, 0-less chance of heart attack)


PROCEDURE:
1. Loaded the csv and removed the duplicates
2. Separated the Features (X) from the target (y)
3. Split into 75% train / 25% test (Random = 42)
4. Scaled the features (for SVM only)
5. Generated a t-SNE plot to check class separation
6. Trained SVM with linear, polynomial and RBF kernels
7. Trained Decision Tree varying max_depth and min_samples_split


OBSERVATIONS MADE:
- t-SNE plot tells that the classes are partly separated; so the 13 features carry signal but do not split the classes cleanly
- Among SVM, RBF performed the best (0.735) because it can bend its boundary around the data
- Among Decision Tree,
- The best performed model on this split is The Decision Tree which shows 118 vs 111 correct predictions


FILES:
-Assignment-3
--Assignment-3_26CEM5R11.py
--cleaned_merged_heart_dataset.csv
--results_summary.csv
--model_comparision.png
--svm_confusion_matrices.png
--tree_confusion_matrices.png
--svm_decision_boundaries.png
--tree_decision_boundaries.png
--tsne_plot.png
