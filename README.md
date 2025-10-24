# Life Insurance Purchase Prediction using Logistic Regression

This project demonstrates how to build a logistic regression model to predict whether a person will buy life insurance based on their age.

## Project Goal

The goal of this project is to:
- Load and explore a dataset containing age and insurance purchase information.
- Visualize the relationship between age and insurance purchase.
- Split the data into training and testing sets.
- Train a logistic regression model.
- Evaluate the model's performance using metrics like classification report, confusion matrix, and ROC curve.
- Perform cross-validation to assess model robustness.
- Tune hyperparameters to potentially improve model performance.
- Save the trained model and scaler for future predictions.
- Demonstrate how to make predictions on new data.

## Dataset

The dataset used in this project is `insurance_data.csv`. It contains two columns:
- `age`: The age of the individual.
- `bought_insurance`: A binary variable indicating whether the individual bought insurance (1) or not (0).

## Project Steps

The notebook follows these steps:

1.  **Import Libraries**: Import necessary libraries for data manipulation, visualization, and model building (`pandas`, `numpy`, `matplotlib`, `sklearn`, `joblib`).
2.  **Load Dataset**: Load the `insurance_data.csv` file into a pandas DataFrame.
3.  **Exploratory Data Analysis (EDA)**: Perform basic checks on the data, including displaying the head of the DataFrame, checking data types, and looking for missing values. A scatter plot is generated to visualize the relationship between age and insurance purchase.
4.  **Data Splitting**: Split the dataset into training (80%) and testing (20%) sets using `train_test_split`. Stratification is used to ensure the proportion of the target variable (`bought_insurance`) is the same in both sets.
5.  **Feature Scaling**: Standardize the `age` feature using `StandardScaler` to ensure the logistic regression model converges properly.
6.  **Model Training**: Train a `LogisticRegression` model on the scaled training data.
7.  **Model Evaluation**: Evaluate the trained model's performance on the test set using:
    -   Classification Report (`classification_report`)
    -   Confusion Matrix (`confusion_matrix`), which is also visualized.
    -   ROC Curve and AUC (`roc_curve`, `auc`), which is also visualized.
8.  **Cross-Validation**: Perform 5-fold cross-validation on the entire dataset to get a more robust estimate of the model's accuracy.
9.  **Hyperparameter Tuning**: Use `GridSearchCV` to find the best hyperparameters for the `LogisticRegression` model.
10. **Visualize Decision Boundary**: Plot the logistic regression decision boundary to visually represent the model's prediction probability based on age.
11. **Save Model and Scaler**: Save the trained logistic regression model and the fitted scaler object using `joblib` for later use without retraining.
12. **Make Predictions**: Demonstrate how to load the saved model and scaler and make predictions on new age values.

