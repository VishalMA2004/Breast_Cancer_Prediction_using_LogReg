# Logistic Regression for Breast Cancer Prediction

This project demonstrates the use of a logistic regression model to predict whether a tumor is benign (B) or malignant (M) based on the Breast Cancer dataset. Below is a detailed explanation of how the code works and how you can interact with it.

## Requirements

Ensure you have the following Python libraries installed:
- pandas
- scikit-learn
- matplotlib
- seaborn

You can install them using:
```bash
pip install pandas scikit-learn matplotlib seaborn
```

## Dataset

The code uses the `BreastCancer.csv` dataset. Ensure the dataset is located in the same directory as the script or adjust the file path accordingly. The dataset should have the following structure:
- **id**: Unique identifier for each sample.
- **diagnosis**: Diagnosis of the tumor (B for benign, M for malignant).
- **Feature columns**: Numerical values representing various features of the tumors.

## Steps

1. **Load and Preprocess Data**
   - The dataset is loaded using `pandas`.
   - Unnecessary columns like `Unnamed: 32` are dropped.
   - The `diagnosis` column is mapped to binary values (0 for benign, 1 for malignant).
   - Missing values in the features are handled using the mean imputation strategy.
   - The features are standardized using `StandardScaler` for better model performance.

2. **Correlation Heatmap**
   - A heatmap of the correlation between standardized features is generated to visualize relationships between features.

3. **Split Dataset**
   - The dataset is split into training and testing sets with an 80-20 split ratio.
   - The split is stratified to maintain the proportion of benign and malignant samples.

4. **Train Logistic Regression Model**
   - A logistic regression model is trained on the training data.
   - The model uses a maximum of 200 iterations for convergence.

5. **Evaluate the Model**
   - Predictions are made on the test data.
   - The classification report and confusion matrix are displayed to assess performance.
   - The AUC-ROC score is calculated, and the ROC curve is plotted.

6. **Feature Importance**
   - The coefficients of the logistic regression model are visualized to understand feature importance.

## Outputs

- **Heatmap**: Visualizes correlations among features.
- **Classification Report**: Shows precision, recall, and F1-score.
- **Confusion Matrix**: Displays true positives, false positives, true negatives, and false negatives.
- **ROC Curve**: Plots the true positive rate against the false positive rate.
- **Feature Coefficients**: Bar plot of feature coefficients to indicate their contribution to the prediction.

## How to Run the Code

1. Save the code in a Python file, e.g., `breast_cancer_logistic_regression.py`.
2. Place the `BreastCancer.csv` dataset in the same directory as the script.
3. Run the script using:
   ```bash
   python breast_cancer_logistic_regression.py
   ```

## Interactivity

You can modify the following sections to explore the data further:

1. **Adjust Test Size**
   - Change the `test_size` parameter in the `train_test_split` function to use a different train-test split ratio.

2. **Visualize Other Features**
   - Customize the heatmap or create scatter plots to explore relationships between specific features.

3. **Try Other Models**
   - Replace `LogisticRegression` with other models from `sklearn` like `RandomForestClassifier` or `SVC` to compare performance.

4. **Hyperparameter Tuning**
   - Experiment with different hyperparameters for `LogisticRegression`, such as `penalty`, `C`, and `solver`.

5. **Feature Selection**
   - Remove features with low correlation or test the model with a subset of features to analyze performance impact.

## Notes

- Ensure the dataset is clean and follows the expected format before running the code.
- If the dataset contains a different structure, update the column handling accordingly.
- Interpret the outputs carefully to make meaningful insights into the predictions and feature importance.

## Conclusion

This project demonstrates a straightforward implementation of logistic regression for binary classification. It provides essential insights into data preprocessing, visualization, model training, and evaluation while allowing for further exploration and customization.

