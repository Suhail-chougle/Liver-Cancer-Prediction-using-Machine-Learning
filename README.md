**Liver Cancer Prediction using Machine Learning**

## Introduction

Liver cancer, also known as hepatic cancer, is a serious health concern with a significant impact on global mortality. Early detection of liver cancer is critical for effective treatment and improved patient outcomes. This project aims to build a machine learning model for liver cancer prediction using a dataset containing various patient features. The primary objective is to develop a model with reduced False Negatives to enhance diagnostic accuracy.

## Dataset

The dataset used for this project contains patient information, including demographics, medical history, and diagnostic test results. Key features include age, gender, patient status (healthy or diagnosed with liver cancer), and several other relevant attributes.

## Data Exploration and Preprocessing

The project begins with data exploration to identify missing values and redundant features. Null values are handled by either replacing them with the mode for categorical data or the mean for numerical data. Outliers are addressed using quantile-based flooring and capping. Additionally, some features are transformed to improve their relevance and usefulness for the model.

## Feature Selection

Feature selection is a crucial step in building an accurate model. Pearson's correlation is used to identify highly correlated features, and duplicate and irrelevant features are removed from the dataset. This process results in a more efficient and focused set of features for model training.

## Model Building

Three different classifiers are employed for model building: XGBoost, Random Forest, and Logistic Regression. Each model is trained on the training dataset and evaluated on the test dataset. Hyperparameter tuning using Randomized Search Cross-Validation helps identify the best combination of parameters for optimal model performance.

## Evaluation Metrics

The model's performance is evaluated using several metrics, including accuracy, specificity, sensitivity, precision, and F1-score. The main focus is on reducing False Negatives to improve the diagnostic capability of the model.

## Results

The model evaluation shows that XGBoost performs the best, providing the highest accuracy and the lowest False Negatives among the three classifiers. Logistic Regression also performs well, closely behind XGBoost. The project highlights the importance of metrics like specificity and sensitivity in diagnostic analyses to minimize false diagnoses.

## Conclusion

The Liver Cancer Prediction using Machine Learning project presents a robust model for early liver cancer detection. The implementation of XGBoost and Logistic Regression models, along with careful feature selection and preprocessing, leads to enhanced accuracy and reduced False Negatives. This project has the potential to assist medical professionals in diagnosing liver cancer at an early stage, thereby improving patient prognosis and contributing to better healthcare outcomes. Further research and model optimization can be pursued to make the prediction system even more efficient and applicable in real-world healthcare settings.

## Dependencies

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

