# GroupNo.15_SportsPrediction
  The Python version of our Colab 3.10.12

  link to the video demonstration: 
  https://drive.google.com/file/d/18Hhzlrv_fIBHGjPx4ytAkqUFC7RBHN79/view?usp=sharing
  
# FIFA Player Rating Prediction

This repository contains a Python notebook for predicting FIFA player ratings using an ensembled machine learning model. The dataset used for this project consists of FIFA player data from FIFA 21 (training data) and FIFA 22 (test data). The goal is to train machine learning models that can predict a player's overall rating based on various features.

## Getting Started

### Prerequisites

Make sure you have the following Python libraries installed:

- `scikit-learn` (scikit-learn version 1.3.1 is recommended)
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `xgboost`
- `google.colab` (for Google Colab users)

You can install the required libraries using the following command:

```python
!pip install --upgrade scikit-learn pandas numpy seaborn matplotlib xgboost
```

### Data

You will need the following data files:

- `players_21.csv` (FIFA 21 player data)
- `players_22.csv` (FIFA 22 player data)

You can load these datasets into the notebook using `pd.read_csv()`.

### Code Structure

The notebook is structured as follows:

1. **Loading and Previewing the Data**: This section covers loading the training and test data and provides a preview of the data.

2. **Exploratory Data Analysis and Data Cleaning**: In this section, we explore the dataset, check for missing values, and perform data cleaning. We also eliminate unwanted columns that do not contribute to the prediction.

3. **Dealing with Missing Numerical Values**: Missing numerical values are replaced using median imputation.

4. **Dealing with Skills Columns**: The skills columns are converted to numerical values.

5. **One-Hot Encoding Categorical Columns**: Categorical columns are one-hot encoded for model training.

6. **Splitting the Data into Features and Target**: The dataset is split into features (X) and the target variable (y).

7. **Feature Extraction and Selection**: Feature extraction is performed to select the most relevant features using correlation analysis. The notebook demonstrates how to select the top features and visualize their correlations with the target variable.

8. **Feature Scaling**: Feature scaling is applied to the selected features.

9. **Model Training and Evaluation**: Various machine learning models are trained, including Linear Regression, Random Forest, Gradient Boosting, K-Nearest Neighbors (KNN), and XGBoost. Model evaluation metrics such as Mean Absolute Error, Mean Squared Error, and R-squared are provided.

10. **Hyperparameter Tuning**: GridSearchCV is used to search for the best hyperparameters for the XGBoost model.

11. **Cross-Validation**: Cross-validation is performed to assess the models' performance using RMSE (Root Mean Squared Error).

12. **Conclusion**: A summary of the model evaluations and insights from the analysis.

## Model Performance

Here are the mean RMSE values and 95% confidence intervals for the best-performing models:

- Random Forest:
  - Mean RMSE: 4.1455
  - 95% Confidence Interval: (1.5153, 6.7758)

- Gradient Boosting:
  - Mean RMSE: 4.2064
  - 95% Confidence Interval: (2.0001, 6.4127)

- XGBoost:
  - Mean RMSE: 3.7583
  - 95% Confidence Interval: (1.3590, 6.1577)

These metrics provide an assessment of the models' performance in predicting FIFA player ratings.   


The other two components of the project:

1. A web application built using Flask, which functions as the user interface for user input and predictions.
2. A pre-trained machine learning model that predicts player ratings based on user-provided input.

## Requirements

Before getting started with the project, please ensure you have the following software and resources installed on your system:

- Python 3.x
- Flask
- Pandas
- Scikit-Learn
- Joblib (used for model and scaler loading)
- HTML (for the website template)
- A pre-trained machine learning model (provided as 'ensemble_model.pkl')
- The scaler used during model training (provided as 'scaler.pkl')


