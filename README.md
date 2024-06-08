# Ashutosh_dscTaskRound
This project analyzes the 2021 New Coder Survey data to uncover insights about new coders, including their learning habits, programming experience, financial investments, and income. The analysis includes data preprocessing, exploratory data analysis (EDA), clustering, and classification tasks.

Table of Contents
Getting Started
Data Preprocessing
Exploratory Data Analysis
Clustering
Classification
Results
Dependencies
Usage
Getting Started
To get started, download the 2021 New Coder Survey dataset and place it in the appropriate directory as indicated in the code. The main analysis script will handle the rest.

Data Preprocessing
The data preprocessing steps include:

Custom Converters: Convert mixed data types to floats, handling non-numeric characters such as commas and dollar signs.
Timestamp Conversion: Convert the Timestamp column to datetime.
Missing Value Handling: Fill missing values for numeric columns with their median and for non-numeric columns with their mode.
Exploratory Data Analysis
The EDA includes:

Visualizing Employment Status: A bar plot showing the distribution of different employment statuses among the respondents.
Income Distribution: A histogram with a KDE plot to show the distribution of respondents' incomes.
Correlation Matrix: A heatmap of the correlation matrix for numeric columns to identify relationships between variables.
Clustering
The clustering analysis involves:

Feature Selection: Using the number of hours spent learning each week, months of programming experience, and income.
Standardization: Standardizing the features before clustering.
Elbow Method: Determining the optimal number of clusters.
KMeans Clustering: Applying KMeans clustering with the optimal number of clusters.
PCA Visualization: Visualizing the clusters using Principal Component Analysis (PCA).
Classification
The classification analysis involves:

Feature Selection: Using features related to learning hours, programming experience, money spent on learning, employment status, and current software development job status.
Binary Classification: Creating a binary target variable to classify respondents as high income (≥ $30,000) or not.
Train-Test Split: Splitting the data into training and testing sets.
Model Training: Training multiple models including Random Forest, Logistic Regression, Support Vector Machine (SVM), and Gradient Boosting.
Model Evaluation: Evaluating the models using accuracy, classification report, and confusion matrix.
Results
The results include:

Clustering: Visualization of clusters showing different groups of respondents based on their learning habits, experience, and income.
Classification: Accuracy and performance metrics for different models in predicting high-income respondents.
Feature Importance: Ranking of features based on their importance in the Random Forest model.

Dependencies
To run this analysis, you need the following Python libraries:

pandas
numpy
seaborn
matplotlib
scikit-learn

Prepare the Data: Ensure the survey data CSV file is correctly placed in the specified directory.
Run the Analysis: Execute the main analysis script. It will process the data, perform EDA, clustering, and classification, and generate visualizations.
python analysis_script.py
Inspect the Results: Review the generated plots and printed evaluation metrics to understand the findings from the survey data.
