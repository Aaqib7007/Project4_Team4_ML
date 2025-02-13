# Project4_Team4_ML
Machine Learning (ML)
# Project 4: Social Media Marketing Campaign Analytics Using Machine Learning

## Project Team
- Aaqib Jabbar
- Sofia Pai
- Prachi Patel
- Richard Encarnacion

## Project Overview
Our group will analyze and visualize the effectiveness of digital marketing campaigns using machine learning (ML) and other technologies we have learned. The project aims to identify key factors driving customer engagement and conversion rates by leveraging predictive modeling techniques.

## Problem Statement
Marketing campaigns generate vast amounts of data, but businesses often struggle to interpret this data to optimize their strategies. Our goal is to develop an ML model that predicts customer conversion rates based on marketing-specific and demographic variables, helping businesses maximize their return on investment (ROI).

## Scope & Purpose
- **Scope**: We will use machine learning techniques to analyze customer interactions with marketing campaigns and predict conversion rates.
- **Purpose**: To provide data-driven insights that help businesses allocate their ad spend more effectively and optimize their campaign strategies.

## Dataset
- **Source**: [Kaggle - Social Media Marketing Campaign Analytics Data](https://www.kaggle.com/datasets/rabieelkharoua/predict-conversion-in-digital-marketing-dataset)
- **Size**: 8000 rows of data
- **Features**:
  - **Demographic Information**: CustomerID, Age, Gender, Income
  - **Marketing-Specific Variables**: CampaignChannel, CampaignType, AdSpend, ClickThroughRate, ConversionRate
  - **Customer Engagement Variables**: WebsiteVisits, PagesPerVisit, TimeOnSite, SocialShares, EmailOpens, EmailClicks
  - **Historical Data**: PreviousPurchases, LoyaltyPoints
  - **Target Variable**: Conversion (1 or 0)

## Key Questions to Answer
1. Which marketing channels yield the highest conversion rates?
2. What customer demographics are most likely to convert?
3. How does ad spend influence conversion rates?
4. What engagement metrics (e.g., email opens, social shares) are the strongest predictors of conversion?
5. Can we build an ML model to predict customer conversion likelihood based on historical and engagement data?

## Technologies & Tools Used
- **Machine Learning**: Scikit-learn
- **Data Processing & Analysis**: Python Pandas
- **Visualization**: Python Matplotlib, Tableau
- **Jupyter Notebook**: For exploration, cleanup, and analysis

## Project Milestones & Timeline
1. **Project Ideation (Week 1, Day 1)**
   - Define project scope, goals, and dataset selection from Kaggle
2. **Data Fetching & Exploratory Data Analysis (EDA) (Week 1, Day 2)**
   - Download dataset, clean and preprocess data
   - Visualize trends, detect missing values, and perform feature engineering
3. **Building the ML Model (Week 1, Day 3)**
   - Train and test different machine learning models (e.g., Scikit-learn)
   - Build a Tableau visualization model
4. **Model Evaluation & Optimization (Week 2, Day 1)**
   - Fine-tune hyperparameters, optimize performance, visualizations, etc.
5. **Testing & Validation (Week 2, Day 1)**
   - Validate model on unseen data, assess accuracy in Jupyter notebook and Tableau
6. **Creating Documentation (Week 2, Day 2)**
   - Document methodology, findings, and insights and upload the project to GitHub
7. **Final Presentation Preparation (Week 2, Day 2)**
   - Develop Tableau Story using Tableau dashboards and presentation slides

## Implementation Steps
1. **Data Fetching and Preparation**
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from skimpy import skim
   import ppscore as pps

   data = pd.read_csv("/Desktop/digital_marketing_campaign_dataset.csv")
   data.head()
   ```

2. **Data Cleaning and Exploration**
   ```python
   # Checking for duplicate rows
   print(f"Total duplicate rows: {data.duplicated().sum()}")
   # Checking for missing values
   print(data.isnull().sum())
   # Summary statistics
   skim(data)
   ```

3. **Data Visualization**
   ```python
   # Plotting conversion distribution
   plt.figure(figsize=(5, 3))
   plt.show()
   ```

4. **Model Building**
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler, LabelEncoder
   
   # Encoding categorical variables
   categorical_features = ['CampaignChannel', 'CampaignType', 'Gender']
   
   # Splitting the data
   X = data.drop(columns=['Conversion'])
   y = data['Conversion']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   # Scaling numerical features
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```

5. **Model Training & Evaluation**
   ```python
   # Predictions
   y_pred = model.predict(X_test_scaled)
   
   # Evaluation
   from sklearn.metrics import classification_report, confusion_matrix
   print(classification_report(y_test, y_pred))
   print(confusion_matrix(y_test, y_pred))
   ```

## Expected Outcomes
- A trained ML model that predicts customer conversion rates
- Data visualizations highlighting key insights
- A Tableau dashboard summarizing findings
- A final report and presentation summarizing results and recommendations

This project will enable us to apply machine learning techniques to real-world marketing data while utilizing the various technologies we have learned.

