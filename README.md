Machine Learning (ML)
# Project 4: Social Media Marketing Campaign Analytics Using Machine Learning

## Project Team
- Aaqib Jabbar
- Sofia Pai
- Prachi Patel
- Richard Encarnacion

## Project Overview
Our group will analyze and visualize the effectiveness of digital marketing campaigns using machine learning (ML) and other technologies we have learned. The project aims to identify key factors driving customer engagement and conversion rates by leveraging predictive modeling techniques.

## File Description
- data_cleaning.ipynb: Jupyter notebook with data cleaning steps
- digital_marketing_analysis_colab.ipynb: Jupyter notebook with Machine Learning models (Neural Network and Random Forest)
- Project4_Team4_Tableau.twbx: Desktop version of the Tableau dashboard
- Link to Tableau public: https://public.tableau.com/app/profile/aaqib.jabbar6339/viz/Project4_Team4_Tableau/Story1?publish=yes

## Problem Statement
Marketing campaigns generate vast amounts of data, but businesses often struggle to interpret this data to optimize their strategies. Our goal is to develop an ML model that predicts customer conversion rates based on marketing-specific and demographic variables, helping businesses maximize their return on investment (ROI).

## Scope & Purpose
- **Scope**: We will use machine learning techniques to analyze customer interactions with marketing campaigns and predict conversion rates.
- **Purpose**: To provide data-driven insights that help businesses allocate their ad spend more effectively and optimize their campaign strategies.

## Dataset
`digital_marketing_campaign_dataset.csv`

`cleaned_data.csv`

URL - https://raw.githubusercontent.com/hillz246/digital_market_csv/refs/heads/main/cleaned_data.csv

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

**Outcome**:

**1. Logistic Regression:**

1. Target Variable Imbalance:
The target variable "Conversion" has a severe class imbalance. Out of 7999 instances (7012 with a value of 1 and 988 with a value of 0), the majority (88%) belong to the positive class (1) while only a small fraction (12%) belong to the negative class (0).
In other words, the model is dealing with far more instances of one class (1) than the other (0), which can cause the model to be biased toward predicting the majority class. This results in poor predictive performance for the minority class (0).
2. Accuracy (88%):
Accuracy here is relatively high, but this number alone doesn’t fully capture the model's performance due to the imbalance. A model could achieve high accuracy by simply predicting the majority class most of the time, which would not necessarily indicate good overall performance, especially for the minority class.
For example, if the model always predicts "1", it would still achieve 88% accuracy (since 7012 out of 7999 instances are "1"), but it wouldn't be helpful for identifying the minority class ("0").
3. Precision:
Precision for Class 0 (0.60): This metric tells us that when the model predicts a "0" (no conversion), it is correct 60% of the time. In other words, 60% of the instances predicted as "0" are actually "0", while 40% are false positives (misclassifying a "1" as "0").
Precision for Class 1 (0.88): When the model predicts "1" (conversion), it is correct 88% of the time. This suggests that the model is more reliable when predicting the majority class.
4. Recall:
Recall for Class 0 (0.01): This is extremely low, indicating that the model is failing to identify most of the actual "0" cases (i.e., the non-conversion cases). Only 1% of the actual "0" instances are being correctly classified as "0". This is a significant issue, as it suggests that the model is not capturing many of the relevant "0" cases at all.
Recall for Class 1 (1.00): The recall for class "1" is perfect, meaning that the model correctly identifies all instances of "1" (conversion). While this is good, it also highlights the imbalance issue because the model is more focused on correctly predicting the majority class.
5. Key Takeaways:
Class Imbalance: The model’s performance is skewed toward predicting the majority class (1), which makes it less effective at identifying the minority class (0). The imbalance in the dataset is contributing to a high accuracy but poor performance for detecting non-conversions.
Precision-Recall Tradeoff: The model’s high precision for "1" and low recall for "0" suggests that it is cautious about predicting the minority class. It's likely avoiding false positives for "0" (non-conversion) but at the cost of missing most of those cases (low recall).

*2. Neural Network Architecture and Layers:*
   
First Attempt:

First Hidden Layer: 8 nodes with ReLU (Rectified Linear Unit) activation.

Second Hidden Layer: 5 nodes with Sigmoid activation.

Output Layer: Sigmoid activation (typically used for binary classification).

Accuracy: 0.89

Loss: 0.3043

Explanation:

The architecture is relatively simple with just two hidden layers.
The ReLU activation in the first hidden layer is often used to allow the model to learn complex patterns by introducing non-linearity. It typically works well for deeper networks.
The Sigmoid activation in the second hidden layer is less common for hidden layers but can still work. It has outputs between 0 and 1, which is useful in some cases, but can also lead to issues like vanishing gradients during training.
The Sigmoid output layer is standard for binary classification tasks where the goal is to predict either a 0 or 1.
Accuracy (0.89) indicates good performance, and loss (0.3043) suggests that the model's predictions are relatively close to the true values.
Second Attempt:

First Hidden Layer: 30 nodes with ReLU activation.

Second Hidden Layer: 20 nodes with Sigmoid activation.

Third Hidden Layer: 17 nodes with Sigmoid activation.

Fourth Hidden Layer: 5 nodes with Sigmoid activation.

Output Layer: Sigmoid activation.

Accuracy: 0.8540

Loss: 0.8611

Explanation:

This attempt is more complex, with four layers in total and increasing layers with fewer nodes as the network progresses.
The ReLU activation is still used in the first hidden layer, which is typical for the initial layers to allow for more powerful learning.
The subsequent layers use Sigmoid activation, which, as mentioned earlier, could lead to challenges like vanishing gradients or slower learning when used in deeper layers.
Sigmoid output layer is still appropriate for binary classification.
The lower accuracy (0.8540) and higher loss (0.8611) suggest that the model performed worse compared to Attempt 1. This could be due to:
Overcomplication: Adding multiple hidden layers and more nodes doesn't necessarily improve performance, and may actually lead to overfitting or difficulty in training, especially with a relatively small dataset.
Vanishing Gradients: Sigmoid functions in deeper layers may cause gradients to vanish during backpropagation, slowing learning and making optimization harder.
Overfitting: With more nodes and layers, the model may overfit to the training data, especially if the dataset is small, leading to poor generalization on unseen data.

2. Evaluation Metrics:
Accuracy: Measures the proportion of correct predictions. In Attempt 1, an accuracy of 0.89 shows strong performance, indicating that most predictions are correct. In Attempt 2, an accuracy of 0.8540 is still decent but lower than Attempt 1, meaning that the more complex architecture may not have generalized as well to the data.
Loss: Represents how well the model's predictions match the true labels. A lower loss means the model is making more accurate predictions. Attempt 1 has a loss of 0.3043, which is relatively low, suggesting a good fit. Attempt 2, with a loss of 0.8611, has a higher loss, indicating poorer performance and potentially an overfitting issue or that the model is too complex for the task.

   
   ## Tableau Story
- **Dashboard 1**: ![Screenshot 2025-02-12 194233](https://github.com/user-attachments/assets/e06be899-2a08-4458-a82d-aa693ccd0c6c)

- **Dashboard 2**: ![Screenshot 2025-02-12 194248](https://github.com/user-attachments/assets/5b8f5916-6e27-4ab2-9865-52cc40f3a164)

- **Dashboard 3**: ![Screenshot 2025-02-12 194257](https://github.com/user-attachments/assets/9c9d4633-bcd9-4804-871a-01df1ec51f3d)

- **Dashboard 4**: ![Screenshot 2025-02-12 194316](https://github.com/user-attachments/assets/c6d6c32a-9fc8-456a-be55-5873e8fbe611)

- **Final Story Presentation**: A step-by-step visualization showcasing insights, trends, and model predictions in an interactive Tableau format.


## Expected Outcomes
- A trained ML model that predicts customer conversion rates
- Data visualizations highlighting key insights
- A Tableau dashboard summarizing findings
- A final report and presentation summarizing results and recommendations

This project will enable us to apply machine learning techniques to real-world marketing data while utilizing the various technologies we have learned.

