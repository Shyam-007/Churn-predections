Assignment : 
  Predicting Customer Churn in a Telecommunications Company
  
Objective :
  The primary objective of this project is to develop a predictive model that can identify customers at risk of churning, enabling the company to take proactive measures to retain them.

1) Overview:
In this project, I aim to develop a predictive model for identifying customers at risk of churn within a telecommunications company. Utilizing a comprehensive dataset, I apply various machine learning techniques to predict churn behaviors, enabling targeted intervention strategies. The project involves rigorous data preprocessing, innovative feature engineering, and the application of models such as Logistic Regression, Random Forest, and Gradient Boosting, with a particular focus on addressing data imbalances and optimizing model performance to improve predictive accuracy.
The dataset comprises 7,043 entries, each representing a customer, across 21 attributes. These attributes include customer demographics, account information, and service usage details, with the target variable being Churn.

3) Data Cleaning and Preprocessing:
In the initial stages of the project, I focused on preparing the dataset for analysis and modeling, which involved several key steps to ensure data quality and usability:

i) Data Loading: 
I started by importing the dataset using Pandora's read_csv function, which allowed me to efficiently load and inspect the dataset.
  
ii) Handling Missing Values: 
Upon initial inspection, I replaced empty string values, represented as spaces, with NaN to standardize missing data representation. This was crucial for accurately assessing the presence of missing values across the dataset.
  
iii) Removing Duplicates and Irrelevant Data: 
I removed duplicate entries to prevent any bias or skewed analysis due to repeated records. Additionally, unnecessary columns that would not contribute to the churn analysis, such as 'customerID', were dropped.
  
iv) Data Type Conversion: 
A significant step involved converting the 'TotalCharges' column from a string to a numeric format. Errors during this conversion flagged entries that could not be converted, which were then handled appropriately to maintain data integrity.
  
v) Label Encoding: 
To prepare categorical data for modeling, I used Label Encoding to transform non-numeric columns into a machine-readable format. This included columns such as 'InternetService' and 'PaymentMethod'. Each category within these columns was assigned a unique integer, and I preserved the encoders for potential inverse transformations in future analysis or model interpretation.
  
vi) Feature Scaling: 
Numerical features like 'tenure', 'MonthlyCharges', and 'TotalCharges' were scaled using StandardScaler. This normalization step is essential to eliminate biases associated with differing ranges of data values, ensuring that each feature contributes equally to the model’s predictions.


3)EDA findings:
Distributions and Descriptive Statistics

i) Categorical Variables:
Gender: The distribution was nearly even, suggesting a balanced dataset in terms of gender representation.
Internet Service: A significant preference for Fiber optic services was observed, overshadowing DSL and those without internet service. This could indicate a trend towards higher-speed internet options among the customers.
Contract: The majority of customers were on month-to-month contracts, followed by one-year and two-year contracts. This suggests a prevalence of short-term commitments which may be influencing customer retention strategies.

ii) Numerical Variables:
Tenure: Showed a bimodal distribution with peaks at both low (new customers) and high (long-term customers) tenure ranges. This indicates two distinct customer segments—those newly acquired and those well retained.
Monthly Charges: Exhibited right-skewness, with many customers paying lower monthly fees but a tail of customers facing higher charges. This highlights a variance in service plans or additional service utilization.
Total Charges: Also right-skewed, suggesting that while most customers have lower cumulative expenditures, a smaller number have much higher total spending over their customer lifetime.

iii) Correlation Analysis
Tenure and Churn: The negative correlation suggests that customers who have stayed with the company longer are less likely to leave, possibly due to satisfaction or sunk cost effects.
Monthly Charges and Churn: The positive correlation implies that higher monthly costs could be a driver of customer turnover, possibly due to perceived value or competitive pricing issues.
Total Charges and Churn: The slight negative correlation indicates that customers with higher total expenditures tend to stay with the company, potentially due to higher engagement or more services used.

iv) Feature Influence on Churn
Service Features:
Lack of Online Security and Tech Support: These were notably linked to higher churn rates. Customers without these services might feel less secure or supported, leading to dissatisfaction.
Contract and Billing:
Month-to-month Contracts: These customers showed a much higher propensity to churn, possibly reflecting a lack of commitment or dissatisfaction with services that prevent long-term commitments.
Electronic Payment Methods: This method had a higher churn rate, which could suggest issues with automatic billing or the demographic segment using electronic payments.
Demographics:
Senior Citizens and Dependents: Seniors showed higher churn, particularly those without dependents. This might reflect differing service needs or dissatisfaction among this demographic group.

v)Insights
Gender Impact on Churn: Graphs showed no significant difference in churn between genders, suggesting that churn is influenced more by service-related factors than by gender.
Internet Service Impact on Churn: Fiber optic users had a notably higher churn rate than DSL users, which could be due to higher expectations or reliability issues associated with high-speed internet services.
Contract Length Impact on Churn: Short-term, month-to-month customers were much more likely to churn than those bound by longer contractual obligations. This visualization underscores the importance of contract length as a key factor in customer retention strategies.

4) Feature Engneering :
I focused on developing a set of advanced features to enhance the predictive model's accuracy for customer churn at a telecommunications company. My goal was to delve deeper into the data to uncover more nuanced insights and strengthen the model's ability to forecast churn based on various customer behaviors and interactions with service offerings. Below, I outline the feature engineering techniques I employed, detailing the rationale and implementation for each.

i) Total Charges Cubed
Purpose: I introduced a cubic transformation of the TotalCharges variable to capture non-linear effects that might influence churn, hypothesizing that higher expenditure over time might impact customer decisions differently depending on the magnitude.
Implementation: data['TotalCharges_cubed'] = data['TotalCharges'] ** 3

ii) Interaction between Monthly Charges and Contract Type
Purpose: Recognizing that different contract types could interact with monthly charges to affect churn, I created this feature to explore if customers on month-to-month contracts with high monthly fees are more likely to churn.
Implementation: data['Interact_MonthlyContract'] = data['Contract'] * data['MonthlyCharges']

iii.High Churn Payment Method
Purpose: With the knowledge that certain payment methods correlate with higher churn rates, I flagged customers using electronic checks—a payment method associated with higher churn.
Implementation: data['HighChurn_PaymentMethod'] = (data['PaymentMethod'] == label_encoders['PaymentMethod'].transform(['Electronic check'])[0]).astype(int)

iv.Senior and Alone
Purpose: To identify potentially vulnerable customer segments, I combined demographic factors—being a senior and living without dependents—to capture a group that might face higher churn risks.
Implementation: data['Senior_and_Alone'] = data['SeniorCitizen'] * (1 - data['Dependents'])

v.Monthly to Total Charges Ratio
Purpose: This ratio helps identify how much a customer is paying relative to their total interaction with the company, giving insight into whether customers might perceive their charges as reasonable over their tenure.
Implementation: data['MonthlyTotalRatio'] = data['MonthlyCharges'] / (data['TotalCharges'] + 1e-5)

vi.Bundle Penetration
Purpose: By measuring the number of service categories a customer subscribes to, this feature aims to determine if deeper integration with the company's services correlates with lower churn rates.
Implementation: data['BundlePenetration'] = data[service_categories].gt(0).sum(axis=1)

5) Model Selection :
In the process of developing a predictive model for customer churn, I explored several machine learning algorithms, including Logistic Regression, XGBoost, and Random Forest. Here, I detail the steps taken for model selection and the rationale behind choosing the best-performing model, followed by hyperparameter tuning to further refine its performance.

i) Logistic Regression:
Overview: Started with logistic regression due to its simplicity and interpretability. It provides a good baseline model to understand the influence of different features on the likelihood of churn.
Performance: While effective for linear relationships, it was less capable of capturing complex interactions and non-linear patterns present in the data.

ii)XGBoost:
Overview: Utilized XGBoost for its strength in handling varied data types, missing data, and its ability to model complex relationships through boosting.
Performance: XGBoost performed well, particularly in terms of handling non-linearities and interactions, but required careful tuning to avoid overfitting.

iii)Random Forest:
Overview: Chose Random Forest for its robustness, ability to model non-linear interactions, and ease of use in feature importance evaluation.
Performance: Random Forest outperformed the other models in initial tests. It demonstrated high accuracy and good generalizability without extensive hyperparameter tuning. The ensemble approach, using multiple decision trees, provided a more stable prediction across the data.

Final Model Choice
Based on the initial performance metrics, Random Forest was selected as the most promising model due to its superior accuracy and balance between bias and variance. Its performance suggested that it could effectively capture the complex patterns and interactions in the churn data.

Hyperparameter Tuning
To optimize the Random Forest model, I applied grid search (GridSearchCV) for hyperparameter tuning, focusing on parameters such as n_estimators, max_features, max_depth, and min_samples_split.

GridSearchCV:
Purpose: Automates the process of finding the most effective parameters. It iteratively trains the model using combinations of parameters from a predefined grid, evaluating using cross-validation.
Implementation: Defined a parameter grid with a range of values for n_estimators, max_depth, etc., and conducted a grid search to find the combination that resulted in the best cross-validated accuracy.
Outcome: The best parameters involved a higher number of trees, deeper trees, and a balance in the minimum samples per split, which enhanced the model’s ability to learn from the data without fitting excessively to the training set.

6) Evaluation Result :
The chosen model, Random Forest Classifier, has been evaluated using a comprehensive set of performance metrics which are essential for understanding its effectiveness in predicting customer churn. Below, I provide a detailed assessment based on the key metrics: accuracy, precision, recall, F1-score, and the AUC score.
Metrics Overview

i) Accuracy:
Value: 0.85
Interpretation: The model accurately predicts whether a customer will churn or not 85% of the time. This high level of accuracy indicates that the model is robust and performs well on the test dataset.

ii) Precision:
Class 0 (No Churn): 0.86
Class 1 (Churn): 0.84
Interpretation: The precision score reflects the model's ability to label as churn only those customers who actually churn. High precision for both classes means fewer false positives, which is crucial for not targeting wrong customers with retention strategies.

iii)Recall:
Class 0 (No Churn): 0.83
Class 1 (Churn): 0.86
Interpretation: Recall indicates the model's ability to find all the relevant instances of churn. The model’s recall suggests it is slightly better at identifying customers who will churn than those who will not.

iv)F1-Score:
Class 0 (No Churn): 0.85
Class 1 (Churn): 0.85
Interpretation: F1-score is the harmonic mean of precision and recall, and a score of 0.85 for both classes indicates a well-balanced model that is robust across both precision and recall metrics.

v)AUC Score:
Value: 0.8471
Interpretation: The AUC score is close to 1, which is excellent. It measures the model’s ability to discriminate between those who churn and those who do not. A higher AUC value suggests that the model has a good measure of separability between positive and negative classes.

Thus performance metrics suggest that the Random Forest Classifier is highly effective in predicting churn. Its balanced precision and recall imply that it is capable of identifying most of the churn accurately without a significant number of false positives, which is essential for implementing effective customer retention strategies.

7) Challenges faced :
i). Data Quality and Integrity
Challenge: The dataset had missing values and inconsistencies, especially in the TotalCharges field.
Solution: Implemented data cleaning, handled missing values, and ensured consistency across the dataset for reliable modeling.

ii. Feature Engineering Complexity
Challenge: Determining predictive features required complex feature engineering with various interactions.
Solution: Explored numerous feature combinations and transformations to better capture data relationships.

iii. Balancing Model Complexity and Overfitting
Challenge: Advanced models like XGBoost and Random Forest risked overfitting the training data.
Solution: Used cross-validation, grid search for hyperparameter tuning, and regularization to prevent overfitting.

