# StarkHealth-DiabetesPredictor
## Machine Learning for Diabetes Prediction and Early Detection

<img src="https://github.com/jamesehiabhi/StarkHealth-DiabetesPredictor/blob/main/Displays/Cover.png" alt="Displays" width="900" height="400"/> 

### 🩺INTRODUCTION
Diabetes is a chronic condition that affects millions of people worldwide. Early detection and prediction of diabetes can significantly improve patient outcomes and reduce healthcare costs. In this report, we delve into a comprehensive analysis of a diabetes prediction dataset using Python. The dataset, named _diabetes_prediction_dataset.csv_ contains various features such as gender, age, hypertension, heart disease, smoking history, BMI, HbA1c level, blood glucose level, and diabetes status. Our goal is to explore this dataset, clean it, and derive meaningful insights that can aid in predicting diabetes. 

<img src="https://github.com/jamesehiabhi/StarkHealth-DiabetesPredictor/blob/main/Displays/Cover1.png" alt="Displays" width="900" height="400"/> 

### 🩺DATASET OVERVIEW
The dataset comprises of 100,000 entries with the following features:
- **gender:** Gender of the individual (e.g., Female, Male)
- **age:** Age of the individual
- **hypertension:** Presence of hypertension (0 or 1)
- **heart_disease:** Presence of heart disease (0 or 1)
- **smoking_history:** Smoking history of the individual (e.g., never, current, former)
- **bmi:** Body Mass Index (BMI) of the individual
- **HbA1c_level:** HbA1c level, a measure of long-term blood glucose control
- **blood_glucose_level:** Blood glucose level
- **diabetes:** Diabetes status (0 or 1)

**Initial Observations**
- The dataset is well-structured with no missing values, which simplifies the data cleaning process.
- The age column was initially in float64 format but was converted to integer for consistency.
- The dataset includes a mix of categorical and numerical features, making it suitable for various machine learning models.

**Aim of the Project**
- Develop a robust machine learning model to predict diabetes onset.
- Enable early identification of high-risk patients for timely interventions.
- Improve patient outcomes by reducing complications through proactive care.
- Optimize healthcare resource allocation by prioritizing at-risk individuals.
- Lower long-term healthcare costs associated with diabetes management and treatment.
- Strengthen StarkHealth Clinic's role as a leader in technology-driven and patient-focused care.

### 🩺METHODOLOGY SUMMARY
**Step 1: Data Cleaning**
- Handle missing values with imputation techniques.
- Remove duplicates and irrelevant columns.
- Correct anomalies to ensure data quality.

**Step 2: Exploratory Data Analysis (EDA)**
- Visualize feature distributions and correlations.
- Identify patterns and anomalies.
- Formulate hypotheses for feature engineering and model selection.

**Step 3: Data Preprocessing**
- Scale/normalize numerical features and encode categorical variables.
- Split data into training, validation, and test sets.

**Step 4: Model Training**
- Train models like Logistic Regression, Random Forest, Gradient Boosting.
- Perform hyperparameter tuning and k-fold cross-validation.
- Compare multiple algorithms.

**Step 5: Model Evaluation**
- Assess performance using metrics like accuracy, precision, recall, F1-score, ROC-AUC.
- Analyze performance across subsets and perform error analysis.
- Compare to a baseline model.

**Step 6: Model Optimization**
- Fine-tune hyperparameters with Grid Search or Random Search.
- Apply regularization or ensemble methods to address overfitting.
- Refine feature selection for better generalization.

### 🩺EXPLORATORY DATA ANALYSIS (EDA)
**The analysis of categorical data revealed:**
- The **Smoking History** variable is heavily right-skewed.
- A higher proportion of **males have diabetes** compared to females (univariate and bivariate examination).
- **Former smokers** have the highest prevalence of diabetes, followed by those categorized as "ever" and "never" smokers.
- Patients over **50 years old** were predominantly former smokers when comparing smoking history with age.
- Patients with **no smoking information** had the highest proportion of negative diabetes cases (0).
- Those who **never smoked** exhibited the highest proportion of positive diabetes cases (1) compared to other categories.

<img src="https://github.com/jamesehiabhi/StarkHealth-DiabetesPredictor/blob/main/Displays/Categorical.png" alt="Displays" width="800" height="400"/> 

**The summary of the numerical features:**
- **age:** Ranges from 0 to 80 years, with a mean of approximately 41.88 years.
- **hypertension:** About 7.49% of the individuals have hypertension.
- **heart_disease:** Approximately 3.94% of the individuals have heart disease.
- **bmi:** Ranges from 10.01 to 95.69, with a mean of 27.32.
- **HbA1c_level:** Ranges from 3.5 to 9.0, with a mean of 5.53.
- **blood_glucose_level:** Ranges from 80 to 300, with a mean of 138.06.
- **diabetes:** About 8.5% of the individuals have diabetes.

<img src="https://github.com/jamesehiabhi/StarkHealth-DiabetesPredictor/blob/main/Displays/EDA%202.png" alt="Displays" width="900" height="400"/> 

**Distribution of Key Features**
- **Age Distribution:** The age distribution is relatively uniform, with a slight peak around the 40s.
- **BMI Distribution:** The BMI values are normally distributed around the mean, indicating a healthy range for most individuals.
- **HbA1c and Blood Glucose Levels:** These features show a right-skewed distribution, suggesting that a majority of individuals have lower levels, with a few outliers having higher levels.

<img src="https://github.com/jamesehiabhi/StarkHealth-DiabetesPredictor/blob/main/Displays/EDA%201.png" alt="Displays" width="800" height="400"/> 

**Correlation Analysis**
A correlation matrix can help identify relationships between features. For instance, the numerical variables indicated no negative correlations among the features. The strongest correlation was observed between **Blood Glucose Level** and **Diabetes** at 0.42, followed by **HbA1c Level** and **Diabetes** at 0.40, and **BMI** and **Age** at 0.34. The weakest correlation was between **HbA1c Level** and **Age**, with a value of 0.10. Visualizing these correlations can provide deeper insights into the factors contributing to diabetes.

<img src="https://github.com/jamesehiabhi/StarkHealth-DiabetesPredictor/blob/main/Displays/Correlation.png" alt="Displays" width="800" height="400"/> 

### 🩺DATA PREPROCESSING
During data preprocessing, categorical variables were encoded into numerical formats, the dataset was split into an 80% training set and a 20% testing set, and feature scaling was applied to normalize numerical variables. These steps ensured compatibility, accuracy, and a robust evaluation of the model's performance, establishing a solid foundation for reliable predictive modeling.

1. **Splitting the Data:** The dataset was split into training and testing sets (80-20 split) to evaluate the performance of predictive models.
  
2. **Gender:** Encoded gender into numerical values (Female: 0, Male: 1) for machine learning models.
  
3. **Smoking History:** Encoded smoking history into numerical categories (never: 0, current: 1, former: 2, ever: 3, not current: 4).

<img src="https://github.com/jamesehiabhi/StarkHealth-DiabetesPredictor/blob/main/Displays/Encode%20n%20split.png" alt="Displays" width="900" height="600"/> 

### 🩺MODEL TRAINING AND SELECTION
Following data preprocessing we explored to identify the model delivering the best performance to ensure the selection of a robust and accurate predictive model for diabetes onset. Model training commenced with the potential models for diabetes prediction including:
- **Logistic Regression:** A simple and interpretable model for binary classification.
- **Random Forest:** An ensemble method that can capture complex interactions between features.
- **Gradient Boosting:** A powerful method that builds trees sequentially to correct errors from previous trees.

### 🩺MODEL EVALUATION
The trained models were evaluated to assess their performance across key metrics, including **accuracy, precision, recall,** and **F1 score**, yielding an average score of **96%**  Among the four trained models, the **Random Forest classifier** demonstrated the best performance:
- **100% recall**
- **97% precision**
- **98% F1 score**
- **97% accuracy**

These results highlight Random Forest's superior ability to balance predictive accuracy and sensitivity, making it the most effective model for diabetes risk prediction.

<img src="https://github.com/jamesehiabhi/StarkHealth-DiabetesPredictor/blob/main/Displays/Model%20Ev.png" alt="Displays" width="800" height="500"/> 
 
### 🩺KEY INSIGHTS
- **Data Quality:** No missing values were identified, and anomalies such as date format issues were resolved to ensure data accuracy.
- **Age and Diabetes:** 
  - Older individuals from 50 years were predominantly former smokers, and those who never smoked had the highest positive diabetes cases, as indicated by the higher prevalence in the 41-60 and 61-80 age groups.

- **Feature Analysis:**
  - Higher BMI is strongly correlated with diabetes, suggesting that weight management is crucial in diabetes prevention.
  - The strongest correlations included Blood Glucose Level vs. Diabetes (0.42) and HbA1c Level vs. Diabetes (0.40).
  - The weakest correlation observed was between HbA1c Level and Age (0.10).

- **Modeling Results:**
  - Logistic Regression provided a baseline, while Decision Tree, SGD, and Random Forest models improved predictive performance.
  - Random Forest outperformed all other models, achieving **100% recall, 97% precision, 98% F1 score,** and **97% accuracy**.

- **Impactful Features:** Blood Glucose Level and HbA1c Level emerged as key predictors of diabetes onset.

**Project Outcome:** The Random Forest model's exceptional performance ensures robust diabetes risk prediction, enabling timely and targeted interventions for improved patient outcomes.


### 🩺RECOMMENDATIONS

**1. Early Screening and Monitoring:** Regular screening for individuals with higher BMI, older age groups, and those with elevated HbA1c and blood glucose levels to ensure early detection and effective management.

**2. Lifestyle and Targeted Interventions:** Promote healthy lifestyles, including diet and exercise, and develop specific preventive programs for former and current smokers, especially those over 50 years old.

**3. Enhanced Screening and Personalized Care:** Implement Random Forest-based predictive models for accurate identification of high-risk patients and design personalized care plans focusing on lifestyle modifications, regular monitoring, and early interventions.

**4. Educational Campaigns and Resource Allocation:** Launch awareness campaigns on the impact of smoking and obesity on diabetes risk and allocate healthcare resources efficiently to high-risk groups for timely interventions and cost reduction.

________________________________________
### 🩺CONCLUSION

This analysis provides a comprehensive overview of the diabetes prediction dataset, highlighting key features and their relationships with diabetes status. By leveraging machine learning models, we can predict diabetes with high accuracy, enabling early intervention and better management of the condition. The insights derived from this analysis can inform healthcare strategies and improve patient outcomes.
This project successfully developed a robust diabetes prediction model using advanced machine learning techniques, with the Random Forest classifier demonstrating exceptional performance in identifying at-risk individuals. Key predictors such as Blood Glucose Level and HbA1c Level were identified as critical to diabetes risk. The insights derived provide **StarkHealth Clinic** with actionable recommendations to enhance patient care, optimize resource allocation, and implement targeted preventive measures. By integrating this predictive model into its operations, **StarkHealth** can take a proactive approach to combating diabetes, improving patient outcomes, and reducing long-term healthcare costs. ✨📊
________________________________________

**For more information, you can contact me**

 <br>
 
### *Kindly share your feedback and I am happy to Connect✨*

<img src="https://github.com/jamesehiabhi/StarkHealth-DiabetesPredictor/blob/main/Displays/My%20Card1.jpg" alt="Displays" width="600" height="150"/>

