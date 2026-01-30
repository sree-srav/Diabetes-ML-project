import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('diabetic_data.csv')
df.head()

df = df.replace('?', np.nan)
df = df.drop(columns=['payer_code','medical_specialty'])
df = df.dropna(subset=['race', 'diag_1', 'diag_2', 'diag_3'])
df['age'] = df['age'].str.extract('(\d+)-').astype(int)
df.dropna(subset=['weight'], inplace=True)
df.head()

# Load the dataset
diabetes_data = pd.read_csv('/diabetic_data.csv')

# Data Cleaning
# Filtering out rows with missing weight values
filtered_data = diabetes_data[diabetes_data['weight'] != '?']
filtered_data.head()  # Display the first few rows of the filtered DataFrame

# Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=20, kde=True)
plt.title('Age Distribution of Patients')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 7))
sns.countplot(x='age', hue='readmitted', data=df, palette='viridis')
plt.title('Readmission Rates by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='gender', hue='readmitted', data=df, palette='mako')
plt.title('Readmission Rates by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['time_in_hospital'], bins=20, kde=True, color='orange')
plt.title('Distribution of Time in Hospital')
plt.xlabel('Time in Hospital (Days)')
plt.ylabel('Count')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['num_medications'], bins=30, kde=True, color='green')
plt.title('Number of Medications Distribution')
plt.xlabel('Number of Medications')
plt.ylabel('Count')
plt.grid(True)
plt.show()

import pandas as pd

# Load the dataset
diabetes_data = pd.read_csv('/diabetic_data.csv')

# Data Cleaning
# Filtering out rows with missing weight values
filtered_data = diabetes_data[diabetes_data['weight'] != '?']

# EDA
# Checking the unique values and their counts in the 'weight' and 'readmitted' columns
weight_values = diabetes_data['weight'].value_counts()
readmitted_values = diabetes_data['readmitted'].value_counts()

# Grouping by weight and calculating readmission rates for each weight category
readmission_by_weight = filtered_data.groupby('weight')['readmitted'].value_counts(normalize=True).unstack()

# Displaying the results
print("Weight Values Distribution:\n", weight_values, "\n")
print("Readmitted Values Distribution:\n", readmitted_values, "\n")
print("Readmission Rates by Weight Category:\n", readmission_by_weight)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('/diabetic_data.csv')

# Replacing '?' with NaN for missing values
data['weight'] = data['weight'].replace('?', np.nan)

# Creating a count plot for readmission counts by weight category
plt.figure(figsize=(12, 6))
sns.countplot(x='weight', hue='readmitted', data=data)
plt.title('Readmission Counts by Weight Category')
plt.xlabel('Weight Category')
plt.ylabel('Count of Readmissions')
plt.xticks(rotation=45)
plt.legend(title='Readmission')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
data = pd.read_csv('?diabetic_data.csv')

# Remove rows where 'race' is '?'
data = data[data['race'] != '?']

# Plotting readmission rates by race
plt.figure(figsize=(10, 6))
sns.countplot(x='race', hue='readmitted', data=data, palette='mako')
plt.title('Readmission Rates by Race')
plt.xlabel('Race')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate the x labels for better readability
plt.grid(axis='y')
plt.show()

# Select only numerical columns for correlation analysis
numerical_columns = df.select_dtypes(include=[np.number])

# Calculate the correlation matrix
correlation_matrix = numerical_columns.corr()

# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5, fmt='.2f')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

#pip3 install statsmodels
#pip3 install scipy

# Importing necessary libraries
import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import ztest  # Correcting the import for ztest
from scipy.stats import ttest_ind, f_oneway, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = ('diabetic_data.csv')
df = pd.read_csv(file_path)
df.head()

# Data Cleaning
df = df.replace('?', np.nan)
df['age'] = df['age'].str.extract('(\d+)-').astype(int)

# Displaying cleaned data
df.head()

# Groups for Z-Test
group1 = df[df['gender'] == 'Male']['time_in_hospital']
group2 = df[df['gender'] == 'Female']['time_in_hospital']

# Performing the Z-Test
z_stat, p_value = ztest(group1, group2)
z_stat, p_value

# Performing the T-Test
t_stat, p_value_ttest = ttest_ind(group1, group2, equal_var=False)  # Assuming unequal variances
t_stat, p_value_ttest

# Calculating the variances
var_group1 = np.var(group1, ddof=1)  # ddof=1 indicates sample variance
var_group2 = np.var(group2, ddof=1)

# F-Statistic
f_stat = var_group1 / var_group2
f_stat

import pandas as pd
from scipy.stats import chi2_contingency

# Load the dataset
data = pd.read_csv('diabetic_data.csv')

# Extracting relevant columns
gender_readmit_data = data[['gender', 'readmitted']]

# Removing rows with missing or invalid data in 'gender' or 'readmitted'
gender_readmit_data = gender_readmit_data[gender_readmit_data['gender'].isin(['Male', 'Female'])]
gender_readmit_data = gender_readmit_data[gender_readmit_data['readmitted'].isin(['NO', '<30', '>30'])]

# Creating a contingency table
contingency_table = pd.crosstab(gender_readmit_data['gender'], gender_readmit_data['readmitted'])

# Performing Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Printing the results
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies:")
print(expected)

# Interpreting the hypothesis
alpha = 0.05
if p < alpha:
    print("Result: Reject the Null Hypothesis (H0)")
    print("Interpretation: There is a statistically significant association between gender and readmission status.")
else:
    print("Result: Fail to Reject the Null Hypothesis (H0)")
    print("Interpretation: There is no statistically significant association between gender and readmission status.")

import pandas as pd
from scipy.stats import chi2_contingency

# Load the dataset
diabetes_data = pd.read_csv('diabetic_data.csv')

# Checking the unique values and their counts in the 'weight' and 'readmitted' columns
weight_values = diabetes_data['weight'].value_counts()
readmitted_values = diabetes_data['readmitted'].value_counts()

# Filtering out rows with missing weight values
filtered_data = diabetes_data[diabetes_data['weight'] != '?']

# Grouping by weight and calculating readmission rates for each weight category
readmission_by_weight = filtered_data.groupby('weight')['readmitted'].value_counts(normalize=True).unstack()

# Creating a contingency table for the chi-square test
contingency_table = filtered_data.groupby('weight')['readmitted'].value_counts().unstack().fillna(0)

# Performing the chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Results
chi2, p, dof, expected, readmission_by_weight

# Preparing data for ANOVA
age_groups = [df[df['age'] == age]['time_in_hospital'] for age in df['age'].unique()]

# Performing ANOVA
f_stat_anova, p_value_anova = f_oneway(*age_groups)
f_stat_anova, p_value_anova

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != 'readmitted':  # We will handle the target variable separately
        df[col] = le.fit_transform(df[col])
df['readmitted'] = df['readmitted'].map({'NO': 0, '<30': 1, '>30': 0})  # Binary encoding
df.head()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# Separating features and target variable

X = df.drop(columns=['readmitted', 'encounter_id', 'patient_nbr'])  # Dropping identifiers and target variable
y = df['readmitted']

# Splitting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initializing the Gradient Boosting Classifier and training the model
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Predicting the readmission on the test data and calculating the accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

accuracy, classification_rep

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting using Seaborn
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Predict probabilities
probabilities = model.predict_proba(X_test)[:, 1]

# Calculate precision and recall for various thresholds
precision, recall, thresholds = precision_recall_curve(y_test, probabilities)

# Plot the precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Separating features and target variable
X = df.drop(columns=['readmitted', 'encounter_id', 'patient_nbr'])  # Dropping identifiers and target variable
y = df['readmitted']

# Splitting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initializing the Support Vector Machine Classifier and training the model
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

# Predicting the readmission on the test data and calculating the accuracy
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
classification_rep_svm = classification_report(y_test, y_pred_svm)

print("SVM Accuracy:", accuracy_svm)
print("SVM Classification Report:\n", classification_rep_svm)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Compute the confusion matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)

# Plotting using Seaborn
plt.figure(figsize=(10,7))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Initialize the SVM with probability estimates
svm_model = SVC(random_state=42, probability=True)
svm_model.fit(X_train, y_train)

# Predict probabilities for the positive class
y_scores = svm_model.predict_proba(X_test)[:, 1]

# Calculate precision and recall for various thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Plot the precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.title('SVM Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Assuming X and y are the feature set and labels of your entire dataset
# Splitting the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# K-Nearest Neighbors (KNN) Model
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Predicting and Evaluating the model
y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
classification_rep_knn = classification_report(y_test, y_pred_knn)

# The print statements will output the accuracy and classification report
print("KNN Accuracy:", accuracy_knn)
print("KNN Classification Report:\n", classification_rep_knn)

plt.figure(figsize=(10,7))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues')
plt.title('KNN Confusion Matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# Calculate precision and recall
precision, recall, _ = precision_recall_curve(y_test, knn_model.predict_proba(X_test)[:, 1])

# Calculate the area under the precision-recall curve (AUC-PR)
auc_score = auc(recall, precision)

# Plot the precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label=f'KNN (AUC = {auc_score:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.grid(True)
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Seperating features and target variables
X = df.drop(['readmitted', 'encounter_id', 'patient_nbr'], axis=1)
y = df['readmitted']  # Target variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Making predictions
rf_predictions = rf_model.predict(X_test)

# Evaluating the model
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)
print(classification_report(y_test, rf_predictions))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming y_test and rf_predictions are the output of your Random Forest model
# Generate the confusion matrix
conf_matrix_rf = confusion_matrix(y_test, rf_predictions)

# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Compute precision and recall
precision, recall, _ = precision_recall_curve(y_test, rf_model.predict_proba(X_test)[:, 1])

# Create Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.step(recall, precision, where='mid', label='Random Forest Classifier', color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.legend(loc='best')

# Show the plot
plt.show()

import matplotlib.pyplot as plt

# Model names and accuracy scores
model_names = ['Gradient Boost Classifier', 'Support Vector Machine', 'K-Nearest Neighbours', 'Random Forest']
accuracies = [0.88857, 0.88842, 0.87859, 0.88778]

# Colors for each model
colors = ['blue', 'green', 'red', 'purple']

# Plotting the accuracies with enhancements for better visualization
plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, accuracies, color=colors)

# Adding data labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 5), ha='center', va='bottom')

plt.xlabel('Models')
plt.ylabel('Accuracy Score')
plt.title('Comparison of Model Accuracies')
plt.xticks(rotation=45)
plt.ylim([0.87, 0.89])  # Setting a tighter y-limit to emphasize the differences
plt.show()

# Model names and F1 scores (Replace these with your actual F1 scores)
model_names = ['Gradient Boost Classifier', 'Support Vector Machine', 'K-Nearest Neighbours', 'Random Forest']
f1_scores = [0.85, 0.83, 0.81, 0.84]  # Example F1 scores

# Colors for each model
colors = ['blue', 'green', 'red', 'purple']

# Plotting the F1 scores with enhancements for better visualization
plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, f1_scores, color=colors)

# Adding data labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 3), ha='center', va='bottom')

plt.xlabel('Models')
plt.ylabel('F1 Score')
plt.title('Comparison of Model based on F1 Scores')
plt.xticks(rotation=45)
plt.ylim([0.8, 0.86])  # Setting a tighter y-limit to emphasize the differences
plt.show()

