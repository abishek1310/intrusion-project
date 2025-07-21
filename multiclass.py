import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import pickle
from os import path

from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('UNSW_NB15.csv')

data.head(n=5)
print("Feature info")
data.info()

data[data['service']=='-']

data['service'].replace('-',np.nan,inplace=True)

data.isnull().sum()

data.shape
data.dropna(inplace=True)
data.shape

data['attack_cat'].value_counts()
data['state'].value_counts()

data

features = pd.read_csv('UNSW_NB15_features.csv')

features.head()

features['Type '] = features['Type '].str.lower()

# selecting column names of all data types
nominal_names = features['Name'][features['Type ']=='nominal']
integer_names = features['Name'][features['Type ']=='integer']
binary_names = features['Name'][features['Type ']=='binary']
float_names = features['Name'][features['Type ']=='float']

# selecting common column names from dataset and feature dataset
cols = data.columns
nominal_names = cols.intersection(nominal_names)
integer_names = cols.intersection(integer_names)
binary_names = cols.intersection(binary_names)
float_names = cols.intersection(float_names)

# Converting integer columns to numeric
for c in integer_names:
  pd.to_numeric(data[c])

# Converting binary columns to numeric
for c in binary_names:
  pd.to_numeric(data[c])

# Converting float columns to numeric
for c in float_names:
  pd.to_numeric(data[c])
print("Feature info1")
data.info()

data

# **Data Visualization**
plt.figure(figsize=(8,8))
plt.pie(data.label.value_counts(),labels=['normal','abnormal'],autopct='%0.2f%%')
plt.title("Pie chart distribution of normal and abnormal labels",fontsize=16)
plt.legend()
plt.savefig('plots/Pie_chart_binary.png')
plt.show()

plt.figure(figsize=(8,8))
plt.pie(data.attack_cat.value_counts(),labels=data.attack_cat.unique(),autopct='%0.2f%%')
plt.title('Pie chart distribution of multi-class labels')
plt.legend(loc='best')
plt.savefig('plots/Pie_chart_multi.png')
plt.show()

# **One hot encoding**
num_col = data.select_dtypes(include='number').columns

# selecting categorical data attributes
cat_col = data.columns.difference(num_col)
cat_col = cat_col[1:]
cat_col

# creating a dataframe with only categorical attributes
data_cat = data[cat_col].copy()
data_cat.head()

# one-hot-encoding categorical attributes using pandas.get_dummies() function
data_cat = pd.get_dummies(data_cat,columns=cat_col)

data_cat.head()

data.shape
data = pd.concat([data, data_cat],axis=1)
data.shape
data.drop(columns=cat_col,inplace=True)
data.shape

# **Data Normalization**
num_col = list(data.select_dtypes(include='number').columns)
num_col.remove('id')
num_col.remove('label')
print(num_col)

# using minmax scaler for normalizing data
minmax_scale = MinMaxScaler(feature_range=(0, 1))
def normalization(df,col):
  for i in col:
    arr = df[i]
    arr = np.array(arr)
    df[i] = minmax_scale.fit_transform(arr.reshape(len(arr),1))
  return df

data = normalization(data.copy(),num_col)
data.head()

# **Label Encoding**

# changing attack labels into two categories 'normal' and 'abnormal'
bin_label = pd.DataFrame(data.label.map(lambda x:'normal' if x==0 else 'abnormal'))

# creating a dataframe with binary labels (normal,abnormal)
bin_data = data.copy()
bin_data['label'] = bin_label

# label encoding (0,1) binary labels
le1 = preprocessing.LabelEncoder()
enc_label = bin_label.apply(le1.fit_transform)
bin_data['label'] = enc_label

le1.classes_
np.save("le1_classes.npy",le1.classes_,allow_pickle=True)

# **Multi-class Labels**
multi_data = data.copy()
multi_label = pd.DataFrame(multi_data.attack_cat)

multi_data = pd.get_dummies(multi_data,columns=['attack_cat'])

# label encoding (0,1,2,3,4,5,6,7,8) multi-class labels
le2 = preprocessing.LabelEncoder()
enc_label = multi_label.apply(le2.fit_transform)
multi_data['label'] = enc_label

le2.classes_
np.save("le2_classes.npy",le2.classes_,allow_pickle=True)

num_col.append('label')

# **Correlation Matrix for Multi-class Labels**
num_col = list(multi_data.select_dtypes(include='number').columns)

# Correlation Matrix for Multi-class Labels
plt.figure(figsize=(20,8))
corr_multi = multi_data[num_col].corr()
sns.heatmap(corr_multi,vmax=1.0,annot=False)
plt.title('Correlation Matrix for Multi Labels',fontsize=16)
plt.savefig('plots/correlation_matrix_multi.png')
plt.show()

# **Feature Selection**

multi_cols = multi_data.columns
multi_data = multi_data[multi_cols].copy()

# **Data Splitting**
X = multi_data.drop(columns=['label'],axis=1)
Y = multi_data['label']

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.20, random_state=100)

# Assuming X_test is already defined
np.savetxt('X_test.txt', X_test)

# Initialize the models
adaboost = AdaBoostClassifier(n_estimators=100, random_state=123)
adaboost.fit(X_train, y_train)

# Evaluate AdaBoost
y_pred = adaboost.predict(X_test)
print("****************** **AdaBoost Classifier***************************")
print("Mean Absolute Error - ", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error - ", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error - ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score - ", metrics.explained_variance_score(y_test, y_pred) * 100)
print("Accuracy - ", accuracy_score(y_test, y_pred) * 100)
print(classification_report(y_test, y_pred, target_names=le2.classes_))

# Plot between Real and Predicted Data for AdaBoost
plt.figure(figsize=(20,8))
plt.plot(y_pred[:200], label="predictions", linewidth=2.0, color='blue')
plt.plot(y_test[:200].values, label="real_values", linewidth=2.0, color='lightcoral')
plt.legend(loc="best")
plt.title("AdaBoost Classifier Multi-class Classification")
plt.savefig('plots/adaboost_real_pred_multi.png')
plt.show()

# Initialize LightGBM
lightgbm = GradientBoostingClassifier(n_estimators=100, random_state=123)
lightgbm.fit(X_train, y_train)

# Evaluate LightGBM
y_pred = lightgbm.predict(X_test)
print("****************** **LightGBM Classifier***************************")
print("Mean Absolute Error - ", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error - ", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error - ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score - ", metrics.explained_variance_score(y_test, y_pred) * 100)
print("Accuracy - ", accuracy_score(y_test, y_pred) * 100)
print(classification_report(y_test, y_pred, target_names=le2.classes_))

# Plot between Real and Predicted Data for LightGBM
plt.figure(figsize=(20,8))
plt.plot(y_pred[:200], label="predictions", linewidth=2.0, color='blue')
plt.plot(y_test[:200].values, label="real_values", linewidth=2.0, color='lightcoral')
plt.legend(loc="best")
plt.title("LightGBM Classifier Multi-class Classification")
plt.savefig('plots/lightgbm_real_pred_multi.png')
plt.show()

# Initialize QDA
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

# Evaluate QDA
y_pred = qda.predict(X_test)
print("****************** **Quadratic Discriminant Analysis (QDA)***************************")
print("Mean Absolute Error - ", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error - ", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error - ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score - ", metrics.explained_variance_score(y_test, y_pred) * 100)
print("Accuracy - ", accuracy_score(y_test, y_pred) * 100)
print(classification_report(y_test, y_pred, target_names=le2.classes_))

# Plot between Real and Predicted Data for QDA
plt.figure(figsize=(20,8))
plt.plot(y_pred[:200], label="predictions", linewidth=2.0, color='blue')
plt.plot(y_test[:200].values, label="real_values", linewidth=2.0, color='lightcoral')
plt.legend(loc="best")
plt.title("QDA Multi-class Classification")
plt.savefig('plots/qda_real_pred_multi.png')
plt.show()

# Initialize SVM with One-vs-Rest
svc_ovr = SVC(kernel='linear', gamma='auto', decision_function_shape='ovr', probability=True)
svc=SVC()
svc.fit(X_train, y_train)


# Evaluate SVM (OvR)
y_pred = svc.predict(X_test)
print("****************** **Support Vector Machine (OvR)***************************")
print("Mean Absolute Error - ", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error - ", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error - ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score - ", metrics.explained_variance_score(y_test, y_pred) * 100)
print("Accuracy - ", accuracy_score(y_test, y_pred) * 100)
print(classification_report(y_test, y_pred, target_names=le2.classes_))

# Plot between Real and Predicted Data for SVM (OvR)
plt.figure(figsize=(20,8))
plt.plot(y_pred[:200], label="predictions", linewidth=2.0, color='blue')
plt.plot(y_test[:200].values, label="real_values", linewidth=2.0, color='lightcoral')
plt.legend(loc="best")
plt.title("SVM (OvR) Multi-class Classification")
plt.savefig('plots/svm_ovr_real_pred_multi.png')
plt.show()

# Initialize Bagging Classifier
bagging = BaggingClassifier(n_estimators=50, random_state=123)
bagging.fit(X_train, y_train)

# Evaluate Bagging
y_pred = bagging.predict(X_test)
print("****************** **Bagging Classifier***************************")
print("Mean Absolute Error - ", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error - ", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error - ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score - ", metrics.explained_variance_score(y_test, y_pred) * 100)
print("Accuracy - ", accuracy_score(y_test, y_pred) * 100)
print(classification_report(y_test, y_pred, target_names=le2.classes_))

# Plot between Real and Predicted Data for Bagging
plt.figure(figsize=(20,8))
plt.plot(y_pred[:200], label="predictions", linewidth=2.0, color='blue')
plt.plot(y_test[:200].values, label="real_values", linewidth=2.0, color='lightcoral')
plt.legend(loc="best")
plt.title("Bagging Classifier Multi-class Classification")
plt.savefig('plots/bagging_real_pred_multi.png')
plt.show()

# Define the base classifiers for stacking
estimators = [
    ('adaboost', adaboost),
    ('lightgbm', lightgbm),
    ('qda', qda),
    ('svc', svc_ovr),
    ('bagging', bagging)
]

# Define the meta-classifier
meta_classifier = LogisticRegression(random_state=123, max_iter=5000, solver='newton-cg', multi_class='multinomial')

# Create the Stacking Classifier
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=meta_classifier)

# Train the Stacking Classifier
stacking_clf.fit(X_train, y_train)

# Make predictions with Stacking Classifier
y_pred = stacking_clf.predict(X_test)

# Print metrics for Stacking Classifier
print("****************** **Stacking Classifier***************************")
print("Mean Absolute Error - ", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error - ", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error - ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score - ", metrics.explained_variance_score(y_test, y_pred) * 100)
print("Accuracy - ", accuracy_score(y_test, y_pred) * 100)
print(classification_report(y_test, y_pred, target_names=le2.classes_))

# Plot between Real and Predicted Data for Stacking Classifier
plt.figure(figsize=(20,8))
plt.plot(y_pred[:200], label="predictions", linewidth=2.0, color='blue')
plt.plot(y_test[:200].values, label="real_values", linewidth=2.0, color='lightcoral')
plt.legend(loc="best")
plt.title("Stacking Classifier Multi-class Classification")
plt.savefig('plots/stacking_real_pred_multi.png')
plt.show()

# Save model to disk
pkl_filename = "./models/stacking_classifier.pkl"
if not path.isfile(pkl_filename):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(stacking_clf, file)
    print("Saved model to disk")
else:
    print("Model already saved")

# Accuracy of different classifiers
models = ['AdaBoost', 'LightGBM', 'QDA', 'SVM (OvR)', 'Bagging', 'Stacking']
accuracies = [
    accuracy_score(y_test, adaboost.predict(X_test)) * 100,
    accuracy_score(y_test, lightgbm.predict(X_test)) * 100,
    accuracy_score(y_test, qda.predict(X_test)) * 100,
    accuracy_score(y_test, svc_ovr.predict(X_test)) * 100,
    accuracy_score(y_test, bagging.predict(X_test)) * 100,
    accuracy_score(y_test, stacking_clf.predict(X_test)) * 100
]

# Plotting the accuracies
plt.figure(figsize=(12, 6))
sns.barplot(x=models, y=accuracies, palette='viridis')
plt.title('Accuracy Comparison of Different Models')
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.savefig('plots/model_accuracy_comparison.png')
plt.show()
