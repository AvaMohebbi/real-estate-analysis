import numpy as np
import pandas as pd
from imblearn.metrics import specificity_score
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from xgboost import XGBRFClassifier
from sklearn.metrics import accuracy_score
from prettytable import PrettyTable
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, precision_score, recall_score, \
    f1_score, mean_squared_error, silhouette_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, StratifiedKFold
import statsmodels.api as sm
from tabulate import tabulate
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import DBSCAN
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import logging
logging.getLogger("pycharm_display").setLevel(logging.CRITICAL)

#show all columns
pd.set_option("display.max_columns", None)
#format numbers
pd.set_option('display.float_format', '{:.3f}'.format)

#import data
df = pd.read_csv('/Users/avamohebbi/Documents/MENg/Machine Learning I/Project Proposal/realtor-data.zip.csv')
print(df.head())
print(df.shape)

#PHASE 1: EDA

#check/drop duplicates
duplicates = df.duplicated()
num_duplicates = duplicates.sum()
print("number of duplicates", num_duplicates)

#describe data
print("Dataset Describe:")
print(df.describe())

#check number of values
print('number of unique values:')
print(df.nunique())

#check for missing values in data
print("Missing Data Percentages:")
print()
missing_val_df = (df.isna().sum() / len(df)) * 100
missing_val_df = missing_val_df.reset_index()
missing_val_df.rename(columns={'index': 'Feature', 0: 'Percent Missing'}, inplace=True)
print(missing_val_df)

print(tabulate(missing_val_df, headers='keys', tablefmt='psql'))

missing_val_df.sort_values(by='Percent Missing', inplace=True, ascending=False)

#create table of
print(tabulate(missing_val_df, headers='keys', tablefmt='psql'))

#impute missing values
df['brokered_by'] = df['brokered_by'].fillna(df['brokered_by'].mode()[0])
df['status'] = df['status'].fillna(df['status'].mode()[0])
df['price'] = df['price'].fillna(df['price'].median())
df['bed'] = df['bed'].fillna(df['bed'].median())
df['bath'] = df['bath'].fillna(df['bath'].median())
df['acre_lot'] = df['acre_lot'].fillna(df['acre_lot'].median())
df['street'] = df['street'].fillna(df['street'].mode()[0])
df['city'] = df['city'].fillna(df['city'].mode()[0])
df['state'] = df['state'].fillna(df['state'].mode()[0])
df['zip_code'] = df['zip_code'].fillna(df['zip_code'].mode()[0])
df['house_size'] = df['house_size'].fillna(df['house_size'].median())
df['prev_sold_date'] = df['prev_sold_date'].fillna(df['prev_sold_date'].mode()[0])

#print message to confirm data is clean
if df.isnull().values.any() == False:
    print('CONFIRMED: DATASET IS CLEANED')
else:
    print('THERE ARE NULL VALUES REMAINING IN DATASET')

#Use IQR to determine low, medium, high price
Quantile_25 = np.quantile(df['price'], 0.25)
Quantile_75 = np.quantile(df['price'], 0.75)
IQR = Quantile_75 - Quantile_25

def categorize_price(x):
    if x < Quantile_25:
        return 'Low Price'
    elif x > Quantile_75:
        return 'High Price'
    else:
        return 'Medium Price'

df['price category'] = df['price'].apply(categorize_price)
print('Dataframe after discretization')
print(df.head())

#downsample data
train_sample_df, test_sample_df = train_test_split(df, test_size=0.03, stratify = df['price category'], random_state=42)
sample_df = test_sample_df.copy()
print('Downsampled Head')
print(sample_df.head())
print(sample_df.head())
print('Downsampled Shape')
print(sample_df.shape)

#drop the price category after downsampling
sample_df.drop(columns=['price category'], inplace=True)

#VISUALIZE DATA HERE

sample_df_num = sample_df[['price', 'bed', 'bath', 'acre_lot','house_size']]

# sns.set(rc = {'figure.figsize':(20,20)})
fig, ax = plt.subplots(figsize=(20, 20))
sns.boxplot(data=sample_df_num)
plt.title('Box Plot For Numerical Features')
plt.yscale('log')
plt.show()

state_counts = sample_df['state'].value_counts()
plt.figure(figsize=(12, 8))
sns.countplot(data=sample_df, x='state')
plt.xticks(rotation=90)
plt.title("State Counts")
plt.tight_layout()
plt.show()

status_counts = sample_df['status'].value_counts()
plt.figure(figsize=(12, 8))
sns.countplot(data=sample_df, x='status')
plt.xticks(rotation=90)
plt.title("Status Counts")
plt.tight_layout()
plt.show()

#drop unneccesary features
sample_df.drop(columns=['street'], axis=1, inplace=True)

#list of numeric features
sample_df_numeric = sample_df[['price', 'bed', 'bath', 'acre_lot', 'house_size']]

#list of categorical features
categorical_features = ['brokered_by', 'status', 'city', 'state', 'zip_code']

#label encoding for categorical features
le_brokered_by = LabelEncoder()
le_zip_code = LabelEncoder()
le_city = LabelEncoder()

sample_df['brokered_by_encoded'] = le_brokered_by.fit_transform(sample_df['brokered_by'])
sample_df['zip_code_encoded'] = le_zip_code.fit_transform(sample_df['zip_code'])
sample_df['city_encoded'] = le_city.fit_transform(sample_df['city'])

sample_df['prev_sold_date'] = pd.to_datetime(sample_df['prev_sold_date'])
sample_df['month'] = sample_df['prev_sold_date'].dt.month
sample_df['year'] = sample_df['prev_sold_date'].dt.year

#copy dataframe to use for apriori
sample_df_arm = sample_df.copy()

year_mapping = {1901: 0, 1910: 1, 1952: 2, 1964: 3, 1966: 4, 1967: 5, 1968: 6, 1969: 7, 1970: 8, 1971: 9,
 1972: 10, 1973: 11, 1974: 12, 1975: 13, 1976: 14, 1977: 15, 1978: 16, 1979: 17, 1980: 18,
 1981: 19, 1982: 20, 1983: 21, 1984: 22, 1985: 23, 1986: 24, 1987: 25, 1988: 26, 1989: 27,
 1990: 28, 1991: 29, 1992: 30, 1993: 31, 1994: 32, 1995: 33, 1996: 34, 1997: 35, 1998: 36,
 1999: 37, 2000: 38, 2001: 39, 2002: 40, 2003: 41, 2004: 42, 2005: 43, 2006: 44, 2007: 45,
 2008: 46, 2009: 47, 2010: 48, 2011: 49, 2012: 50, 2013: 51, 2014: 52, 2015: 53, 2016: 54,
 2017: 55, 2018: 56, 2019: 57, 2020: 58, 2021: 59, 2022: 60, 2023: 61}

sample_df["year_encoded"] = sample_df["year"].map(year_mapping)

#one hot encoding for state/status
sample_df = pd.get_dummies(sample_df, columns=['status', 'state', 'month'], dtype=int, drop_first=True)

print('sample df head after encoding:')
print(sample_df.head())

#drop the original categorical columns
sample_df.drop(columns=['brokered_by', 'city', 'zip_code', 'prev_sold_date', 'year'], inplace=True)

print('sample df columns:', sample_df.columns)

sample_df_scaled = sample_df.copy()

numerical_features_lof = ['bed', 'bath', 'acre_lot','house_size']
# numerical_features = ['acre_lot','house_size']
scaler_lof = StandardScaler()

sample_df_scaled[numerical_features_lof] = scaler_lof.fit_transform(sample_df_scaled[numerical_features_lof])

#nomalize the label encoded data
label_encoded_features_lof = ['zip_code_encoded','city_encoded', 'brokered_by_encoded']
Label_Encoded_Scaler_lof = MinMaxScaler()
sample_df_scaled[label_encoded_features_lof] = Label_Encoded_Scaler_lof.fit_transform(sample_df_scaled[label_encoded_features_lof])

#outlier detection with LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
y_pred = lof.fit_predict(sample_df_scaled)
sample_df['outlier'] = y_pred
sample_df['LOF_score'] = lof.negative_outlier_factor_
outliers = sample_df[sample_df['LOF_score'] < -2.5]
print(outliers.head())
print('outliers shape', outliers.shape)

sample_df.drop(sample_df[sample_df['LOF_score'] < -2.5].index, inplace=True)
print('outliers removed head', sample_df.head())

#go back and drop the outlier columns
sample_df.drop(columns=['LOF_score', 'outlier'], inplace=True)
print('df with outliers removed head:', sample_df.head())
print('df with outliers removed shape:', sample_df.shape)


#split the data into dependent and independent variables (unscaled data)
X = sample_df.drop('price', axis=1)
y = sample_df['price']

#scale the data
numerical_features = ['bed', 'bath', 'acre_lot','house_size']
# numerical_features = ['acre_lot','house_size']
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numerical_features] = scaler.fit_transform(X_scaled[numerical_features])

#nomalize the label encoded data
label_encoded_features = ['zip_code_encoded','city_encoded', 'brokered_by_encoded']
Label_Encoded_Scaler = MinMaxScaler()
X_scaled = X.copy()
X_scaled[label_encoded_features] = Label_Encoded_Scaler.fit_transform(X_scaled[label_encoded_features])

print(X_scaled.head())

# # #1: random forest
print('RANDOM FOREST FEATURE SELECTION')
random_forest_model = RandomForestRegressor(random_state=1)
random_forest_model.fit(X, y)
features = sample_df.columns
importances = random_forest_model.feature_importances_
indices = np.argsort(importances)[-77:]
print('Random Forest Indices', [features[i] for i in indices])

plt.figure(figsize=(10, 10))
plt.title('Random Forest Feature Importance')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()

feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances.copy()})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
print("Feature Importances Sorted")
print(feature_importance_df.to_string())

not_important_features = []

for index, row in feature_importance_df.iterrows():
    if row['importance'] < 0.001:
        not_important_features.append(row['feature'])

print('not important features:', not_important_features)

#
# #
# # #2:PCA
print("PRINCIPAL COMPONENT ANALYSIS")
#create instance of PCA
pca = PCA(n_components='mle', svd_solver='full')

#computes principle components from scaled data
pca.fit(X_scaled)
X_PCA = pca.transform(X_scaled)
print('explained variance', pca.explained_variance_ratio_)
explained_var_cum = pca.explained_variance_ratio_.cumsum()
num_component_over_95 = np.argmax(explained_var_cum >= 0.95) + 1
print("number of components that explain over 95% of variance:", num_component_over_95)
#
#Plot PCA
plt.figure(figsize=(20, 10))
plt.plot(np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1, 1), np.cumsum(pca.explained_variance_ratio_))
plt.xticks(np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1, 1))
plt.axhline(y=0.95, color='r', linestyle='--')
plt.axvline(x=num_component_over_95, color='b', linestyle='--')
plt.grid()
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title('PCA')
plt.show()

covariance_matrix_pca = pca.get_covariance()
condition_number = np.linalg.cond(covariance_matrix_pca)
print("Condition number:", condition_number)

# #
# #3: SVD
print('SINGULAR VALUE DECOMPOSITION ANALYSIS')
svd = TruncatedSVD(n_components=5, random_state=42)
svd.fit(X_scaled)
print('svd explained variance ratio', svd.explained_variance_ratio_)
print('svd explained variance ratio sum', svd.explained_variance_ratio_.sum())
print('svd singular values', svd.singular_values_)

#get feature importance scores and sort the features by importance
features_SVD = X_scaled.columns
feature_importance = np.sum(np.abs(svd.components_), axis=0)
sorted_indices = np.argsort(feature_importance)[-77:]

plt.figure(figsize=(10, 10))
plt.title('SVD Feature Importance')
plt.barh(range(len(sorted_indices)), feature_importance[sorted_indices], color='b', align='center')
plt.yticks(range(len(sorted_indices)), [features_SVD[i] for i in sorted_indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()
#
# # #4: VIF (Variance Inflation Factor)

#create new df for VIF
df_VIF = sample_df.copy()

#drop the target column, and keep independent variables
df_VIF = df_VIF.drop('price', axis=1)

#create a new table to display the VIF outputs
VIF_table = pd.DataFrame()
VIF_table['Feature'] = df_VIF.columns
VIF_table['VIF'] = [variance_inflation_factor(df_VIF.values, i) for i in range(len(df_VIF.columns))]
VIF_table_sorted = VIF_table.sort_values(by='VIF', ascending=False)
print('VIF Table')
print(VIF_table_sorted.to_string(index=False))

# #covariance heatmap
# plt.figure(figsize=(200,200))
covar_heat_map = sns.heatmap(sample_df.cov(), cmap="YlGnBu", annot=True)
# plt.title("Covariance Matrix")
# plt.savefig("CovarianceMatrix.pdf", format="pdf", bbox_inches="tight")
# plt.show()
#
# #correlation heat map
#
# plt.figure(figsize=(200,200))
corr_heat_map = sns.heatmap(sample_df.corr(), cmap="YlGnBu", annot=True)
# plt.title('Correlation Heatmap')
# plt.savefig("CorrelationMatrix.pdf", format="pdf", bbox_inches="tight")
# plt.show()


#drop collinear featuresc
sample_df.drop(columns=['zip_code_encoded', 'state_California',
                        'state_Texas', 'state_Washington', 'house_size',
                        'state_Arizona', 'bed', 'year_encoded', 'state_Oregon',
                        'month_3', 'state_Illinois', 'state_Florida', 'state_Colorado',
                        'state_Oklahoma', 'state_Missouri', 'state_Vermont', 'state_Delaware',
                        'state_Puerto Rico', 'state_Hawaii', 'state_North Dakota', 'state_Wyoming',
                        'state_South Dakota', 'state_Maine', 'state_Nevada', 'state_Kansas', 'state_Idaho',
                        'state_Utah', 'state_Rhode Island', 'state_Connecticut', 'state_District of Columbia'], inplace=True)

X.drop(columns=['zip_code_encoded', 'state_California',
                        'state_Texas', 'state_Washington', 'house_size',
                        'state_Arizona', 'bed', 'year_encoded', 'state_Oregon',
                        'month_3', 'state_Illinois', 'state_Florida', 'state_Colorado',
                        'state_Oklahoma', 'state_Missouri', 'state_Vermont', 'state_Delaware',
                        'state_Puerto Rico', 'state_Hawaii', 'state_North Dakota', 'state_Wyoming',
                        'state_South Dakota', 'state_Maine', 'state_Nevada', 'state_Kansas', 'state_Idaho',
                        'state_Utah', 'state_Rhode Island', 'state_Connecticut', 'state_District of Columbia'], inplace=True)

df_cluster = sample_df.copy()


#Backward Stepwise Regression

#create table for stepwise regression
stepwise_reg_table = PrettyTable(['Step', 'AIC', 'BIC', 'Adjusted R-Square', 'Feature with Max. p-value [DROP]', 'p-value'])
stepwise_reg_table.title = 'Backward Stepwise Regression Table'

#train test split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5805, shuffle=True)

updated_numerical_features = ['acre_lot', 'bath']
updated_label_encoded_features = ['city_encoded', 'brokered_by_encoded']

#scale numerical columns
scaler_reg = StandardScaler()
X_train[updated_numerical_features] = scaler_reg.fit_transform(X_train[updated_numerical_features])
X_test[updated_numerical_features] = scaler_reg.fit_transform(X_test[updated_numerical_features])
Label_Encoded_Scaler = MinMaxScaler()
X_train[updated_label_encoded_features] = Label_Encoded_Scaler.fit_transform(X_train[updated_label_encoded_features])
X_test[updated_label_encoded_features] = Label_Encoded_Scaler.fit_transform(X_test[updated_label_encoded_features])

#print train/test sets
print("First 5 rows of the training set (X):")
print(X_train.head())
print('X_train shape:', X_train.shape)

print("First 5 rows of the test set (X):")
print(X_test.head())
print('X_test shape:', X_test.shape)

X_test = sm.add_constant(X_test)

#create function for stepwise regression
def backward_stepwise_reg(training_set, target_set):
    step = 1
    while True:
        training_set = sm.add_constant(training_set)
        model1 = sm.OLS(target_set, training_set).fit()
        print('Results 1:', model1.summary())
        stepwise_reg_table.add_row([step, model1.aic, model1.bic, model1.rsquared_adj, model1.pvalues.idxmax(),
                       model1.pvalues.loc[model1.pvalues.idxmax()]])
        if model1.pvalues.loc[model1.pvalues.idxmax()] > 0.05:
            training_set.drop(columns = model1.pvalues.idxmax(), inplace = True)
            step += 1
        else:
            break

#implmenet and print backward stepwise regression
backward_stepwise_reg(X_train, y_train)
print(stepwise_reg_table)

#develop final regression model
X_train = sm.add_constant(X_train)
final_regression_model = sm.OLS(y_train, X_train).fit()
print(final_regression_model.summary())
final_regression_predictions = final_regression_model.predict(X_test)
final_regression_predictions_train = final_regression_model.predict(X_train)
final_model_mse = mean_squared_error(y_test, final_regression_predictions)

# #create table for final regression model
final_regression_table = PrettyTable(['AIC', 'BIC', 'R-Square', 'Adjusted R-Square', 'MSE'])
final_regression_table.title = 'Final Regression Model'
final_regression_table.add_row([final_regression_model.aic, final_regression_model.bic, final_regression_model.rsquared, final_regression_model.rsquared_adj, final_model_mse])

print(final_regression_table)


#F-Statistic Analysis
f_stat = final_regression_model.fvalue
f_stat_p_value = final_regression_model.f_pvalue
print('F-Statistic', f_stat)
print('p-value', f_stat_p_value)

if f_stat_p_value < 0.05:
    print('p value for f-statistic is less than 0.05 so reject the null hypothesis (the model is significant)')
else:
    print('p value for f-statistic is more than 0.05 so fail to reject the null hypothesis')


# plt.figure(figsize = (10, 10))
plt.title('Final Regression Model [TEST]')
plt.scatter(y_test, final_regression_predictions, s=1, color='blue')


max_val = max(np.max(y_test), np.max(final_regression_predictions))
min_val = min(np.min(y_test), np.min(final_regression_predictions))
plt.xlabel('Actual Value [Test]')
plt.ylabel('Predicted Values')
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.show()


# plt.figure(figsize = (10, 10))
plt.title('Final Regression Model [TRAIN]')
plt.scatter(y_train, final_regression_predictions_train, s=1, color='blue')


max_val_train = max(np.max(y_train), np.max(final_regression_predictions_train))
min_val_train = min(np.min(y_train), np.min(final_regression_predictions_train))
plt.xlabel('Actual Value [Train]')
plt.ylabel('Predicted Values')
plt.plot([min_val_train, max_val_train], [min_val_train, max_val_train], 'r--')
plt.show()

#confidence interval analysis
confidence_intervals = final_regression_model.conf_int(alpha = 0.95)
print('Confidence Intervals', confidence_intervals)

final_regression_get_predictions = final_regression_model.get_prediction(X_test)
prediction_interval = final_regression_get_predictions.conf_int(alpha = 0.05)
prediction_mean = final_regression_get_predictions.predicted_mean

#get lower/upper bound of interval
lower_bound = prediction_interval[:,0]
upper_bound = prediction_interval[:,1]

#flatten
prediction_flat = prediction_mean.flatten()
lower_flat = lower_bound.flatten()
upper_flat = upper_bound.flatten()

#plot confidence intervals
plt.figure(figsize=(100, 8))
plt.title("Sales Prediction With Confidence Interval")
plt.xlabel("# of Samples")
plt.ylabel("Price")
plt.plot(prediction_flat, label='Predicted Price', color='blue', linewidth=0.25)
plt.fill_between(range(len(prediction_flat)), lower_flat, upper_flat, color='c', alpha=0.5, label = 'CI')
plt.grid()
plt.legend()
plt.savefig("ConfidenceInterval.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()

#classification
# #Use IQR to determine low or high price
median = np.median(sample_df['price'])
Quantile_25 = np.quantile(df['price'], 0.25)
Quantile_75 = np.quantile(df['price'], 0.75)
IQR = Quantile_75 - Quantile_25

def categorize_price(x):
    if x > median:
        return 'High Price'
    else:
        return 'Low Price'


sample_df['price category'] = sample_df['price'].apply(categorize_price)

print(sample_df.head())

sns.countplot(x=sample_df['price category'], data=sample_df)
plt.title('Price Category Distribution')
plt.show()

print('unbalanced data counts', sample_df['price category'].value_counts())

#encode the price category
price_map = {'Low Price': 0, 'High Price': 1}
le = LabelEncoder()
sample_df['price category'] = le.fit_transform(sample_df['price category'].map(price_map))

X_classification = sample_df.drop(['price', 'price category'], axis=1)
y_classification = sample_df['price category']

#train test split
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
  X_classification, y_classification, random_state=42,test_size=0.20, shuffle=True)

#use SMOTE to balance dataset
smote = SMOTE(random_state = 42)
X_train_class, y_train_class = smote.fit_resample(X_train_class, y_train_class)
print('Check if balanced:')
print(y_train_class.value_counts())
print('x balanced', X_train_class)
print('y balanced', y_train_class)

#plot balanced data
print('y class value counts', y_train_class.value_counts())

scaler_class = StandardScaler()
X_train_class[updated_numerical_features] = scaler_class.fit_transform(X_train_class[updated_numerical_features])
X_test_class[updated_numerical_features] = scaler_class.fit_transform(X_test_class[updated_numerical_features])
Label_Encoded_Scaler_class = MinMaxScaler()
X_train_class[updated_label_encoded_features] = Label_Encoded_Scaler_class.fit_transform(X_train_class[updated_label_encoded_features])
X_test_class[updated_label_encoded_features] = Label_Encoded_Scaler_class.fit_transform(X_test_class[updated_label_encoded_features])

clf_table = PrettyTable(['Classifier', 'Precision', 'Sensitivity/Recall', 'Specificity', 'F-score'])

#stratified k fold object
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#Naive Bayes Classifier
print('Naive Bayes')

NB_model = GaussianNB()
param_grid_NB = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}

#GridSearch
grid_search_NB = GridSearchCV(estimator=NB_model, param_grid=param_grid_NB, cv=skf, scoring='accuracy')
grid_search_NB.fit(X_train_class, y_train_class)

# #Best Model
best_NB_model = grid_search_NB.best_estimator_
y_pred_NB = best_NB_model.predict(X_test_class)
print("Best parameters [Naive Bayes]:", grid_search_NB.best_params_)
print("Best cross-validated accuracy [Naive Bayes]:", grid_search_NB.best_score_)

#accuracy for train/test set
y_pred_train_NB = best_NB_model.predict(X_train_class)
y_test_acc_NB = accuracy_score(y_test_class, y_pred_NB)
y_train_acc_NB = accuracy_score(y_train_class, y_pred_train_NB)
print('accuracy on train data [naive bayes]', y_train_acc_NB)
print('accuracy on test data [naive bayes]', y_test_acc_NB)

# #confusion matrix
cm_NB = confusion_matrix(y_test_class, y_pred_NB)
display_NB = ConfusionMatrixDisplay(confusion_matrix=cm_NB)
display_NB.plot()
plt.title('Naive Bayes Confusion Matrix')
plt.show()

#print precision, recall, f1
precision_NB = precision_score(y_test_class, y_pred_NB)
print("Naive Bayes Precision:", precision_NB)
recall_NB = recall_score(y_test_class, y_pred_NB)
print("Naive Bayes Recall:", recall_NB)
specificity_NB = specificity_score(y_test_class, y_pred_NB)
print("Naive Bayes Specificity:", specificity_NB )
F1_NB = f1_score(y_test_class, y_pred_NB)
print("Naive Bayes F1:", F1_NB)

#add row to table
clf_table.add_row(["Naive Bayes", precision_NB, recall_NB, specificity_NB, F1_NB])

#ROC
RocCurveDisplay.from_estimator(best_NB_model, X_test_class, y_test_class)
y_prob_test_NB = best_NB_model.predict_proba(X_test_class)[:, 1]
plt.title('Naive Bayes ROC')
plt.plot([0, 1], [0, 1], '--', color='grey', label='Chance Level (AUC=0.5)')
plt.legend(loc='lower right')
plt.show()


#
#Decision Tree Classifier
print('Decision Tree')
DT_model = DecisionTreeClassifier(random_state=42)

DT_model.fit(X_train_class, y_train_class)

y_pred_DT = DT_model.predict(X_test_class)

y_pred_train_DT = DT_model.predict(X_train_class)
y_test_acc_DT = accuracy_score(y_test_class, y_pred_DT)
y_train_acc_DT = accuracy_score(y_train_class, y_pred_train_DT)
print('accuracy on train data [DT]', y_train_acc_DT)
print('accuracy on test data [DT]', y_test_acc_DT)


#confusion matrix
# #confusion matrix
cm_DT = confusion_matrix(y_test_class, y_pred_DT)
display_DT = ConfusionMatrixDisplay(confusion_matrix=cm_DT)
display_DT.plot()
plt.title('Decision Tree Confusion Matrix')
plt.show()

#print precision, recall, f1
precision_DT = precision_score(y_test_class, y_pred_DT)
print("Decision Tree Precision:", precision_DT)
recall_DT = recall_score(y_test_class, y_pred_DT)
print("Decision Tree Recall:", recall_DT)
specificity_DT = specificity_score(y_test_class, y_pred_DT)
print("Decision Tree Specificity:", specificity_DT)
F1_DT = f1_score(y_test_class, y_pred_DT)
print("Decision Tree F1:", F1_DT)
#


#add row to table
clf_table.add_row(["Decision Tree", precision_DT, recall_DT, specificity_DT, F1_DT])

#ROC
RocCurveDisplay.from_estimator(DT_model, X_test_class, y_test_class)
y_prob_test_DT = DT_model.predict_proba(X_test_class)[:, 1]
plt.plot([0, 1], [0, 1], '--', color='grey', label='Chance Level (AUC=0.5)')
plt.title('Decision Tree [No Pruning] ROC')
plt.legend(loc='lower right')
plt.show()
#
#DT MODEL PREPRUNING
param_grid_DT_Prepruning = [{'max_depth': [1, 2, 3, 4, 5],
                     'min_samples_split': [20,30,40],
                     'min_samples_leaf': [10,20,30],
                     'criterion':['gini','entropy','log_loss'],
                     'splitter':['best','random'],
                     'max_features':['sqrt','log2']}]

#GridSearch
grid_search_DT_Prepruning = GridSearchCV(DT_model,
                           param_grid=param_grid_DT_Prepruning,
                           cv=skf,
                           scoring='accuracy')

grid_search_DT_Prepruning.fit(X_train_class, y_train_class)

#Best Model
best_DT_model_Prepruning = grid_search_DT_Prepruning.best_estimator_
y_pred_DT_Prepruning = best_DT_model_Prepruning.predict(X_test_class)
print("Best parameters [Decision Tree Prepruning]:", grid_search_DT_Prepruning.best_params_)
print("Best cross-validated accuracy [Decision Tree Prepruning]:", grid_search_DT_Prepruning.best_score_)

y_pred_train_DT_Pre = best_DT_model_Prepruning.predict(X_train_class)
y_test_acc_DT_pre = accuracy_score(y_test_class, y_pred_DT_Prepruning)
y_train_acc_DT_pre = accuracy_score(y_train_class, y_pred_train_DT_Pre)
print('accuracy on train data [DT Prepruning]', y_train_acc_DT_pre)
print('accuracy on test data [DT Prepruning]', y_test_acc_DT_pre)

#confusion matrix
# #confusion matrix
cm_DT_Prepruning = confusion_matrix(y_test_class, y_pred_DT_Prepruning)
display_DT_Prepruning = ConfusionMatrixDisplay(confusion_matrix=cm_DT_Prepruning)
display_DT_Prepruning.plot()
plt.title('Decision Tree Prepruning')
plt.show()

#print precision, recall, f1
precision_DT_Prepruning = precision_score(y_test_class, y_pred_DT_Prepruning)
print("Decision Tree Precision [Prepruning]:", precision_DT_Prepruning)
recall_DT_Prepruning = recall_score(y_test_class, y_pred_DT_Prepruning)
print("Decision Tree Recall [Prepruning]:", recall_DT_Prepruning)
specificity_DT_Prepruning = specificity_score(y_test_class, y_pred_DT_Prepruning)
print("Decision Tree Specificity [Prepruning]:", specificity_DT_Prepruning)
F1_DT_Prepruning = f1_score(y_test_class, y_pred_DT_Prepruning)
print("Decision Tree F1:", F1_DT_Prepruning)

#add row to table
clf_table.add_row(["Decision Tree [Prepruning]", precision_DT_Prepruning, recall_DT_Prepruning,
                   specificity_DT_Prepruning, F1_DT_Prepruning])

#ROC
RocCurveDisplay.from_estimator(best_DT_model_Prepruning, X_test_class, y_test_class)
y_prob_test_DT_Prepruning = best_DT_model_Prepruning.predict_proba(X_test_class)[:, 1]
plt.plot([0, 1], [0, 1], '--', color='grey', label='Chance Level (AUC=0.5)')
plt.title('Decision Tree [Pre Pruning] ROC')
plt.legend(loc='lower right')
plt.show()

# #Decision True Post Pruning
path = DT_model.cost_complexity_pruning_path(X_train_class, y_train_class)
alphas = path['ccp_alphas']
print('alphas', alphas)
alfa = np.linspace(0,1,100)
param_grid_DT_Postpruning = {'ccp_alpha': alfa,
                  'criterion': ['gini', 'entropy'],
                  'splitter': ['best'],
                  'max_features':['sqrt']
                  }

#GridSearch
grid_search_DT_Postpruning = GridSearchCV(DT_model,
                           param_grid=param_grid_DT_Postpruning,
                           cv=skf,
                           scoring='accuracy', n_jobs=-1, verbose=2)

grid_search_DT_Postpruning.fit(X_train_class, y_train_class)

#Best Model
best_DT_model_Postpruning = grid_search_DT_Postpruning.best_estimator_
y_pred_DT_Postpruning = best_DT_model_Postpruning.predict(X_test_class)
print("Best parameters [Decision Tree Postpruning]:", grid_search_DT_Postpruning.best_params_)
print("Best cross-validated accuracy [Decision Tree Postpruning]:", grid_search_DT_Postpruning.best_score_)

#accuracy for train/test set
y_pred_train_DT_Postpruning = best_DT_model_Postpruning.predict(X_train_class)
y_test_acc_DT_Postpruning = accuracy_score(y_test_class, y_pred_DT_Postpruning)
y_train_acc_DT_Postpruning = accuracy_score(y_train_class, y_pred_train_DT_Postpruning)
print('accuracy on train data [decision tree with post-pruning]', y_train_acc_DT_Postpruning)
print('accuracy on test data [decision tree with post-pruning]', y_test_acc_DT_Postpruning)



#confusion matrix
# #confusion matrix
cm_DT_Postpruning = confusion_matrix(y_test_class, y_pred_DT_Postpruning)
display_DT_Postpruning = ConfusionMatrixDisplay(confusion_matrix=cm_DT_Postpruning)
display_DT_Postpruning.plot()
plt.title('Decision Tree Postpruning')
plt.show()

#print precision, recall, f1
precision_DT_Postpruning = precision_score(y_test_class, y_pred_DT_Postpruning)
print("Decision Tree Precision [Postpruning]:", precision_DT_Postpruning)
recall_DT_Postpruning = recall_score(y_test_class, y_pred_DT_Postpruning)
print("Decision Tree Recall [Postpruning]:", recall_DT_Postpruning)
specificity_DT_Postpruning = specificity_score(y_test_class, y_pred_DT_Postpruning)
print("Decision Tree Specificity [Postpruning]:", specificity_DT_Postpruning)
F1_DT_Postpruning = f1_score(y_test_class, y_pred_DT_Postpruning)
print("Decision Tree F1 [Postpruning]:", F1_DT_Postpruning)

#add row to table
clf_table.add_row(["Decision Tree[Postpruning]", precision_DT_Postpruning, recall_DT_Postpruning, specificity_DT_Postpruning, F1_DT_Postpruning])

# #ROC
RocCurveDisplay.from_estimator(best_DT_model_Postpruning, X_test_class, y_test_class)
y_prob_test_DT_Postpruning = best_DT_model_Postpruning.predict_proba(X_test_class)[:, 1]
plt.plot([0, 1], [0, 1], '--', color='grey', label='Chance Level (AUC=0.5)')
plt.title('Decision Tree [Post Pruning] ROC')
plt.legend(loc='lower right')
plt.show()


#Logistic Regression
LR_model = LogisticRegression(max_iter=1000, random_state=42)

param_grid_LR = {
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'C': [1, 10, 100, 1000]
}

LR_model.fit(X_train_class, y_train_class)
y_pred_LR = LR_model.predict(X_test_class)
LR_model.fit(X_train_class, y_train_class)

#GridSearch
grid_search_LR = GridSearchCV(LR_model,
                           param_grid=param_grid_LR,
                           cv=skf,
                           scoring='accuracy')

grid_search_LR.fit(X_train_class, y_train_class)

#Best Model
best_LR_model = grid_search_LR.best_estimator_
y_pred_LR = best_LR_model.predict(X_test_class)
print("Best parameters[Log Reg]:", grid_search_LR.best_params_)
print("Best cross-validated accuracy[Log Reg]:", grid_search_LR.best_score_)

#accuracy for train/test set
y_pred_train_LR = best_LR_model.predict(X_train_class)
y_test_acc_LR = accuracy_score(y_test_class, y_pred_LR)
y_train_acc_LR = accuracy_score(y_train_class, y_pred_train_LR)
print('accuracy on train data [logistic regression]', y_train_acc_LR)
print('accuracy on test data [logistic regression]', y_test_acc_LR)


cm_LR= confusion_matrix(y_test_class, y_pred_LR)
display_LR = ConfusionMatrixDisplay(confusion_matrix=cm_LR)
display_LR.plot()
plt.title('Logistic Regression')
plt.show()

#print precision, recall, f1
precision_LR = precision_score(y_test_class, y_pred_LR)
print("Logistic Regression Precision:", precision_LR)
recall_LR = recall_score(y_test_class, y_pred_LR)
print("Logistic Regression Recall:", recall_LR)
specificity_LR = specificity_score(y_test_class, y_pred_LR)
print("Logistic Regression Specificity:", specificity_LR)
F1_LR = f1_score(y_test_class, y_pred_LR)
print("Logistic Regression F1:", F1_LR)

#add row to table
clf_table.add_row(["Logistic Regression", precision_LR, recall_LR, specificity_LR, F1_LR])

#ROC
RocCurveDisplay.from_estimator(best_LR_model, X_test_class, y_test_class)
y_prob_test_LR = best_LR_model.predict_proba(X_test_class)[:, 1]
plt.plot([0, 1], [0, 1], '--', color='grey', label='Chance Level (AUC=0.5)')
plt.title('Logistic Regression ROC')
plt.legend(loc='lower right')
plt.show()



#SVM Classifier
print('Support Vector Machine')

SVM_model = SVC(probability=False, cache_size=1000, random_state=42)
param_grid_SVM = {
    'kernel': ['linear', 'poly', 'rbf'],
    'degree': [2, 3]
}

# GridSearch
grid_search_SVM = GridSearchCV(
    estimator=SVM_model,
    param_grid=param_grid_SVM,
    cv=skf,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)
grid_search_SVM.fit(X_train_class, y_train_class)

# Best Model
best_SVM_model = grid_search_SVM.best_estimator_
y_pred_SVM = best_SVM_model.predict(X_test_class)
print("Best parameters [SVM]:", grid_search_SVM.best_params_)
print("Best cross-validated accuracy [SVM]:", grid_search_SVM.best_score_)

#accuracy for train/test set
y_pred_train_SVM = best_SVM_model.predict(X_train_class)
y_test_acc_SVM = accuracy_score(y_test_class, y_pred_SVM)
y_train_acc_SVM = accuracy_score(y_train_class, y_pred_train_SVM)
print('accuracy on train data [support vector machine]', y_train_acc_SVM)
print('accuracy on test data [support vector machine]', y_test_acc_SVM)

# Confusion matrix
cm_SVM = confusion_matrix(y_test_class, y_pred_SVM)
display_SVM = ConfusionMatrixDisplay(confusion_matrix=cm_SVM)
display_SVM.plot()
plt.title('SVM Confusion Matrix')
plt.show()

# Print precision, recall, f1
precision_SVM = precision_score(y_test_class, y_pred_SVM)
print("SVM Precision:", precision_SVM)
recall_SVM = recall_score(y_test_class, y_pred_SVM)
print("SVM Recall:", recall_SVM)
specificity_SVM = specificity_score(y_test_class, y_pred_SVM)
print("SVM Specificity:", specificity_SVM)
F1_SVM = f1_score(y_test_class, y_pred_SVM)
print("SVM F1:", F1_SVM)

#add row to table
clf_table.add_row(["Support Vector Machine", precision_SVM, recall_SVM, specificity_SVM, F1_SVM])

# ROC
RocCurveDisplay.from_estimator(best_SVM_model, X_test_class, y_test_class)
# y_prob_test_SVM = best_SVM_model.predict_proba(X_test_class)[:, 1]
plt.plot([0, 1], [0, 1], '--', color='grey', label='Chance Level (AUC=0.5)')
plt.title('SVM ROC')
plt.legend(loc='lower right')
plt.show()

error = []

for k in range(1, 40):
    model_KN = KNeighborsClassifier(n_neighbors=k, weights='uniform')
    model_KN.fit(X_train_class, y_train_class)
    prediction_KN_k = model_KN.predict(X_test_class)
    e = np.mean(prediction_KN_k != y_test_class)
    error.append(e)

#plot error rate for each k
plt.plot(range(1, 40), error, marker='o', color = 'blue', linestyle='dashed', markerfacecolor='red', markeredgecolor='black')
plt.xlabel('K value')
plt.ylabel('Error Rate')
plt.title('Error Rate for Each K Value')
plt.show()

print('kNN Classifier')
kNN_model = KNeighborsClassifier(n_neighbors=19)
param_grid_kNN = {'leaf_size': (20,40,1), 'p': (1,2), 'weights': ('uniform', 'distance'), 'metric': ('minkowski', 'chebyshev')}

#GridSearch
grid_search_kNN = GridSearchCV(estimator=kNN_model, param_grid=param_grid_kNN, cv=skf, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search_kNN.fit(X_train_class, y_train_class)

# #Best Model
best_kNN_model = grid_search_kNN.best_estimator_
y_pred_kNN = best_kNN_model.predict(X_test_class)
print("Best parameters [kNN]:", grid_search_kNN.best_params_)
print("Best cross-validated accuracy [kNN]:", grid_search_kNN.best_score_)


#accuracy for train/test set
y_pred_train_kNN = best_kNN_model.predict(X_train_class)
y_test_acc_kNN = accuracy_score(y_test_class, y_pred_kNN)
y_train_acc_kNN = accuracy_score(y_train_class, y_pred_train_kNN)
print('accuracy on train data [k-nearest neighbors]', y_train_acc_kNN)
print('accuracy on test data [k-nearest neighbors]', y_test_acc_kNN)

#
# #confusion matrix
cm_kNN = confusion_matrix(y_test_class, y_pred_kNN)
display_kNN = ConfusionMatrixDisplay(confusion_matrix=cm_kNN)
display_kNN.plot()
plt.title('kNN Confusion Matrix')
plt.show()

#print precision, recall, f1
precision_kNN = precision_score(y_test_class, y_pred_kNN)
print("kNN Precision:", precision_kNN)
recall_kNN = recall_score(y_test_class, y_pred_kNN)
print("kNN Recall:", recall_kNN)
specificity_kNN = specificity_score(y_test_class, y_pred_kNN)
print("kNN Specificity:", specificity_kNN)
F1_kNN = f1_score(y_test_class, y_pred_kNN)
print("kNN F1:", F1_kNN)

#add row to table
clf_table.add_row(["kNN", precision_kNN, recall_kNN, specificity_kNN, F1_kNN])

#ROC
RocCurveDisplay.from_estimator(best_kNN_model, X_test_class, y_test_class)
# y_prob_test_NB = best_kNN_model.predict_proba(X_test_class)[:, 1]
plt.plot([0, 1], [0, 1], '--', color='grey', label='Chance Level (AUC=0.5)')
plt.title('kNN ROC')
plt.legend(loc='lower right')
plt.show()

#Neural Network

MLP_model = MLPClassifier(random_state=42)

param_grid_MLP = {
    'hidden_layer_sizes': [(62,), (62, 31), (62, 40), (62, 155), (62, 124), (62,40,20),(62,124,62)],
    'solver': ['sgd', 'adam'],
    'activation': ['tanh', 'relu'],
}

MLP_model.fit(X_train_class, y_train_class)
y_pred_LR = MLP_model.predict(X_test_class)
MLP_model.fit(X_train_class, y_train_class)

#GridSearch
grid_search_MLP = GridSearchCV(MLP_model,
                           param_grid=param_grid_MLP,
                           cv=skf,
                           scoring='accuracy', n_jobs=-1, verbose=2)

grid_search_MLP.fit(X_train_class, y_train_class)

#Best Model
best_MLP_model = grid_search_MLP.best_estimator_
y_pred_MLP = best_MLP_model.predict(X_test_class)
print("Best parameters[MLP]:", grid_search_MLP.best_params_)
print("Best cross-validated accuracy[MLP]:", grid_search_MLP.best_score_)

#accuracy for train/test set
y_pred_train_MLP = best_MLP_model.predict(X_train_class)
y_test_acc_MLP = accuracy_score(y_test_class, y_pred_MLP)
y_train_acc_MLP = accuracy_score(y_train_class, y_pred_train_MLP)
print('accuracy on train data [multilayer perceptron]', y_train_acc_MLP)
print('accuracy on test data [multilayer perceptron]', y_test_acc_MLP)


cm_MLP = confusion_matrix(y_test_class, y_pred_MLP)
display_MLP = ConfusionMatrixDisplay(confusion_matrix=cm_MLP)
display_MLP.plot()
plt.title('MLP Confusion Matrix')
plt.show()

#print precision, recall, f1
precision_MLP = precision_score(y_test_class, y_pred_MLP)
print("MLP Precision:", precision_MLP)
recall_MLP = recall_score(y_test_class, y_pred_MLP)
print("MLP Recall:", recall_MLP)
specificity_MLP = specificity_score(y_test_class, y_pred_MLP)
print("MLP Specificity:", specificity_MLP)
F1_MLP = f1_score(y_test_class, y_pred_MLP)
print("MLP F1:", F1_MLP)

#add row to table
clf_table.add_row(["MLP", precision_MLP, recall_MLP, specificity_MLP, F1_MLP])

#ROC
RocCurveDisplay.from_estimator(best_MLP_model, X_test_class, y_test_class)
y_prob_test_LR = best_MLP_model.predict_proba(X_test_class)[:, 1]
plt.plot([0, 1], [0, 1], '--', color='grey', label='Chance Level (AUC=0.5)')
plt.title('MLP ROC')
plt.legend(loc='lower right')
plt.show()


#random forest [BAGGING]
print('Random Forest [bagging]')
RF_model = RandomForestClassifier(random_state=42)
param_grid_RF = {
    'n_estimators': [25, 50, 100, 150],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [3, 6, 9],
    'max_leaf_nodes': [3, 6, 9],
}

#GridSearch
grid_search_RF = GridSearchCV(estimator=RF_model,
                           param_grid=param_grid_RF,
                           cv=skf,
                           scoring='accuracy', verbose=2,n_jobs=-1)

grid_search_RF.fit(X_train_class, y_train_class)

#Best Model
best_RF_model = grid_search_RF.best_estimator_
y_pred_RF = best_RF_model.predict(X_test_class)
print("Best parameters[Random Forest - Bagging]:", grid_search_RF.best_params_)
print("Best cross-validated accuracy[Random Forest - Bagging]:", grid_search_RF.best_score_)

#confusion matrix
# #confusion matrix
cm_RF = confusion_matrix(y_test_class, y_pred_RF)
display_RF = ConfusionMatrixDisplay(confusion_matrix=cm_RF)
display_RF.plot()
plt.title('Random Forest [Bagging] Confusion Matrix')
plt.show()

#print precision, recall, f1
precision_RF = precision_score(y_test_class, y_pred_RF)
print("Random Forest Precision:", precision_RF)
recall_RF = recall_score(y_test_class, y_pred_RF)
print("Random Forest Recall:", recall_RF)
specificity_RF = specificity_score(y_test_class, y_pred_RF)
print("Random Forest Specificity:", specificity_RF)
F1_RF = f1_score(y_test_class, y_pred_RF)
print("Random Forest F1:", F1_RF)

#ROC
RocCurveDisplay.from_estimator(best_RF_model, X_test_class, y_test_class)
plt.plot([0, 1], [0, 1], '--', color='grey', label='Chance Level (AUC=0.5)')
plt.title('Random Forest [Bagging] ROC')
plt.legend(loc='lower right')
plt.show()

#random forest [BOOSTING]
print('Random Forest [BOOSTING]')
RF_model_boost = XGBRFClassifier(random_state=42)
param_grid_RF_boost = {
    'n_estimators': [25, 50, 100, 150],
    'max_depth': [3, 6, 9, 12],
}

#GridSearch
grid_search_RF_boost = GridSearchCV(estimator=RF_model_boost,
                           param_grid=param_grid_RF_boost,
                           cv=skf,
                           scoring='accuracy', verbose=2,n_jobs=-1)

grid_search_RF_boost.fit(X_train_class, y_train_class)

#Best Model
best_RF_model_boost = grid_search_RF_boost.best_estimator_
y_pred_RF_boost = best_RF_model_boost.predict(X_test_class)
print("Best parameters[Random Forest - BOOSTING]:", grid_search_RF_boost.best_params_)
print("Best cross-validated accuracy[Random Forest - BOOSTING]:", grid_search_RF_boost.best_score_)

#confusion matrix
# #confusion matrix
cm_RF_boost = confusion_matrix(y_test_class, y_pred_RF)
display_RF_boost = ConfusionMatrixDisplay(confusion_matrix=cm_RF_boost)
display_RF_boost.plot()
plt.title('Random Forest [Boosting] Confusion Matrix')
plt.show()

#print precision, recall, f1
precision_RF_boost = precision_score(y_test_class, y_pred_RF_boost)
print("Random Forest Precision:", precision_RF_boost)
recall_RF_boost = recall_score(y_test_class, y_pred_RF_boost)
print("Random Forest Recall:", recall_RF_boost)
specificity_RF_boost = specificity_score(y_test_class, y_pred_RF_boost)
print("Random Forest Specificity:", specificity_RF_boost)
F1_RF_boost = f1_score(y_test_class, y_pred_RF_boost)
print("Random Forest F1:", F1_RF_boost)

#ROC
RocCurveDisplay.from_estimator(best_RF_model_boost, X_test_class, y_test_class)
plt.plot([0, 1], [0, 1], '--', color='grey', label='Chance Level (AUC=0.5)')
plt.title('Random Forest [BOOSTING] ROC')
plt.legend(loc='lower right')
plt.show()

print('Final Classification Summary Table')
print(clf_table)

#scale df for clustering
print(df_cluster.head())

#scale the data
numerical_features_cluster = ['bath', 'acre_lot','price']
standard_scaler_cluster = StandardScaler()
df_cluster[numerical_features_cluster] = standard_scaler_cluster.fit_transform(df_cluster[numerical_features_cluster])

#nomalize the label encoded data
label_encoded_features_cluster = ['city_encoded', 'brokered_by_encoded']
label_encoded_scaler_cluster = MinMaxScaler()
df_cluster[label_encoded_features_cluster] = label_encoded_scaler_cluster.fit_transform(df_cluster[label_encoded_features_cluster])

print('dataframe after scaling:')
print(df_cluster.head())
print("K MEANS: Silhoutte Score Method")

#find best silhoutte score
silhouette_scores = []
num_clusters = [i for i in range(2,10)]

for i in num_clusters:
    k_means_ss = KMeans(n_clusters=i, random_state=42)
    k_means_ss.fit(df_cluster)
    k_means_ss.fit_predict(df_cluster)
    s_score = silhouette_score(df_cluster, k_means_ss.labels_, metric = 'euclidean', random_state=42)
    print('num clusters', i, ', silhouette score:', s_score)
    silhouette_scores.append(s_score)

#plot the scores
plt.plot(num_clusters, silhouette_scores)
plt.title('Silhouette Scores for K')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.show()

#get number of points per cluster
k_means_ss_best = KMeans(n_clusters=2, random_state=42)
k_means_ss_best.fit(df_cluster)
k_means_ss_labels = k_means_ss_best.labels_

unique_values_ss, counts_ss = np.unique(k_means_ss_labels, return_counts=True)
print('unique values and counts [ss]')
print(unique_values_ss, counts_ss)

print("K MEANS: Elbow Method")
wcss_vals = []

for i in range(2,30):
    k_means_wcss = KMeans(n_clusters=i, random_state=42)
    k_means_wcss.fit(df_cluster)
    k_means_wcss.fit_predict(df_cluster)
    wcss = k_means_wcss.inertia_
    print('num clusters', i, ', wccs:', wcss)
    wcss_vals.append(wcss)

#plot scores
plt.plot(range(2,30), wcss_vals)
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#get number of points per cluster
k_means_wccs_best = KMeans(n_clusters=15, random_state=42)
k_means_wccs_best.fit(df_cluster)
k_means_wccs_labels = k_means_wccs_best.labels_

unique_values, counts = np.unique(k_means_wccs_labels, return_counts=True)
print('unique values and counts [wccs]')
print(unique_values, counts)


#DBSCAN clustering
print("DBSCAN Clustering")
DB_neighbors = NearestNeighbors(n_neighbors=120)
neighbors_fitted = DB_neighbors.fit(df_cluster)
distances, indices = neighbors_fitted.kneighbors(df_cluster)

print('distances', distances)
print(indices)

plt.figure(figsize=(20,20))
dist = np.sort(distances[:,119], axis=0)
plt.plot(dist)
plt.show()

db_default = DBSCAN(eps = 5, min_samples = 120).fit(df_cluster)
db_labels = db_default.labels_
num_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
num_noise = list(db_labels).count(-1)

print('Number of clusters:', num_clusters)
print('Number of noise points:', num_noise)

#Association Rule Mining
sample_df_arm.drop(columns=['prev_sold_date'], inplace=True)

# For bed
Quantile_25_bed = np.quantile(sample_df_arm['bed'], 0.25)
Quantile_75_bed = np.quantile(sample_df_arm['bed'], 0.75)
IQR_bed = Quantile_75_bed - Quantile_25_bed

def categorize_bed(x):
    if x < Quantile_25_bed:
        return 'Low Count'
    elif x > Quantile_75_bed:
        return 'High Count'
    else:
        return 'Medium Count'

sample_df_arm['bed category'] = sample_df_arm['bed'].apply(categorize_bed)

# For bath
Quantile_25_bath = np.quantile(sample_df_arm['bath'], 0.25)
Quantile_75_bath = np.quantile(sample_df_arm['bath'], 0.75)
IQR_bath = Quantile_75_bath - Quantile_25_bath

def categorize_bath(x):
    if x < Quantile_25_bath:
        return 'Low Count'
    elif x > Quantile_75_bath:
        return 'High Count'
    else:
        return 'Medium Count'

sample_df_arm['bath category'] = sample_df_arm['bath'].apply(categorize_bath)

# For acre_lot
Quantile_25_acre_lot = np.quantile(sample_df_arm['acre_lot'], 0.25)
Quantile_75_acre_lot = np.quantile(sample_df_arm['acre_lot'], 0.75)
IQR_acre_lot = Quantile_75_acre_lot - Quantile_25_acre_lot

def categorize_acre_lot(x):
    if x < Quantile_25_acre_lot:
        return 'Low Count'
    elif x > Quantile_75_acre_lot:
        return 'High Count'
    else:
        return 'Medium Count'

sample_df_arm['acre_lot category'] = sample_df_arm['acre_lot'].apply(categorize_acre_lot)

# For house_size
Quantile_25_house_size = np.quantile(sample_df_arm['house_size'], 0.25)
Quantile_75_house_size = np.quantile(sample_df_arm['house_size'], 0.75)
IQR_house_size = Quantile_75_house_size - Quantile_25_house_size

def categorize_house_size(x):
    if x < Quantile_25_house_size:
        return 'Low Count'
    elif x > Quantile_75_house_size:
        return 'High Count'
    else:
        return 'Medium Count'

sample_df_arm['house_size category'] = sample_df_arm['house_size'].apply(categorize_house_size)

# For house_size
Quantile_25_price = np.quantile(sample_df_arm['price'], 0.25)
Quantile_75_price = np.quantile(sample_df_arm['price'], 0.75)
IQR_price = Quantile_75_price - Quantile_25_price

def categorize_price(x):
    if x < Quantile_25_price:
        return 'Low Count'
    elif x > Quantile_75_price:
        return 'High Count'
    else:
        return 'Medium Count'

sample_df_arm['price category'] = sample_df_arm['price'].apply(categorize_price)

bed_map = {'Low Count': 0, 'Medium Count': 1, 'High Count': 2}
le_bed= LabelEncoder()
sample_df_arm['bed category'] = le_bed.fit_transform(sample_df_arm['bed category'].map(bed_map))

bath_map = {'Low Count': 0, 'Medium Count': 1, 'High Count': 2}
le_bath = LabelEncoder()
sample_df_arm['bath category'] = le_bath.fit_transform(sample_df_arm['bath category'].map(bath_map))

lot_map = {'Low Count': 0, 'Medium Count': 1, 'High Count': 2}
le_lot = LabelEncoder()
sample_df_arm['acre_lot category'] = le_bath.fit_transform(sample_df_arm['acre_lot category'].map(lot_map))

size_map = {'Low Count': 0, 'Medium Count': 1, 'High Count': 2}
le_size = LabelEncoder()
sample_df_arm['house_size category'] = le_bath.fit_transform(sample_df_arm['house_size category'].map(size_map))

price_map = {'Low Count': 0, 'Medium Count': 1, 'High Count': 2}
le_price = LabelEncoder()
sample_df_arm['price category'] = le_bath.fit_transform(sample_df_arm['price category'].map(price_map))


sample_df_arm.drop(columns=['brokered_by', 'city_encoded', 'city', 'state', 'zip_code', 'zip_code_encoded', 'brokered_by_encoded', 'price', 'bed', 'bath',
                            'acre_lot', 'house_size'], inplace=True)

sample_df_arm = pd.get_dummies(sample_df_arm, columns = ['bed category', 'bath category', 'acre_lot category', 'house_size category', 'price category', 'year', 'month', 'status'], dtype=bool, drop_first=False)


print(sample_df_arm.head())


print('Dataframe after discretization')
print(sample_df_arm.head())

#Apriori
frequent_item_sets = apriori(sample_df_arm, min_support=0.1, use_colnames=True, verbose=1)
print(frequent_item_sets)
frequent_item_sets_rules = association_rules(frequent_item_sets, metric='confidence', min_threshold=0.6)


frequent_item_sets_rules1 = frequent_item_sets_rules.sort_values(['lift'], ascending =[False])
print(frequent_item_sets_rules1.head(10).to_string())

frequent_item_sets_rules2 = frequent_item_sets_rules.sort_values(['confidence'], ascending =[False])
print(frequent_item_sets_rules2.head(10).to_string())