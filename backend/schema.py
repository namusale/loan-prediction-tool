   # Pydantic Models
# !/usr/bin/env python

# Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

# Dataset Early Analysis
df = pd.read_csv("loan_approval_dataset.csv", delimiter=",")

# df Peek
print(df.head())

df.info()
# No missing values noted


df.isnull().sum().sum()
# Confirmed no missing values


# Clean up Feature names as they have spaces in names
df_columns = []
for feature in list(df.columns):
    df_columns.append(feature.strip())

df.columns = df_columns

#  Checking for Duplicated
df.duplicated().sum()

df.describe()

df.hist()
plt.show()

# Observation:
# Graphically, 3 numerical features are right skewed. We measure the skewness of these features


# Observing Skewness of numerical Data in desceding Order
pd.Series(df.skew(numeric_only=True)).sort_values(ascending=False)

# Observation
# 2 Features (i.e. residential_assets_value and commercial_assets_value) are highly skewed.


# Checking for Outliers
sns.boxplot(df)
plt.xticks(rotation=90)
plt.show()

# Outlier observation: The assets features have outliers and these could be for the highly rich few individuals. These are contributing to the skewness of the data with residential_assets_value being the most skewed. Proposal is to log them when using distance based models.


# Separating categorical and numerical Data
df_cat = df.select_dtypes(include=['object']).drop(['loan_status'], axis=1)
df_num = df.select_dtypes(include=['int64', 'float64'])

# Checking the class distribution for categorical Data
df_cat.groupby(['education']).size()
df_cat.groupby(['self_employed']).size()

# Class distribution in terms of percentages for education feature (categorical Data)
(so.Plot(df_cat, x='education', color='education')
 .add(so.Bar(), so.Hist(stat='percent'))
 .label(x="Education", y='Percentage Distribution', title="Education Feature Distribution"))

# Class distribution in terms of percentages for Self Employed feature (categorical Data)
(so.Plot(df_cat, x="self_employed", color="self_employed")
 .add(so.Bar(), so.Hist(stat='percent'))
 .label(x="Self or Not Self employed", y="Distribution in Percentage",
        title="Self Employed Feature Distribution"))

# Checking for class distribution for the target
(so.Plot(df.loan_status)
 .add(so.Bar(), so.Hist(stat='percent'))
 .label(x='Loan Status', y="Loan Approval Status in %", title="Loan Approval Distribution"))

# Observation:
#  Categorical data have binary classes with distribution of close to 50 percent. However the target is to some extent unbalances with a binary distribution of 63% to 37%. Since the imbalance is not huge, no SMOTE will be used.

# Correlation and siginificance of Features
# To understand whether some featureare correlated, weuse a phik matrix and to check if the correlation is significant we use significance matrix from phik on the data.


corr_matrix = df.phik_matrix()

sns.heatmap(corr_matrix,
            cmap='Reds',
            annot=True,
            linewidth=0.5,
            fmt=".1f")

plt.show()


sig_mat = df.significance_matrix()
sns.heatmap(sig_mat,
            cmap="Reds",
            annot=True,
            linewidths=0.5,
            fmt=".1f")
plt.show()

feature_phik_data = pd.DataFrame({
    'corr_mat': corr_matrix['loan_status'],
    'sig_mat': sig_mat['loan_status']})

phik_features = feature_phik_data[(feature_phik_data['corr_mat'] > 0.05) & (feature_phik_data['sig_mat'] > 1)] \
    .sort_values(['corr_mat', 'sig_mat'], ascending=False)


list(phik_features.index)

# Features for corr are cibil_score,loan_term,loan_amount and luxury_assets_value


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


le = LabelEncoder()
ohe = OneHotEncoder(handle_unknown='ignore')

# In[27]:


target = le.fit_transform(df['loan_status'])
target = pd.DataFrame(target, columns=['num_loan_status'])
ohe_data = ohe.fit_transform(df_cat).toarray()
ohe_names = ohe.get_feature_names_out()
print(ohe_names)


df_ohe = pd.DataFrame(ohe_data, columns=ohe_names)


df_new = pd.concat([df, df_ohe, target], axis=1).drop(['education', 'self_employed', 'loan_status'], axis=1)


df_new.head()

df_new.describe()

from sklearn.feature_selection import mutual_info_classif


# Data Splitting
X_m = df_new.drop(['num_loan_status'], axis=1)
y_m = df_new['num_loan_status']
mi_features = X_m.columns

X_m.tail()

mi = mutual_info_classif(X_m, y_m)
print(mi)

pd.Series(mi, index=mi_features).sort_values(ascending=False)

# Features from MI deemed to signicantly have impact on target value
# cibil_score                 0.504827
# loan_term                   0.024142
# self_employed_ Yes          0.009416
# luxury_assets_value         0.007694
# income_annum                0.004735
# self_employed_ No           0.003534
# education_ Graduate         0.002954
# education_ Not Graduate     0.002666
# residential_assets_value    0.001192

from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


X = df_new.drop(['num_loan_status'], axis=1)
y = df_new['num_loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = RandomForestClassifier()
model.fit(X_train, y_train)


PartialDependenceDisplay.from_estimator(model, X_train, [('income_annum', 'education_ Graduate')])
plt.show()


PartialDependenceDisplay.from_estimator(model, X_train, [('self_employed_ Yes', 'education_ Graduate')])
plt.show()


PartialDependenceDisplay.from_estimator(model, X_train, [('income_annum', 'no_of_dependents')])


PartialDependenceDisplay.from_estimator(model, X_train, [('income_annum', 'residential_assets_value')])


PartialDependenceDisplay.from_estimator(model, X_train, [('income_annum', 'loan_amount')])


from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split




model = RandomForestClassifier().fit(X_train, y_train)


X = df_new.drop(['num_loan_status'], axis=1)
y = df_new.num_loan_status

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

imp = permutation_importance(model, X_test, y_test)


pd.Series(imp['importances_mean'], index=X_test.columns).sort_values(ascending=False)



df_new.residential_assets_value.hist()


df_new.residential_assets_value.describe()

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_scale = pd.DataFrame(X_train_scaled, columns=scaler.get_feature_names_out())

# In[74]:


X_scale.head()

from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(X_train_scaled, y_train)


scaler_test = RobustScaler().fit(X_test)
transform_test = scaler_test.transform(X_test)


pred = clf.predict(transform_test)


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

# In[80]:


accuracy_score(y_test, pred), precision_score(y_test, pred), recall_score(y_test, pred), f1_score(y_test, pred)

# precision  is TP/(TP + FP) recall =TP/(TP+FN)

# In[81]:


confusion_matrix(y_test, pred)

# In[82]:


from sklearn.neighbors import KNeighborsClassifier


k_clf = KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train)

k_pred = k_clf.predict(transform_test)


accuracy_score(y_test, k_pred), precision_score(y_test, k_pred), \
    recall_score(y_test, k_pred), f1_score(y_test, k_pred)


from sklearn.neural_network import MLPClassifier

n_clf = MLPClassifier(random_state=42, max_iter=2000).fit(X_train_scaled, y_train)


n_pred = n_clf.predict(transform_test)


accuracy_score(y_test, n_pred), precision_score(y_test, n_pred), \
    recall_score(y_test, n_pred), f1_score(y_test, n_pred)


df_new.columns

df_trim = df_new[['income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'luxury_assets_value',
                  'no_of_dependents', 'num_loan_status', 'education_ Graduate', 'education_ Not Graduate', \
                  'self_employed_ No', 'self_employed_ Yes']]


X_trim = df_trim.drop(['num_loan_status'], axis=1)
y_trim = df_trim.num_loan_status


X_train, X_test, y_train, y_test = train_test_split(X_trim, y_trim, test_size=0.3, random_state=42)

scaler = RobustScaler().fit(X_train)
X_train_trim = scaler.transform(X_train)

scaler_test = RobustScaler().fit(X_test)
X_test_trim = scaler_test.transform(X_test)


clf.fit(X_train_trim, y_train)


pred = clf.predict(X_test_trim)

accuracy_score(y_test, pred), precision_score(y_test, pred), recall_score(y_test, pred), f1_score(y_test, pred)

sns.heatmap(confusion_matrix(y_test, pred, normalize='true'),
            annot=True,
            linewidth=0.5,
            cmap='Reds')
plt.title("Classification Heatmap")
plt.show()

mi_feature_list = ["cibil_score", "loan_term", "education_ Not Graduate", "self_employed_ No", \
                   "luxury_assets_value", "bank_asset_value", "commercial_assets_value", "no_of_dependents"]



mi_part_dep = ["cibil_score", "loan_term", "education_ Not Graduate", "self_employed_ No", \
               "luxury_assets_value", "bank_asset_value", "commercial_assets_value", \
               "no_of_dependents", 'income_annum', 'loan_amount', 'self_employed_ Yes', 'education_ Graduate']


perm_imp = ["cibil_score", "loan_term", "loan_amount", "income_annum", "residential_assets_value", \
            "bank_asset_value", "no_of_dependents", "commercial_assets_value"]

features_all = df_new.drop(["num_loan_status"], axis=1).columns
target = df_new.num_loan_status

features_all

feature_group_list = [mi_feature_list, mi_part_dep, perm_imp, features_all]


clf = tree.DecisionTreeClassifier()
for features in feature_group_list:
    X, y = df_new[features], target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler_train = RobustScaler().fit(X_train)
    X_train = scaler_train.transform(X_train)

    scaler_test = RobustScaler().fit(X_test)
    X_test = scaler_test.transform(X_test)

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(accuracy_score(y_test, pred), precision_score(y_test, pred), recall_score(y_test, pred),
          f1_score(y_test, pred))

