'''
This module uses different categories of features in model training. The best performing feature group
is returned to serve as input features for the model. Thw scaler and model used are saved for use with
user tool evaluation
'''

from feature_analysis import mutual_info, feature_importance, access_df
from feature_engineering import corr_feature_analyse
from sklearn.preprocessing import RobustScaler
from sklearn import tree
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import numpy as np
import joblib


def aggregate():
    model_dumping()


# Compiling feature groups for performance comparison
def feature_collect():
    X_train, X_test, y_train, y_test, _ = access_df()
    features_mi = mutual_info()
    features_imp = feature_importance()
    features_phik = corr_feature_analyse()

    all_features = X_train.columns
    mi_features = list((features_mi[features_mi > 0]).index)
    imp_features = list((features_imp[features_imp > 0]).index)
    phik_features = list(features_phik.index)
    phik_features.remove('loan_status')
    features = [all_features, mi_features, imp_features, phik_features]
    for feature in features:
        print(f'we are printing features {feature}')
    return features


# Model training using different categories of features to obtain the best perforning group
def train_model():
    feature_list = feature_collect()
    model_list = []
    f1_values = []
    X_train, X_test, y_train, y_test, _ = access_df()
    for features in feature_list:
        scaler = RobustScaler().fit(X_train[features])
        X_train_scaled = scaler.transform(X_train[features])
        clf = tree.DecisionTreeClassifier()
        clf.fit(X_train_scaled, y_train)
        # model_list.append(joblib.dump(clf, 'model.joblib'))

        scaler_test = RobustScaler().fit(X_test[features])
        print(f'the feathurev are: {X_test[features].shape} ')
        print(f'the feature tandforn feathurev are: {X_test[features].iloc[[0]]} and status is {y_test.iloc[0]}')
        transform_test = scaler_test.transform(X_test[features])
        pred = clf.predict(transform_test)
        performance = [accuracy_score(y_test, pred), precision_score(y_test, pred), recall_score(y_test, pred),
                       f1_score(y_test, pred)]
        f1_values.append(f1_score(y_test, pred))
        print(performance, np.argmax(f1_values))

    return np.argmax(f1_values), feature_list[np.argmax(f1_values)]

# Model Training and dump for front end input evaluation
def model_dumping():
    _, features = train_model()
    X_train, X_test, y_train, y_test, _ = access_df()
    X_train = X_train[features]
    X_test = X_test[features]
    scaler = RobustScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train_scaled, y_train)
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(clf, 'model.joblib')
    return X_test, X_test_scaled, y_test
