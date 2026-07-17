'''
This file analyses mutual importance and feature importance to obtain features that are important to the loan outcome
'''
import pandas as pd
from pandas import Series
import feature_engineering
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def access_df():
    df = feature_engineering.encoding()
    X = df.drop(['num_loan_status'], axis=1)
    y = df['num_loan_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, model


def feature_analyse():
    mutual_info()
    # feature_partial_dependence()
    feature_importance()


# Mutual ifomation of features is computed to grade feature imactfulness.
def mutual_info():
    X_m, _, y_m, _, _ = access_df()
    mi_features = X_m.columns
    mi = mutual_info_classif(X_m, y_m)
    mi_feature_list = pd.Series(mi, index=mi_features).sort_values(ascending=False)
    with open("reports/data.txt", "a") as f:
        f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write("Features List Based of Mutual Information\n")
        f.write(str(mi_feature_list))
        f.close()
    return mi_feature_list


# partial dependence
def feature_partial_dependence():
    X_train, _, _, _, model = access_df()
    #  Partial Dependency for Income and Education Graduate
    PartialDependenceDisplay.from_estimator(model, X_train, [('income_annum', 'education_ Graduate')])
    plt.title("Partial Dependence for Income Annum Vs Education:Graduate")
    plt.savefig("reports/images/income_edu_grad_pd.png")

    PartialDependenceDisplay.from_estimator(model, X_train, [('self_employed_ Yes', 'education_ Graduate')])
    plt.title("Partial Dependence for Self Employed:Yes Vs Education:Graduate")
    plt.savefig("reports/images/self_emp_yes_edu_grad_pd.png")

    PartialDependenceDisplay.from_estimator(model, X_train, [('income_annum', 'no_of_dependents')])
    plt.title("Partial Dependence for Income Annum Vs Number of Dependents")
    plt.savefig("reports/images/income_Vs_no_of_dependents.png")

    PartialDependenceDisplay.from_estimator(model, X_train, [('income_annum', 'residential_assets_value')])
    plt.title("Partial Dependence for Income Annum Vs Residential assets Value")
    plt.savefig("reports/images/income_Vs_res_assets.png")

    PartialDependenceDisplay.from_estimator(model, X_train, [('income_annum', 'loan_amount')])
    plt.title("Partial Dependence for Income Annum Vs Loan Amount")
    plt.savefig("reports/images/income_Vs_loan_amount.png")


# Feature importance to model being analysed in this function and list of feature importance obtained
def feature_importance():
    _, X_test, _, y_test, model = access_df()
    imp = permutation_importance(model, X_test, y_test)
    importance: Series = pd.Series(imp['importances_mean'], index=X_test.columns).sort_values(ascending=False)
    with open("reports/data.txt", "a") as f:
        f.write("\n")
        f.write("\n")
        f.write(" Feature Importance\n")
        f.write(str(importance))
        f.close()
    return importance
