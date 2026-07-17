import feature_analysis
import feature_engineering
from sklearn.preprocessing import RobustScaler
import pandas as pd
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence,permutation_importance


def feature_performance():
    feature_data()
    feature_categories()
    #feature_importance()

# Data splitting for scaling
def feature_data():
    X_train, X_test, y_train, y_test,_ = feature_analysis.access_df()

    scaler = RobustScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_train = pd.DataFrame(X_train, columns=scaler.get_feature_names_out())

    scaler_test = RobustScaler().fit(X_test)
    X_test = scaler_test.transform(X_test)
    model = RandomForestClassifier()
    model.fit(X_train,y_train)
    return X_train, X_test, y_train, y_test,model

def partial_depend(my_list):
    partial_dep_values = []
    X_train, _, y_train, _,model = feature_data()

    print(f" my list is {my_list}")
    #print(f"my df list is {X_train[(my_list)[1]]} ")
    dep_features = list(combinations(my_list,2))
    for features in dep_features:
        dep = partial_dependence(model, X=X_train, features=features)
        partial_dep_values.append([features,dep])
    print(f" partial_dep_values   {partial_dep_values}")


    # for f1,f2 in combinations(my_list,2):
    #     print(f" the featuures are {X_train[f1],X_train[f2]}")
    #     #dep = partial_dependence(model,  X=X_train,features=[])
    #     #partial_dep_values.append(dep)
    # print(f"partial_dep values are {partial_dep_values}")
    return partial_dep_values


# Feature Performance list collection
def feature_categories():
   phik_features = feature_engineering.corr_feature_analyse()
   phik_list_features = list(phik_features.index)

   mi_features = feature_analysis.mutual_info()
   mi_features_list = mi_features.loc[(lambda x: x > 0)]
   mi_features_list = list(mi_features_list.index)

   non_mi_features = mi_features.loc[(lambda x: x <= 0)]
   print(f" non  mi list is {non_mi_features}")
   non_mi_features_list = list(non_mi_features.index)
   partial_depend(non_mi_features_list)




   imp_features = feature_analysis.feature_importance()
   imp_features_list = imp_features.loc[(lambda x: x >= 0)]
   imp_features_list = list(imp_features_list.index)
   #print(imp_features_list)
