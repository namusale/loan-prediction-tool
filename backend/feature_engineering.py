'''
This module computes correlation and significance matrices to understand features that are significant to the outcome.
There is also encoding for categorical data
'''
import matplotlib.pyplot as plt
import pandas as pd
import phik
import utils
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = utils.source_data()


def feature_eng():
    feature_matrices()
    corr_feature_analyse()
    encoding()


# Correlation matrix for feature analysis
def feature_matrices():
    # Correlation Matrix for features
    corr_matrix = df.phik_matrix()
    sns.heatmap(corr_matrix,
                cmap='Reds',
                annot=True,
                linewidth=0.5,
                fmt=".1f")
    plt.xticks(rotation=40)
    plt.savefig("reports/images/corr_matrix.png")

    # Significance Matrix for features
    sig_mat = df.significance_matrix()
    sns.heatmap(sig_mat,
                cmap="Reds",
                annot=True,
                linewidths=0.5,
                fmt=".1f")
    plt.xticks(rotation=40)
    plt.tight_layout()
    plt.savefig("reports/images/sig_matrix.png")

    return corr_matrix, sig_mat


# Correlation and Significance Matrices comparison
def corr_feature_analyse():
    corr_matrix, sig_mat = feature_matrices()
    # Creating mini dataframe to march features in two matrices with target
    feature_phik_data = pd.DataFrame({
        'corr_mat': corr_matrix['loan_status'],
        'sig_mat': sig_mat['loan_status']})

    phik_features = feature_phik_data[(feature_phik_data['corr_mat'] > 0.05) & (feature_phik_data['sig_mat'] > 1)] \
        .sort_values(['corr_mat', 'sig_mat'], ascending=False)
    with open("reports/data.txt", "a") as f:
        f.write(" \n")
        f.write("Features based on Correlations and Significant Scores\n")
        f.write(str(phik_features))
        f.close()
    return phik_features


# Categorical Data encoding
def encoding():
    # function used for encoding categorical data
    df_cat = df.select_dtypes(include=['object']).drop(['loan_status'], axis=1)
    print(f" The columns of df are {df.columns}")

    le = LabelEncoder()
    ohe = OneHotEncoder(handle_unknown='ignore')

    target = le.fit_transform(df['loan_status'])
    target = pd.DataFrame(target, columns=['num_loan_status'])
    ohe_data = ohe.fit_transform(df_cat).toarray()
    ohe_names = ohe.get_feature_names_out()
    # creating dataset for categorical data
    df_ohe = pd.DataFrame(ohe_data, columns=ohe_names)

    # concatenating numerical and categorical data to form new encoded dataset
    df_enc = pd.concat([df, df_ohe, target], axis=1).drop(['education', 'self_employed', 'loan_status'], axis=1)
    return df_enc
