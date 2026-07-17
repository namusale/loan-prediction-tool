'''
The module acceses the dataset and does the following:
    1) cleans spaces in column names
    2) checks for missing data, duplicated entries and summarises data
    3) Checks for data skewness and class distribution for categorical data
'''
# Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so


# Access dataset,cleans columns names and returns dataset
def source_data() -> object:
    # Access to pandas dataset
    data = pd.read_csv("loan_approval_dataset.csv", sep=",")
    df = clean_columns(data)
    return df

def utilities():
    basic_info()
    features_distribution()
    categorical()

# function summarises data and documents in a file(data.txt)
def basic_info(df_head=None):
    df = source_data()
    with open("reports/data.txt", "w") as f:
        pd.set_option('display.max_rows', 100)
        f.write(str(df.head()))
        f.write("  \n")
        f.write("  \n")
        f.write(f"The Dataset Information  {df.info()}\n")
        f.write(f"Number of missing values {df.isnull().sum().sum()}\n")
        f.write(f"Number of duplicated items {df.duplicated().sum()}\n")
        f.write(" \n")
        f.write(f" Dataset summary {df.describe()}\n")
        f.close()

def clean_columns(data: object) -> object:
    # Clean up Feature names as they have spaces in names
    df = data
    df_columns = []
    for feature in list(df.columns):
        df_columns.append(feature.strip())
    df.columns = df_columns
    return df

#Plots data and saves the plots for visual analysis
def features_distribution():
    df = source_data()
    df.hist()
    plt.tight_layout()
    plt.savefig("reports/images/features.png")
    plt.close()

    # Observing Skewness of numerical Data in descending Order
    skew = pd.Series(df.skew(numeric_only=True)).sort_values(ascending=False)
    with open("reports/data.txt","a") as f:
        f.write("  \n")
        f.write(f" Feature Skewness \n")
        f.write(f" {skew}\n")
        f.close()

    # Checking for Outliers
    sns.boxplot(df)
    plt.title(" Feature Skewness")
    plt.xlabel('Features')
    plt.xticks(rotation=40)
    plt.savefig("reports/images/skewness.png")
    plt.close()

# Checks for class distributuion for categorical data and return categorical data frame
def categorical():
    # Separating categorical and numerical Data
    df=source_data()
    df_cat = df.select_dtypes(include=['object']).drop(['loan_status'], axis=1)
    df_num = df.select_dtypes(include=['int64', 'float64'])

    # Class distribution in terms of percentages for education feature (categorical Data)
    p=(so.Plot(df_cat, x='education', color='education')
     .add(so.Bar(), so.Hist(stat='percent'))
     .label(x="Education", y='Percentage Distribution', title="Education Feature Distribution"))
    p.save("reports/images/education.png", dpi=300)

    # Class distribution in terms of percentages for Self Employed feature (categorical Data)
    p = (so.Plot(df_cat, x="self_employed", color="self_employed")
     .add(so.Bar(), so.Hist(stat='percent'))
     .label(x="Self or Not Self employed", y="Distribution in Percentage",
            title="Self Employed Feature Distribution"))
    p.save("reports/images/self_employment.png", dpi=300)

    # Checking for class distribution for the target
    p = (so.Plot(df.loan_status)
     .add(so.Bar(), so.Hist(stat='percent'))
     .label(x='Loan Status', y="Loan Approval Status in %", title="Loan Approval Distribution"))
    p.save("reports/images/loan_status.png", dpi=300)

    return df_cat



