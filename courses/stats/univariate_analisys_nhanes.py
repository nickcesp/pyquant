import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def analyze(plot=1):

    da = pd.read_csv("resources/nhanes_2015_2016.csv")
    # Count Values
    print(da.DMDEDUC2.value_counts())
    # Count nulls
    print(pd.isnull(da.DMDEDUC2).sum())

    # Label variables
    da["DMDEDUC2x"] = da.DMDEDUC2.replace({1: "<9", 2: "9-11", 3: "HS/GED", 4: "Some college/AA", 5: "College",
                                           7: "Refused", 9: "Don't know"})
    da["RIAGENDRx"] = da.RIAGENDR.replace({1: "Male", 2: "Female"})

    # Summary
    print(da.BMXWT.dropna().describe())

    # Graphical summary
    if plot == 1:
        sns.distplot(da.BMXWT.dropna())
    elif plot == 2:
        # Boxplot
        bp = sns.boxplot(data=da.loc[:, ["BPXSY1", "BPXSY2", "BPXDI1", "BPXDI2"]])
        _ = bp.set_ylabel("Blood pressure in mm/Hg")
    elif plot in (3, 4, 5):
        # Stratify (Cut creates a column with the label)
        da["agegrp"] = pd.cut(da.RIDAGEYR, [18, 30, 40, 50, 60, 70, 80])  # Create age strata based on these cut points
        plt.figure(figsize=(12, 5))  # Make the figure wider than default (12cm wide by 5cm tall)
        if plot == 3:  # Stratify by age only
            sns.boxplot(x="agegrp", y="BPXSY1", data=da)  # Make boxplot of BPXSY1 stratified by age group
        elif plot == 4:   # stratify by age and gender and display together
            sns.boxplot(x="agegrp", y="BPXSY1", hue="RIAGENDRx", data=da)
        else:  # Switch display, compare agegroups within ea. gender
            sns.boxplot(x="RIAGENDRx", y="BPXSY1", hue="agegrp", data=da)

    # Create pivot with proportions of graduates in ea. age and gender
    dx = da[~da.DMDEDUC2x.isin(["Don't know", "Missing"])]  # Eliminate rare/missing values
    dx = dx.groupby(["agegrp", "RIAGENDRx"])["DMDEDUC2x"].value_counts()
    dx = dx.unstack()  # Restructure the results from 'long' to 'wide', now ea. value in DMDEDUC2x is a column
    dx = dx.apply(lambda x: x / x.sum(), axis=1)  # Normalize within each stratum to get proportions
    print(dx)

    plt.show()

if __name__ == '__main__':

    analyze(plot=5)

