import pandas as pd
import statsmodels.api as sm
import numpy as np

def ols(df):
    """ Ordinary linear regression """
    model = sm.OLS.from_formula("BPXSY1 ~ RIDAGEYR + RIAGENDRx", data=df)
    res = model.fit()
    print(res.summary())

def glm(df):
    """ Generalized linear model - eg. logistic regression """
    model = sm.GLM.from_formula("smq ~ RIAGENDRx", family=sm.families.Binomial(), data=df)
    res = model.fit()
    print(res.summary())

def glm2(df):
    model = sm.GLM.from_formula("smq ~ RIAGENDRx + RIDAGEYR + DMDEDUC2x", family=sm.families.Binomial(), data=df)
    res = model.fit()
    print(res.summary())

if __name__ == '__main__':
    df = pd.read_csv("resources/nhanes_2015_2016.csv")

    # Drop unused columns, drop rows with any missing values.
    vars = ["BPXSY1", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "DMDEDUC2", "BMXBMI",
            "SMQ020", "SDMVSTRA", "SDMVPSU"]
    df = df[vars].dropna()

    # Data Transformations
    df["RIAGENDRx"] = df.RIAGENDR.replace({1: "Male", 2: "Female"})
    df["smq"] = df.SMQ020.replace({2: 0, 7: np.nan, 9: np.nan})
    df["DMDEDUC2x"] = df.DMDEDUC2.replace({1: "lt9", 2: "x9_11", 3: "HS", 4: "SomeCollege", 5: "College", 7: np.nan, 9: np.nan})

    ols(df)

    glm(df)

    glm2(df)