import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

def biv_analysis(chart=0):
    """ Examples from understanding-visualization-data wk 3 """
    da = pd.read_csv("resources/nhanes_2015_2016.csv")

    # New variables
    # For chart 4
    da["RIAGENDRx"] = da.RIAGENDR.replace({1: "Male", 2: "Female"})

    # For Chart 5
    educ_repl = {1: "<9", 2: "9-11", 3: "HS/GED", 4: "Some college/AA", 5: "College", 7: "Refused", 9: "Don't know"}
    da["DMDEDUC2x"] = da.DMDEDUC2.replace(educ_repl)
    married_rpl = {1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "Never married",
                   6: "Living w/partner", 77: "Refused"}
    da["DMDMARTLx"] = da.DMDMARTL.replace(married_rpl)


    if chart == 0:
        sns.regplot(x="BMXLEG", y="BMXARML", data=da, fit_reg=False, scatter_kws={"alpha": 0.2})
    elif chart == 1:
        sns.jointplot(x="BMXLEG", y="BMXARML", kind='kde', data=da).annotate(stats.pearsonr)
    elif chart == 2:
        sns.jointplot(x="BPXSY1", y="BPXDI1", kind='kde', data=da).annotate(stats.pearsonr)
    elif chart == 3:
        sns.jointplot(x="BPXSY1", y="BPXSY2", kind='kde', data=da).annotate(stats.pearsonr)
    elif chart == 4:
        # Multiple charts stratified
        _ = sns.FacetGrid(da, col="RIDRETH1", row="RIAGENDRx").map(plt.scatter, "BMXLEG", "BMXARML",
                                                                   alpha=0.5).add_legend()
    elif chart == 5:
        # Remove people who didn't answer
        db = da.loc[(da.DMDEDUC2x != "Don't know") & (da.DMDMARTLx != "Refused"), :]
        # Cross tabulation
        x = pd.crosstab(db.DMDEDUC2x, da.DMDMARTLx)
        print(x)
        # Normalize rows (make them sum to 1)
        x_row_norm = x.apply(lambda z: z / z.sum(), axis=1)
        x_col_norm = x.apply(lambda z: z / z.sum(), axis=0)

        # The following line does these steps, reading the code from left to right:
        # 1 Group the data by every combination of gender, education, and marital status
        # 2 Count the number of people in each cell using the 'size' method
        # 3 Pivot the marital status results into the columns (using unstack)
        # 4 Fill any empty cells with 0
        # 5 Normalize the data by row
        db.groupby(["RIAGENDRx", "DMDEDUC2x", "DMDMARTLx"]).size().unstack().fillna(0).apply(lambda z: z / z.sum(),
                                                                                             axis=1)
    elif chart == 6:
        sns.violinplot(x=da.DMDMARTLx, y=da.RIDAGEYR)

    plt.show()

if __name__ == '__main__':
    biv_analysis(chart=6)