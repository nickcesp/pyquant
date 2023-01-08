import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2  # for sig testing

def mixed_model(da):
    # Fit the Model without Centering We will first begin by fitting the model without centering the age
    # component first. This model has both random intercepts and random slopes on age.
    mlm_mod = sm.MixedLM.from_formula(
        formula='vsae ~ age * C(sicdegp)',
        groups='childid',
        re_formula="1 + age",
        data=dat
    )

    # Run the fit
    mlm_result = mlm_mod.fit()

    # Print out the summary of the fit
    print(mlm_result.summary())

    # Build the model - note the re_formula definition now
    # has a 0 instead of a 1. This removes the intercept from
    # the model
    mlm_mod = sm.MixedLM.from_formula(
        formula='vsae ~ age * C(sicdegp)',
        groups='childid',
        re_formula="0 + age",
        data=dat
    )
    mlm_result = mlm_mod.fit()
    print(mlm_result.summary())

    # Random Effects Mixed Model
    mlm_mod = sm.MixedLM.from_formula(
        formula='vsae ~ age * C(sicdegp)',
        groups='childid',
        re_formula="0 + age",
        data=dat
    )

    # OLS model - no mixed effects
    ols_mod = sm.OLS.from_formula(
        formula="vsae ~ age * C(sicdegp)",
        data=dat
    )

    # Run each of the fits
    mlm_result = mlm_mod.fit()
    ols_result = ols_mod.fit()

    # Print out the summary of the fit
    print(mlm_result.summary())
    print(ols_result.summary())

    # Compute the p-value using a mixture of chi-squared distributions
    # Because the chi-squared distribution with zero degrees of freedom has no
    # mass, we multiply the chi-squared distribution with one degree of freedom by
    # 0.5
    """
    Now, we perform the significance test with a mixture of chi-squared distributions. We repeat the information from the Likelihood Ratio Tests writeup for this week here:

    Null hypothesis: The variance of the random child effects on the slope of interest is zero (in other words, these random effects on the slope are not needed in the model)
    Alternative hypothesis: The variance of the random child effects on the slope of interest is greater than zero

    First, fit the model WITH random child effects on the slope of interest, using restricted maximum likelihood estimation
        -2 REML log-likelihood = 4854.18
    Next, fit the nested model WITHOUT the random child effects on the slope:
        -2 REML log-likelihood = 5524.20 (higher value = worse fit!)
    Compute the positive difference in the -2 REML log-likelihood values (“REML criterion”) for the models:
        Test Statistic (TS) = 5524.20 – 4854.18 = 670.02
    Refer the TS to a mixture of chi-square distributions with 1 and 2 DF, and equal weight 0.5:

    """
    pval = 0.5 * (1 - chi2.cdf(670.02, 1))
    print("The p-value of our significance test is: {0}".format(pval))

def marg_model(da):
    # Fit the exchangable covariance GEE
    model_exch = sm.GEE.from_formula(
        formula="vsae ~ age * C(sicdegp)",
        groups="childid",
        cov_struct=sm.cov_struct.Exchangeable(),
        data=dat
    ).fit()

    # Fit the independent covariance GEE
    model_indep = sm.GEE.from_formula(
        "vsae ~ age * C(sicdegp)",
        groups="childid",
        cov_struct=sm.cov_struct.Independence(),
        data=dat
    ).fit()

    # We cannot fit an autoregressive model, but this is how
    # we would fit it if we had equally spaced ages
    # model_indep = sm.GEE.from_formula(
    #     "vsae ~ age * C(sicdegp)",
    #     groups="age",
    #     cov_struct = sm.cov_struct.Autoregressive(),
    #     data=dat
    #     ).fit()


if __name__ == '__main__':
    # Read in the Autism Data
    dat = pd.read_csv("resources/autism.csv")

    # Drop NA's from the data
    dat = dat.dropna()

    #AGE is the age of a child which, for this dataset, is between two and thirteen years
    #VSAE measures a child's socialization
    #SICDEGP is the expressive language group at age two and can take on values ranging from one to three. Higher values indicate more expressive language.
    #CHILDID is the unique ID that is given to each child and acts as their identifier within the dataset

    print(dat.head())

    """We can see that the estimates for the parameters are relatively consistent among each of the modeling methodologies,
     but the standard errors differ from model to model. Overall, the two GEE models are mostly similar and both exhibit 
     standard errors for parameters that are slightly larger than each of their corresponding values in the OLS model. 
     The multilevel model has the largest standard error for the age coefficient, but the smallest standard error 
     for the intercept. Overall, we see that we would make similar inferences regarding the importance of these fixed
     effects, but remember that we need to interpret the multilevel models estimates conditioning on a given child.
     For example, considering the age coefficient in the multilevel model, we would say that as age increases by one
     year for a given child in the first expressive language group, VSAE is expected to increase by 2.73. In the GEE
     and OLS models, we would say that as age increases by one year in general, the average VSAE is expected to 
     increase by 2.60."""