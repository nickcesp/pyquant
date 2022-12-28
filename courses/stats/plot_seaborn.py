# Examples plotting with seaborn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def do_plots(ptype='reg'):

    # Store the url string that hosts our .csv file
    data_path = "resources/Cartwheeldata.csv"

    # Read the .csv file and store it as a pandas Data Frame
    df = pd.read_csv(data_path)

    # Create Scatterplot
    if ptype == 'reg':
        sns.lmplot(x='Wingspan', y='CWDistance', data=df)
    elif ptype == 'scatter':
        sns.lmplot(x='Wingspan', y='CWDistance', data=df,
                   fit_reg=False,  # No regression line
                   hue='Gender')  # Color by evolution stage
    elif ptype == 'swarm':
        sns.swarmplot(x="Gender", y="CWDistance", data=df, hue='Gender')
    elif ptype == 'gen_box':
        sns.boxplot(data=df.loc[:, ["Age", "Height", "Wingspan", "CWDistance", "Score"]])
    elif ptype == 'hist':
        sns.distplot(df.CWDistance)
    elif ptype == 'count':
        # Count Plot (a.k.a. Bar Plot)
        sns.countplot(x='Gender', data=df)
        plt.xticks(rotation=-45)
    else:
        raise ValueError("Invalid ptype")

    plt.show()

def more_seaborn(which=1):

    tips_data = sns.load_dataset('tips')

    # Shows numerical statistics using seaborn
    print(tips_data.describe())

    if which == 1:
        # Plot a histogram of the total bill
        sns.histplot(tips_data["total_bill"], kde=False).set_title("Histogram of Total Bill")
    elif which == 2:
        # Plot a histogram of the Tips only
        sns.histplot(tips_data["tip"], kde=False).set_title("Histogram of Total Tip")
    elif which == 3:
        # Plot a histogram of both the total bill and the tips'
        sns.histplot(tips_data['total_bill'], kde=False)
        sns.histplot(tips_data['tip'], kde=False).set_title('Both thangs')
    elif which == 4:
        # Create a boxplot of the total bill amounts
        sns.boxplot(tips_data["total_bill"]).set_title("Box plot of the Total Bill")
    elif which == 5:
        # Create a boxplot of the tips and total bill amounts - do not do it like this, DO IT BY GROUPS
        sns.boxplot(tips_data["total_bill"])
        sns.boxplot(tips_data["tip"]).set_title("Box plot of the Total Bill and Tips")
    elif which == 6:
        # Create a boxplot and histogram of the tips grouped by smoking status
        sns.boxplot(x=tips_data["tip"], y=tips_data["smoker"])
    elif which in (7, 8):
        # Create a boxplot and histogram of the tips grouped by time of day or the day
        grp_by = 'time' if which == 7 else 'day'
        sns.boxplot(x=tips_data["tip"], y=tips_data[grp_by])

        g = sns.FacetGrid(tips_data, row=grp_by)
        _ = g.map(plt.hist, "tip")

    plt.show()


if __name__ == '__main__':

    #do_plots(ptype='hist')
    more_seaborn(which=8)