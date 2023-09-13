import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy import stats

# custom imports
import wrangle
import prepare



# ------------------------------- DATA DISTRIBUTION FUNCTION -----------------------------
def dist_of_data(df, column_name):
    # Define logarithmically spaced bin edges
    bins = np.logspace(np.log10(df[column_name].min()), np.log10(df[column_name].max()), num=20)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=column_name, bins=bins, color='orange', element='step', edgecolor='black')
    
    # Customize x-axis tick labels to display user-friendly values
    tick_values = [0, 500_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000]  # Adjust as needed
    tick_labels = ['$0', '$0.5M', '$1M', '$1.5M', '$2M', '$2.5M', '$3M']
    
    plt.xscale('linear')  # Use a linear scale for the x-axis
    plt.xticks(tick_values, tick_labels)  # Set custom tick values and labels
    plt.title(f'Distribution of Home Value')
    plt.xlabel('Home Value')
    plt.ylabel('Frequency')
    plt.xlim(0, 3000000)
    
# need to revisit

# def dist_of_objects(df):
    
#     for col in df.columns[df.dtypes == 'object']:
    
#         plt.figure(df)
#         sns.countplot(data = df, x = col)
#         plt.title(f'Count of {col}')
#         plt.show()




# ------------------------------- VAR PAIRS FUNCTION --------------------------------------
# defined function for plotting all variable pairs.
def plot_variable_pairs(df):
    """
    This function plots all of the pairwise relationships along with the regression line for each pair.

    Args:
      df: The dataframe containing the data.

    Returns:
      None.
    """
    sns.set(style="ticks")
    
    # Created a pairplot with regression lines
    sns.pairplot(df, kind="reg", diag_kind="kde", corner=True)
    plt.show()

def data_visual(df):
    # displaying feature percentages
    plt.figure(figsize=(12,20))
    df.drop('parcel_id',axis=1).notnull().mean().sort_values(ascending = True).plot(kind = 'barh')
    plt.title('Percentage of Present Information by Feature')
    plt.show()


# ------------------------------- CAT|CONT VARS FUNCTION --------------------------------------
def plot_categorical_and_continuous_vars(df, continuous_var, categorical_var, n):
    """
    This function outputs three different plots for visualizing a categorical variable and a continuous variable.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        continuous_var (str): The name of the column that holds the continuous feature.
        categorical_var (str): The name of the column that holds the categorical feature.

    Returns:
        None.
    """

    # Created subplots with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Box plot of the continuous variable for each category of the categorical variable
    sns.boxplot(x=categorical_var, y=continuous_var, data=df, ax=axes[0])
    axes[0].set_title('Box plot of {} for each category of {}'.format(continuous_var, categorical_var))

    # Violin plot of the continuous variable for each category of the categorical variable
    sns.scatterplot(x=categorical_var, y=continuous_var, data=df, ax=axes[1])
    axes[1].set_title('Scatter plot of {} for each category of {}'.format(continuous_var, categorical_var))

    # Histogram of the continuous variable for each category of the categorical variable
    for cat in df[categorical_var].unique():
        sns.histplot(df[df[categorical_var] == cat][continuous_var], ax=axes[2], label=cat, kde=True, bins=n)
    axes[2].set_title('Histogram of {} for each category of {}'.format(continuous_var, categorical_var))
    axes[2].legend(title=categorical_var)

    plt.tight_layout()
    plt.show()
'''
Example:

plot_categorical_and_continuous_vars(your_dataframe, "continuous_column", "categorical_column")
'''




    

# __________________________________________ STATS test functions _________________________________________

def evaluate_correlation(x, y, a=0.05, method="Pearson"):
    """
    Calculate and evaluate the correlation between two variables.

    Parameters:
    - x: First variable.
    - y: Second variable.
    - significance_level: The significance level for hypothesis testing.
    - method: The correlation method to use ("Pearson" or "Spearman").

    Returns:
    - correlation_coefficient: The correlation coefficient.
    - p_value: The p-value for the correlation test.
    - conclusion: A string indicating whether to reject or fail to reject the null hypothesis.
    """

    if method == "Pearson":
        
        correlation_coefficient, p_value = stats.pearsonr(x, y)
        
    elif method == "Spearman":
        
        correlation_coefficient, p_value = stats.spearmanr(x, y)
        
    else:
        
        raise ValueError("Invalid correlation method. Use 'Pearson' or 'Spearman'.")

    
    if p_value < a:
        
        conclusion = (f"Reject the null hypothesis.\n\nThere is a significant linear correlation between {x.name} and {y.name}.")
        
    else:
        
        conclusion = (f"Fail to reject the null hypothesis.\n\nThere is no significant linear correlation between {x.name} and {y.name}.")

    return correlation_coefficient, p_value, conclusion


# you don't hvae to round your coefficient, my preference.
# Replace x and y positional arguements with your actual data in the function
# correlation_coefficient, p_value, conclusion = explore.evaluate_correlation(train.tax_amount, train.area, method="Pearson")
# print(f'{conclusion}\n\nCorrelation Coefficient: {correlation_coefficient:.4f}\n\np-value: {p_value}')


# ---------------------- visual functions for analysis ----------------------------

def analysis_1(data):
    
    data = data[data['bedrooms'] <= 8]
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x='bedrooms', y='home_value', color='orange').set(title='Bedrooms Drive Home Value')
    plt.show()

def analysis_2(data):
    
    # Create a new column 'decade' by binning 'year_built' values into decades
    data['decade'] = (data['year_built'] // 10) * 10

    # Create the bar plot to visualize the relationship between 'decade' and 'home_value'
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x='decade', y='home_value', color='orange')
    plt.xlabel('Decade')
    plt.ylabel('Home Value')
    plt.title('Home Value by Decade Built')
    plt.xticks(rotation=45)
    plt.show()

def analysis_3(data):  
    # Create an lmplot to visualize the relationship between 'home_value' and 'area'
    sns.set(style="whitegrid")
    g = sns.lmplot(data=data, x='home_value', y='area', scatter_kws={'color': 'orange'}, line_kws={'color': 'red'}, height=6, aspect=1.5)
    
    # Customize plot title and axis labels
    g.set(title='Property Area Drives Home Value', ylabel='Area', xlabel='Home Value')
    
    # Calculate Pearson correlation coefficient and p-value
    correlation_coefficient, p_value = pearsonr(data['area'], data['home_value'])

    # Determine the conclusion based on the p-value
    if p_value < 0.05:
        conclusion = 'There is a statistically significant correlation between Area and Home Value.'
    else:
        conclusion = 'There is no statistically significant correlation between Area and Home Value.'

    print(f'\n\n{conclusion}\n\np-value: {p_value}\n\n')

def analysis_4(data):
    # Create an lmplot to visualize the relationship between 'home_value' and 'lot_area'
    sns.set(style="whitegrid")
    g = sns.lmplot(data=data, x='home_value', y='lot_area', scatter_kws={'color': 'orange'}, line_kws={'color': 'red'}, height=6, aspect=1.5)
    
    # Customize plot title and axis labels
    g.set(title='Property Lot Area Drives Home Value', ylabel='Lot Area', xlabel='Home Value')
    
    # Calculate Spearman correlation coefficient and p-value
    correlation_coefficient, p_value = spearmanr(data['lot_area'], data['home_value'])

    # Determine the conclusion based on the p-value
    if p_value < 0.05:
        conclusion = 'There is a statistically significant correlation between Lot Area and Home Value.'
    else:
        conclusion = 'There is no statistically significant correlation between Lot Area and Home Value.'

    print(f'\n\n{conclusion}\n\nCorrelation Coefficient: {correlation_coefficient:.4f}\n\np-value: {p_value}\n\n')
