
# coding: utf-8

# In[ ]:


from flask import Flask, redirect, url_for, request
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import anderson, norm, chi2_contingency
from statsmodels.stats.weightstats import ztest
from sklearn.base import BaseEstimator, TransformerMixin


def replace_labels(ax, char, axis='y'):
    # Customize y-labels
    if axis == 'x':
        _ = ticks_loc = ax.get_xticks().tolist()
        _ = ax.set_xticks(ticks_loc)
        _ = ax.set_xticklabels([f'{x:,.0f}{char}' for x in ticks_loc])
    if axis == 'y':
        _ = ticks_loc = ax.get_yticks().tolist()
        _ = ax.set_yticks(ticks_loc)
        _ = ax.set_yticklabels([f'{x:,.0f}{char}' for x in ticks_loc])
    
def perform_anderson_test(data):
	result = anderson(data)
	print('Statistic = %.2f' % result.statistic)

	p = 0
	for i in range(len(result.critical_values)):
		sl, cv = result.significance_level[i], result.critical_values[i]
		if result.statistic < result.critical_values[i]:
			print(f'significance level = {sl:.2f}, critical value = {cv:.2f}, (fail to reject H0)')
		else:
			print(f'significance level = {sl:.2f}, critical value = {cv:.2f}, (reject H0)')
	print('\n')

def get_test_result(name, score, p_value, significance=0.05):
    # Test the p-value
    print(f"{name}")
    print(f"Score: {score:0.2f} and p-value: {p_value:0.2f}")
    if (p_value < significance):
        print(f'H0 can be rejected!')
    else:
        print('Fail to reject H0')

def custom_countplot(x, y, **kwargs):
    ax = sns.barplot(x=x, y=y, estimator=lambda x: len(x) / len(df) * 100)
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    replace_labels(ax, "%")

class TemporalVariableTransformer(BaseEstimator, TransformerMixin):
	# Temporal elapsed time transformer

    def __init__(self, variables, reference_variable):
        
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        
        self.variables = variables
        self.reference_variable = reference_variable

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):

    	# so that we do not over-write the original dataframe
        X = X.copy()
        
        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]

        return X



# categorical missing value imputer
class Mapper(BaseEstimator, TransformerMixin):

    def __init__(self, variables, mappings):

        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables
        self.mappings = mappings

    def fit(self, X, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.mappings)

        return X


class MeanImputer(BaseEstimator, TransformerMixin):
    """Numerical missing value imputer."""

    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        self.variables = variables

    def fit(self, X, y=None):
        # persist mean values in a dictionary
        self.imputer_dict_ = X[self.variables].mean().to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature],
                              inplace=True)
        return X



class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Groups infrequent categories into a single string"""

    def __init__(self, tol=0.05, variables):

        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        
        self.tol = tol
        self.variables = variables

    def fit(self, X, y=None):
        # persist frequent labels in dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            # the encoder will learn the most frequent categories
            t = pd.Series(X[var].value_counts(normalize=True) 
            # frequent labels:
            self.encoder_dict_[var] = list(t[t >= self.tol].index)

        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(
                X[feature].isin(self.encoder_dict_[feature]),
                                X[feature], "Rare")

        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """String to numbers categorical encoder."""

    def __init__(self, variables):

        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        
        self.variables = variables

    def fit(self, X, y):
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ["target"]

        # persist transforming dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            t = temp.groupby([var])["target"].mean().sort_values(ascending=True).index
            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}

        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])

        return X


# Credits to: https://www.kaggle.com/code/kabure/extensive-eda-and-modeling-xgb-hyperopt
# Below code from Kaggle

def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary

## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def CalcOutliers(df_num): 

    # calculating mean and std of the array
    data_mean, data_std = np.mean(df_num), np.std(df_num)

    # seting the cut line to both higher and lower values
    # You can change this value
    cut = data_std * 3

    #Calculating the higher and lower cut values
    lower, upper = data_mean - cut, data_mean + cut

    # creating an array of lower, higher and total outlier values 
    outliers_lower = [x for x in df_num if x < lower]
    outliers_higher = [x for x in df_num if x > upper]
    outliers_total = [x for x in df_num if x < lower or x > upper]

    # array without outlier values
    outliers_removed = [x for x in df_num if x > lower and x < upper]
    
    print('Identified lowest outliers: %d' % len(outliers_lower)) # printing total number of values in lower cut of outliers
    print('Identified upper outliers: %d' % len(outliers_higher)) # printing total number of values in higher cut of outliers
    print('Total outlier observations: %d' % len(outliers_total)) # printing total number of values outliers of both sides
    print('Non-outlier observations: %d' % len(outliers_removed)) # printing total number of non outlier values
    print("Total percentual of Outliers: ", round((len(outliers_total) / len(outliers_removed) )*100, 4)) # Percentual of outliers in points
    
    return

			  
# Display value counts and NaN counts
def show_value_counts(df, column):
    value_count_df = pd.DataFrame(df[column].value_counts().rename_axis(column).reset_index(name='counts'))
    value_count_df['percentage'] = round(100 * (value_count_df['counts'] / len(df)),1)
    return value_count_df

def show_nan_counts(df):
    nan_count_df = pd.DataFrame(df.isna().sum()).sort_values(by=0, ascending = False)
    nan_count_df.columns = ['counts']
    nan_count_df['percentage'] = round(100 * (nan_count_df['counts'] / len(df)),1)
    return nan_count_df

# Display correlation plot as heatmap
def show_heatmap(data, figsize=(12,8), 
                 highest_only=False, 
                 thresold=0.7, # Look at only highly correlated pairs
                 annot=False):
    correlation_matrix = data.corr()
    high_corr = correlation_matrix[np.abs(correlation_matrix )>= thresold]
    plt.figure(figsize=figsize)

    if highest_only:
        sns.heatmap(high_corr, annot=annot, cmap="Greens", 
                    linecolor='black', linewidths=0.1)
    else:
        sns.heatmap(correlation_matrix, annot=annot)
			  
			  
# Display value counts and NaN counts
def show_value_counts(df, column):
    value_count_df = pd.DataFrame(df[column].value_counts().rename_axis(column).reset_index(name='counts'))
    value_count_df['percentage'] = round(100 * (value_count_df['counts'] / len(df)),1)
    return value_count_df

def show_nan_counts(df):
    nan_count_df = pd.DataFrame(df.isna().sum()).sort_values(by=0, ascending = False)
    nan_count_df.columns = ['counts']
    nan_count_df['percentage'] = round(100 * (nan_count_df['counts'] / len(df)),1)
    return nan_count_df

# Display correlation plot as heatmap
def show_heatmap(data, figsize=(12,8), 
                 highest_only=False, 
                 thresold=0.7, # Look at only highly correlated pairs
                 annot=False):
    correlation_matrix = data.corr()
    high_corr = correlation_matrix[np.abs(correlation_matrix )>= thresold]
    plt.figure(figsize=figsize)

    if highest_only:
        sns.heatmap(high_corr, annot=annot, cmap="Greens", 
                    linecolor='black', linewidths=0.1)
    else:
        sns.heatmap(correlation_matrix, annot=annot)

			  
			  
			  
##########################################################################################################
# Data preprocessing functions are listed here.
##########################################################################################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

#############################################
# 1. OUTLIERS
#############################################

def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#############################################
# 2. MISSING VALUES
#############################################
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

#############################################
# 3. ENCODING
#############################################
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

#############################################
# 4. RARE ENCODING
#############################################

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

def rare_encoder(dataframe, rare_perc, cat_cols):
    rare_columns = [col for col in cat_cols if (dataframe[col].value_counts() / len(dataframe) < 0.01).sum() > 1]

    for col in rare_columns:
        tmp = dataframe[col].value_counts() / len(dataframe)
        rare_labels = tmp[tmp < rare_perc].index
        dataframe[col] = np.where(dataframe[col].isin(rare_labels), 'Rare', dataframe[col])

    return dataframe

########################################################################################			  
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor
from mlxtend.regressor import StackingCVRegressor

from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error  
from sklearn.ensemble import ExtraTreesRegressor

et  = ExtraTreesRegressor(n_estimators=950 ,  max_features = 'auto', max_leaf_nodes=None, n_jobs= -1, random_state = 0, verbose = 0)
gbr = GradientBoostingRegressor()
lasso = Lasso()
xgbr = XGBRegressor()
svr = SVR(kernel= 'rbf', gamma= 'auto', tol=0.001, C=100.0, max_iter=-1)
rf = RandomForestRegressor(n_estimators=900,  random_state=0)
lr = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=-1)
knnR = KNeighborsRegressor(n_neighbors=20, n_jobs=-1)
reg = StackingCVRegressor(regressors=[  lasso , xgbr , et],meta_regressor=lasso)

reg.fit(x_train, y_train, groups = None)
			  
##################################################################################			  
			  
app = Flask(__name__)

@app.route('/success/<name>')
def success(name):
    return 'welcome %s' % name

@app.route('/login',methods = ['POST','GET'])
def login():
    if request.method == 'POST':
        user = request.form['nm']
        return redirect(url_for('success',name = user))
    else:
        user = request.args.get('nm')
        return redirect(url_for('success',name = user))

if __name__ == '__main__':
    app.run(debug = True)

