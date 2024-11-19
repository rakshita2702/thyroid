import numpy as np, pandas as pd
from sklearn.experimental import enable_iterative_imputer # Must keep this line to enable MICE Imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTENC

# ---------- Main Pipeline ----------
def rough_preprocessor (x):

# 1. Missing Data
# The dataset contains many entries with missing data. There are two types of missing data: 
# partially missing (some data points missing in a column)
# and wholly missing (entire column has no data, as is the case of TBG)
# remove TSH_measured, T3_measured, TT4_measured, T4U_measured, FTI_measured, and TBG_measured as they give no useful information
# Also helpful to remove columns related to antithyroid medication, surgery, and TBG to prevent data leakeage.

    x = x.drop(['TSH_measured','T3_measured','TT4_measured','T4U_measured','FTI_measured','TBG_measured', 'TBG', 
'query_thyroxine', 'query_hyperthyroid', 'query_hypothyroid','antithyroid_meds','tumor',
'lithium','sick','thyroxine','thyroid_surgery'], axis= 1)

# Now we deal with columns with only partially missing data. First we replace `?` with np.NaN and correctly recast data types:

    x = x.replace('?',np.nan)
    x[['age', 'TSH', 'T3', 'TT4', 'T4U','FTI']] = x[['age', 'TSH', 'T3', 'TT4', 'T4U','FTI']].apply(
        pd.to_numeric, errors='coerce')

# Then we split the reminaing dataset into categorical and numerical columns and use SimpleImputer to fill in missing numerical values

    categorical_cols = [cname for cname in x.columns if x[cname].dtype == "object"]
    numerical_cols = [cname for cname in x.columns if x[cname].dtype in ['int64', 'float64']]
    x_num = x[numerical_cols].copy()
    x_cat = x[categorical_cols].copy()

# Because there is quite a number of data missing in each numerical column (100+ ~ 500+) 
# I decided on using MICE (Multivariate Imputation by Chained Equation) based on a Bayesian Ridge model 
# to maintain random variability and capture the relationship between thyroid hormone levels and age
# normalize quantitative variables (except age):
    
    mice_imputer = IterativeImputer(estimator= linear_model.BayesianRidge(), n_nearest_features= None, imputation_order= 'ascending')
    x_num = pd.DataFrame(mice_imputer.fit_transform(x_num), columns = numerical_cols)

# Add columns:
   
# Only a few data points were missing in the categorical columns
# Considering that women are more at risk of thyroid issues than men, more women are likely to be surveyed for this dataset
    
    x_cat['sex'] = x_cat['sex'].fillna('F')

# EXPERIMENTAL: Bin age into "old" (>median) and "not old" category
    med = x_num['age'].median()+0.01
    x_cat['is_old'] = x_num['age'].apply(lambda x: 'not_old' if x <= med else 'old')

    x_cat['is_old'].fillna(np.nan, inplace=True)
    m1 = (x_cat['is_old'].isnull()) & (x_num['age'] <= med)
    m2 = (x_cat['is_old'].isnull()) & (x_num['age'] > med)
    x_cat.loc[m1, 'is_old'] = x_cat.loc[m1, 'is_old'].fillna('not_old', inplace=True)
    x_cat.loc[m2, 'is_old'] = x_cat.loc[m2, 'is_old'].fillna('not_old', inplace=True)
    # TODO: Can I figure out how to convert columns returning NaN
    #x_cat['is_old'].fillna('unknown', inplace=True)
    print(x_cat['is_old'].isnull().sum())
    categorical_cols.append('is_old')
    
# Join numerical and categorical sub dataframes together and overwrite original dataframe
    x_num.index = x_cat.index
    x = pd.concat([x_num, x_cat], axis = 1)
    return(x, categorical_cols)

def training_pipeline(x,y,catcols):
    x_train = rough_preprocessor(x)[0]
    x_rand, y_rand = rough_oversampler_random(x_train,y)
    x_smote, y_smote = rough_oversampler_smote(x_train,y,catcols)
    return (x_rand, y_rand, x_smote, y_smote)

def val_test_pipeline(x):
    x_val = rough_preprocessor(x)[0]
    return(x_val)

# ---------- Oversampling methods ---------- :
# 1. Random Oversampling method (takes random records and replicates them):

def rough_oversampler_random (x, y):
    random_os = RandomOverSampler(sampling_strategy = 0.5, random_state = 123) # we just want minority (hypothyroid) class to appear more often
    # oversampling minority class until it equals majority class can lead to model overfitting
    x_rand, y_rand = random_os.fit_resample(x, y)
    return (x_rand, y_rand)

# 2. SMOTE oversampling method (makes up records based on available information):

def rough_oversampler_smote(x, y, cols):
    col_indices = [x.columns.get_loc(c) for c in cols]
    smote_os = SMOTENC(categorical_features= col_indices, random_state = 123, sampling_strategy= 0.5) # categorical data in last 4 columns 
    x_smote, y_smote = smote_os.fit_resample(x, y)
    return (x_smote, y_smote)
