import numpy as np
import pandas as pd
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
#loading visualization library
import bokeh

import collections as ccc

print("python version " + sys.version)
print('numpy version ' + np.__version__)
print('pandas version ' + pd.__version__)
print('sklern version ' + '0.23.1')
print('bokeh version ' + bokeh.__version__)
print('statsmodels version ' + '0.9.0')

# Investigate Object Columns
def investigate_object(df):
    """
    This function prints the unique categories of all the object dtype columns.
    It prints '...' if there are more than 13 unique categories.
    """
    col_obj = df.columns[df.dtypes == 'object']

    for i in range(len(col_obj)):
        if len(df[col_obj[i]].unique()) > 13:
            print(col_obj[i]+":", "Unique Values:", np.append(df[col_obj[i]].unique()[:13], "..."))
        else:
            print(col_obj[i]+":", "Unique Values:", df[col_obj[i]].unique())

    del col_obj

#1. Fixing the money and percents#
def fixColumns(ugly_df):
    ugly_df['x12'] = ugly_df['x12'].str.replace('$','')
    ugly_df['x12'] = ugly_df['x12'].str.replace(',','')
    ugly_df['x12'] = ugly_df['x12'].str.replace(')','')
    ugly_df['x12'] = ugly_df['x12'].str.replace('(','-')
    ugly_df['x12'] = ugly_df['x12'].astype(float)
    ugly_df['x63'] = ugly_df['x63'].str.replace('%','')
    ugly_df['x63'] = ugly_df['x63'].astype(float)
    return ugly_df


# one-hot encoding function for all columns with string values
def fixStringColumns(df_strings, trainingVariables, std_scaler, imputer, train, train_imputed):

    if 'y' in df_strings:
        df_imputed = pd.DataFrame(imputer.transform(df_strings.drop(columns=['y', 'x5', 'x31', 'x81' ,'x82'])), columns=train.drop(columns=['y','x5', 'x31', 'x81', 'x82']).columns)
    else:
        df_imputed = pd.DataFrame(imputer.transform(df_strings.drop(columns=['x5', 'x31', 'x81' ,'x82'])), columns=train.drop(columns=['y','x5', 'x31', 'x81', 'x82']).columns)

    df_imputed_std = pd.DataFrame(std_scaler.transform(df_imputed), columns=train_imputed.columns)

    dumb5 = pd.get_dummies(df_strings['x5'], drop_first=True, prefix='x5', prefix_sep='_', dummy_na=True)
    df_imputed_std = pd.concat([df_imputed_std, dumb5], axis=1, sort=False)

    dumb31 = pd.get_dummies(df_strings['x31'], drop_first=True, prefix='x31', prefix_sep='_', dummy_na=True)
    df_imputed_std = pd.concat([df_imputed_std, dumb31], axis=1, sort=False)

    dumb81 = pd.get_dummies(df_strings['x81'], drop_first=True, prefix='x81', prefix_sep='_', dummy_na=True)
    df_imputed_std = pd.concat([df_imputed_std, dumb81], axis=1, sort=False)

    dumb82 = pd.get_dummies(df_strings['x82'], drop_first=True, prefix='x82', prefix_sep='_', dummy_na=True)
    df_imputed_std = pd.concat([df_imputed_std, dumb82], axis=1, sort=False)

    # necessary due to the possibility of get_dummies() not having all of the possible string values in the api dataset
    for x in trainingVariables:
        if x not in df_imputed_std:
            df_imputed_std[x] = 0

    return df_imputed_std

def getSets():
    raw_train=pd.read_csv('exercise_26_train.csv')

    # Overview of data types
    print("object dtype:", raw_train.columns[raw_train.dtypes == 'object'].tolist())
    print("int64 dtype:", raw_train.columns[raw_train.dtypes == 'int'].tolist())
    print("The rest of the columns have float64 dtypes.")

    investigate_object(raw_train)

    train_val = raw_train.copy(deep=True)

    train_val = fixColumns(train_val)

    # 2. Creating the train/val/test set
    x_train, x_val, y_train, y_val = train_test_split(train_val.drop(columns=['y']), train_val['y'], test_size=0.1, random_state=13)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=4000, random_state=13)

    # 3. smashing sets back together
    train = pd.concat([x_train, y_train], axis=1, sort=False).reset_index(drop=True)
    val = pd.concat([x_val, y_val], axis=1, sort=False).reset_index(drop=True)
    test = pd.concat([x_test, y_test], axis=1, sort=False).reset_index(drop=True)

    return train, val, test

def trainModel():

    train, val, test = getSets()

    # With mean imputation from Train set
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    train_imputed = pd.DataFrame(imputer.fit_transform(train.drop(columns=['y', 'x5', 'x31',  'x81' ,'x82'])), columns=train.drop(columns=['y', 'x5', 'x31', 'x81', 'x82']).columns)
    std_scaler = StandardScaler()
    train_imputed_std = pd.DataFrame(std_scaler.fit_transform(train_imputed), columns=train_imputed.columns)

    # create dummies

    dumb5 = pd.get_dummies(train['x5'], drop_first=True, prefix='x5', prefix_sep='_', dummy_na=True)
    train_imputed_std = pd.concat([train_imputed_std, dumb5], axis=1, sort=False)

    dumb31 = pd.get_dummies(train['x31'], drop_first=True, prefix='x31', prefix_sep='_', dummy_na=True)
    train_imputed_std = pd.concat([train_imputed_std, dumb31], axis=1, sort=False)

    dumb81 = pd.get_dummies(train['x81'], drop_first=True, prefix='x81', prefix_sep='_', dummy_na=True)
    train_imputed_std = pd.concat([train_imputed_std, dumb81], axis=1, sort=False)

    dumb82 = pd.get_dummies(train['x82'], drop_first=True, prefix='x82', prefix_sep='_', dummy_na=True)
    train_imputed_std = pd.concat([train_imputed_std, dumb82], axis=1, sort=False)
    train_imputed_std = pd.concat([train_imputed_std, train['y']], axis=1, sort=False)

    del dumb5, dumb31, dumb81, dumb82
    print(train.head())

    #Showing the imputer statistics
    imputer.statistics_

    #Showing the variance
    train_imputed.var()

    exploratory_LR = LogisticRegression(penalty='l1', fit_intercept=False, solver='liblinear')
    exploratory_LR.fit(train_imputed_std.drop(columns=['y']), train_imputed_std['y'])
    exploratory_results = pd.DataFrame(train_imputed_std.drop(columns=['y']).columns).rename(columns={0:'name'})
    exploratory_results['coefs'] = exploratory_LR.coef_[0]
    exploratory_results['coefs_squared'] = exploratory_results['coefs']**2
    var_reduced = exploratory_results.nlargest(25,'coefs_squared')

    variables = var_reduced['name'].to_list()

    logit = sm.Logit(train_imputed_std['y'], train_imputed_std[variables])
    # fit the model
    result = logit.fit()
    result.summary()

    val_imputed_std = fixStringColumns(val, variables, std_scaler, imputer, train, train_imputed)
    test_imputed_std = fixStringColumns(test, variables, std_scaler, imputer, train, train_imputed)

    val_imputed_std = pd.concat([val_imputed_std, val['y']], axis=1, sort=False)
    test_imputed_std = pd.concat([test_imputed_std, test['y']], axis=1, sort=False)

    Outcomes_train = pd.DataFrame(result.predict(train_imputed_std[variables])).rename(columns={0:'probs'})
    Outcomes_train['y'] = train_imputed_std['y']
    print('The C-Statistics is ',roc_auc_score(Outcomes_train['y'], Outcomes_train['probs']))
    Outcomes_val = pd.DataFrame(result.predict(val_imputed_std[variables])).rename(columns={0:'probs'})
    Outcomes_val['y'] = val_imputed_std['y']
    print('The C-Statistics is ',roc_auc_score(Outcomes_val['y'], Outcomes_val['probs']))
    Outcomes_test = pd.DataFrame(result.predict(test_imputed_std[variables])).rename(columns={0:'probs'})
    Outcomes_test['y'] = test_imputed_std['y']
    print('The C-Statistics is ',roc_auc_score(Outcomes_test['y'], Outcomes_test['probs']))
    Outcomes_train['prob_bin'] = pd.qcut(Outcomes_train['probs'], q=20)

    Outcomes_train.groupby(['prob_bin'])['y'].sum()

    train_and_val = pd.concat([train_imputed_std, val_imputed_std])
    all_train = pd.concat([train_and_val, test_imputed_std])
    final_variables = var_reduced['name'].to_list()
    final_logit = sm.Logit(all_train['y'], all_train[final_variables])
    # fit the model
    final_result = final_logit.fit()
    final_result.summary()

    Outcomes_train_final = pd.DataFrame(result.predict(all_train[variables])).rename(columns={0:'probs'})
    Outcomes_train_final['y'] = all_train['y']
    print('The C-Statistics is ',roc_auc_score(Outcomes_train_final['y'], Outcomes_train_final['probs']))
    Outcomes_train_final['prob_bin'] = pd.qcut(Outcomes_train_final['probs'], q=20)
    Outcomes_train_final.groupby(['prob_bin'])['y'].sum()

    return variables, final_result, std_scaler, imputer, Outcomes_train_final, train, train_imputed


def calculate75thPercentile(Outcomes_train_final):
    ranked = Outcomes_train_final['probs']
    ranked_sorted = ranked.sort_values(ascending=False)
    cutoff_index = int(ranked_sorted.count() / 4)
    return ranked_sorted.iloc[cutoff_index]
