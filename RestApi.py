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

from flask import request, url_for
from flask_api import FlaskAPI, status, exceptions
import ast

from StateFarmModel import fixColumns, fixStringColumns, calculate75thPercentile, trainModel

print("python version " + sys.version)
print('numpy version ' + np.__version__)
print('pandas version ' + pd.__version__)
print('sklern version ' + '0.23.1')
print('bokeh version ' + bokeh.__version__)
print('statsmodels version ' + '0.9.0')

# train model and get necessary variables
variables, final_result, std_scaler, imputer, Outcomes_train_final, train, train_imputed = trainModel()
cutoff_75 = calculate75thPercentile(Outcomes_train_final)
print('75th percentile value: ' + str(cutoff_75))

# function that determines whether the value is in the 75th percentile
def score(row):
    if row['phat'] >= cutoff_75:
        return 1
    return 0

app = FlaskAPI(__name__)

@app.route("/predict", methods=['POST'])
def predict():
    print('request received')
    if request.method == 'POST':
        try:
            req_data = ast.literal_eval(str(request.data))

            # if request is an individual row, place in list in order to convert to DataFrame
            if isinstance(req_data, dict):
                req_data = [req_data]

            api_raw = pd.DataFrame(req_data)
            api_fixed = fixColumns(api_raw)
            api_dummy = fixStringColumns(api_fixed, variables, std_scaler, imputer, train, train_imputed)

            prediction = pd.DataFrame(final_result.predict(api_dummy[variables])).rename(columns={0:'phat'})

            scored = pd.concat([api_dummy, prediction], axis=1, sort=False)
            scored['business_outcome'] = scored.apply (lambda row: score(row), axis=1)
            scored = scored.reindex(sorted(scored.columns), axis=1)
        except:
            return {'bad': 'request'}, status.HTTP_400_BAD_REQUEST

        return scored.to_json(orient='records'), status.HTTP_201_CREATED

    # request.method != 'POST'
    return {'bad': 'request'}, status.HTTP_405_METHOD_NOT_ALLOWED

if __name__ == "__main__":
    app.run(port=8888,debug=True)
