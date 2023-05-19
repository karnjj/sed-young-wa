from datetime import  datetime, timedelta
from sklearn.preprocessing import StandardScaler
import warnings
import os
import json
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature



warnings.simplefilter(action='ignore', category=DeprecationWarning)


if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

def build_train_test(df, TIME_OFFSET):
    k_test = 7 #@param {type:"integer"}
    k_train = 25 #@param {type:"integer"}

    current_date = datetime.now() - timedelta(days=TIME_OFFSET)
    start_test_date = current_date - timedelta(days=k_test)
    start_train_date = current_date- timedelta(days=k_test) - timedelta(days=k_train)

    print("train on ", start_train_date, " to ",start_test_date)
    print("test on ", start_test_date, " to ",current_date)

    # filter data
    df_test = df[df['timestamp'] > start_test_date]
    df_train = df[( df['timestamp'] > start_train_date ) & ( df['timestamp'] < start_test_date )]
    total_df = df_test.shape[0] + df_train.shape[0]
    print(f"test set: {df_test.shape[0]}\n train set: {df_train.shape[0]}\n")
    print(f"test set: {df_test.shape[0]/total_df}\n train set: {df_train.shape[0]/total_df}")

    start_column_idx = list(df_train.columns).index('time_of_day')

    X_train = df_train.iloc[:,start_column_idx:]
    y_train = df_train["process_hour"]

    X_test = df_test.iloc[:,start_column_idx:]
    y_test = df_test["process_hour"]
    
    return X_train, y_train, X_test, y_test

def standardScaler(X_train, X_test):
    scaler = StandardScaler()
    X_train[:] = scaler.fit_transform(X_train)
    X_test[:] = scaler.transform(X_test)

    return X_train, X_test, scaler

def train_model(X_train, y_train):
    cv = KFold(5, random_state = 1, shuffle=True)
    param_grid = {'n_estimators': [50],
                'max_depth': [7], #range(5,16,2), 
                'min_samples_split': range(100,1001,200), #range(100,1001,200), 
                'learning_rate':[0.2]}
    clf = GridSearchCV(GradientBoostingRegressor(random_state=1), 
                    param_grid = param_grid, scoring='r2', 
                    cv=cv).fit(X_train, y_train)

    best_model = clf.best_estimator_
    print(best_model) 
    print("R Squared:",clf.best_score_)

    return best_model, cv

def make_score(best_model, cv, X_test, y_test):
    r2_test = cross_val_score(best_model, X_test, y_test, cv=cv, scoring='r2').mean()
    RMSE_test = np.sqrt((-1) * cross_val_score(best_model, X_test, y_test, cv=cv, scoring='neg_mean_squared_error').mean())

    return r2_test, RMSE_test

@data_exporter
def export_data(df, *args, **kwargs):

    X_train, y_train, X_test, y_test = build_train_test(df, kwargs['TIME_OFFSET'])

    X_train, X_test, scaler = standardScaler(X_train, X_test)

    file_name = os.path.join(kwargs['CONFIG_DIR'], "scalar.pkl")
    with open(file_name,'wb') as f:
        pickle.dump(scaler, f)

    with mlflow.start_run():
    
        model, cv = train_model(X_train, y_train)

        r2_test, RMSE_test = make_score(model, cv, X_test, y_test)
        print(" == Test set ==")
        print("R Seuared of test set: %.2f" % r2_test)
        print("RMSE of test set: %.2f" % RMSE_test)

        signature = infer_signature(X_test)

        mlflow.log_metric("rmse", RMSE_test)
        mlflow.log_metric("r2", r2_test)

        mlflow.log_artifact(os.path.join(kwargs['CONFIG_DIR'], "scalar.pkl"), "model/pipeline_config") 
        mlflow.log_artifact(os.path.join(kwargs['CONFIG_DIR'], "selected_feature_value.json"), "model/pipeline_config")

        mlflow.sklearn.log_model(model, "model", signature=signature, registered_model_name="gradient-boosting-reg-model")

    # Save model
    filename = os.path.join(kwargs['CONFIG_DIR'],'model_satee.sav')
    pickle.dump(model, open(filename, 'wb'))
    
    # Specify your data exporting logic here

