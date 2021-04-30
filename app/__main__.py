import math as math
import sys
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from pyspark.sql import SparkSession
from sklearn.model_selection import GridSearchCV


def main():
    spark = SparkSession \
        .builder \
        .appName("MLops_search_python") \
        .getOrCreate()
    uid = sys.argv[1]
    spark.conf.set("spark.sql.execution.arrow.enabled", True)
    print(f"reading delta table: dbfs:/datalake/stocks_{uid}/data")
    try:
        df = spark.read.format("delta").load(f"dbfs:/datalake/stocks_{uid}/data")
    except Exception as e:
        print(f"There was an error loading the delta stock table, : error:{e}")
    pdf = df.select("*").toPandas()
    df_2 = pdf.loc[:, ["AdjClose", "Volume"]]
    df_2["High_Low_Pert"] = (pdf["High"] - pdf["Low"]) / pdf["Close"] * 100.0
    df_2["Pert_change"] = (pdf["Close"] - pdf["Open"]) / pdf["Open"] * 100.0
    df_2.fillna(value=-99999, inplace=True)
    forecast_out = int(math.ceil(0.01 * len(df_2)))
    forecast_col = "AdjClose"
    df_2['label'] = df_2[forecast_col].shift(-forecast_out)
    X = np.array(df_2.drop(['label'], 1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    y = np.array(df_2['label'])
    y = y[:-forecast_out]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print("creating MLflow project")
    experiment_id = mlflow.create_experiment(f"/Users/bclipp770@yandex.com/datalake/stocks/experiments/{uid}")
    experiment = mlflow.get_experiment(experiment_id)
    print("Name: {}".format(experiment.name))
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
    print("building our model")
    mlflow.sklearn.autolog(log_model_signatures=True, log_models=True)
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }
    # Create a based model
    rf = RandomForestRegressor()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rf,
                               param_grid=param_grid,
                               cv=3,
                               n_jobs=-1,
                               verbose=2)

    grid_search.fit(X_train, y_train)
    grid_search.best_params_
    best_grid = grid_search.best_estimator_


if __name__ == "__main__":
    main()
