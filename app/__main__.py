import math as math
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import mlflow
import mlflow.sklearn
import time
import uuid


def objective(hypers):
    regr = RandomForestRegressor(max_depth=hypers["max_depth"],
                                 max_features=hypers["max_features"],
                                 min_samples_leaf=hypers["min_samples_leaf"],
                                 min_samples_split=hypers["min_samples_split"],
                                 n_estimators=hypers["n_estimators"]
                                 )
    accuracy = cross_val_score(regr, X_train, y_train).mean()
    return {'loss': -accuracy, 'status': STATUS_OK}


def main():
    spark = ..
    now = datetime.now()
    timestamp = now.strftime("%m%d%Y%H%M"
    uid = str(uuid.uuid1()).replace('-', '')
    df = spark.read.format("delta").load((f"/dbfs/datalake/strocks_{uid}_{timestamp}/data"))
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
    experiment_id = mlflow.create_experiment(f"/Users/bclipp770@yandex.com/stocks_{id}_{timestamp}-search")
    experiment = mlflow.get_experiment(experiment_id)
    search_space = {
        'max_depth': hp.choice('max_depth', range(1, 110)),
        'max_features': hp.choice('max_features', np.arange(0.1, 1.0, 0.1)),
        "min_samples_leaf": hp.choice('min_samples_leaf', range(3, 5)),
        "min_samples_split": hp.choice("min_samples_split", range(8, 12)),
        'n_estimators': hp.choice('n_estimators', range(100, 500))}

    trials = Trials()
    with mlflow.start_run():
        argmin = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=16,
            trials=trials)


if __name__ == "__main__":
    main()
