from prefect import flow, task
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

@task()
def data_loader(path):
    dataframe = pd.read_parquet(path)
    return dataframe

@task()
def data_preprocessor(data):
    data['duration'] = data['lpep_dropoff_datetime'] - data['lpep_pickup_datetime']
    data.duration = data.duration.apply(lambda td: td.total_seconds()/60)
    data = data[(data.duration >= 3.) & (data.duration <= 120.)] 
    data.fillna(0, inplace=True) #maybe debug this later

    num_features = ['total_amount', 'trip_distance', 'passenger_count']
    cat_features = ['PULocationID', 'DOLocationID']
    features = data[num_features + cat_features]
    label = data['duration']
    return features, label   

@task()
def model_traning(features, label):
    model = RandomForestRegressor(random_state=42, max_depth=5, n_estimators=50, n_jobs=-1)   
    model.fit(features, label)
    return model

@task()
def model_estimator(model, features):
    preds = model.predict(features)
    return preds

@task(name="RMSE", retries=2, retry_delay_seconds=10, log_prints=True)
def health_check(preds, label):
    rmse = root_mean_squared_error(preds, label)
    print(f"rmse for hold-out dataset: {rmse}")
    return rmse

@flow(name="NYC taxi route prediction model retraining")
def retrain_flow():
    train = data_loader("data/green_tripdata_2025-06.parquet")
    test = data_loader("data/green_tripdata_2025-07.parquet")

    X_train, y_train = data_preprocessor(train)
    X_test, y_test = data_preprocessor(test)

    model = model_traning(X_train, y_train)
    
    test_preds = model_estimator(model, X_test)
    rmse = health_check(test_preds, y_test)

# run the flow!
if __name__ == "__main__":
    retrain_flow()