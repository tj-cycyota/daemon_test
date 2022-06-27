def run_function():
  import mlflow.sklearn
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.metrics import mean_squared_error
  import pandas as pd
  from sklearn.model_selection import train_test_split
  
  df = pd.read_parquet("/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet")
  X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df["price"], random_state=42)
  X_train.head()
  
  experiment_id = "2827316969844731"

  with mlflow.start_run(experiment_id=experiment_id, run_name="Basic RF Run") as run:
      # Create model, train it, and create predictions
      rf = RandomForestRegressor(random_state=42)
      rf.fit(X_train, y_train)
      predictions = rf.predict(X_test)

      # Log model
      mlflow.sklearn.log_model(rf, "random_forest_model")

      # Log metrics
      mse = mean_squared_error(y_test, predictions)
      mlflow.log_metric("mse", mse)

      run_id = run.info.run_id
      experiment_id = run.info.experiment_id

      print(f"Inside MLflow Run with run_id `{run_id}` and experiment_id `{experiment_id}`")
    