import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

experiment_name = "Premier_League_Basic_Modelling"
mlflow.set_experiment(experiment_name)

def run_basic_modelling():
    # 1. Load Data (Gunakan path folder sesuai kriteria)
    PATH_DATA = 'premier_league_complete_stats_until31thGameDayOnSeason2025-26_preprocessing.csv'
    df = pd.read_csv(PATH_DATA)
    X = df.drop(columns=['target_rating'])
    y = df['target_rating']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Aktifkan Autolog
    mlflow.sklearn.autolog()

    # 3. Jalankan Eksperimen
    with mlflow.start_run(run_name="RF_Basic_Autolog"):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        experiment = mlflow.get_experiment_by_name(experiment_name)
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
        latest_run_id = runs.iloc[0]["run_id"]

        print("Model Basic berhasil dilatih")

if __name__ == "__main__":
    run_basic_modelling()
