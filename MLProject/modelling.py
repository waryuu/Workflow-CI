import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def run_tuning_modelling():
    PATH_DATA = 'premier_league_complete_stats_until31thGameDayOnSeason2025-26_preprocessing.csv'
    
    # 1. Validasi dataset
    if not os.path.exists(PATH_DATA):
        print(f"[ERROR] Data tidak ditemukan: {PATH_DATA}")
        return
    
    # 2. Load Data
    df = pd.read_csv(PATH_DATA)
    X = df.drop(columns=['target_rating'])
    y = df['target_rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Setup Tracking
    # dagshub.init(repo_owner='waryuu', repo_name='Eksperimen_SML_Wahyudi-Putra', mlflow=True)
    # mlflow.set_experiment("Premier_League_Tuning_Experiment")

    # 4. Training dengan Hyperparameter Tuning
    with mlflow.start_run(run_name="RF_Manual_Tuning_Final"):
        rf = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
        }
        
        # Cross-validation
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # 5. Manual Logging Metrics & Params
        val_mse = mean_squared_error(y_test, y_pred)
        val_r2 = r2_score(y_test, y_pred)

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("mse", val_mse)
        mlflow.log_metric("r2_score", val_r2)
        
        # Log Model
        mlflow.sklearn.log_model(best_model, "best_random_forest_model")

        # 6. LOG 4 ARTEFAK TAMBAHAN

        # ARTEFAK 1: Metric Info (JSON)
        metric_info = {
            "best_params": grid_search.best_params_,
            "final_metrics": {"mse": val_mse, "r2": val_r2},
            "total_data": len(df)
        }
        with open("metric_info.json", "w") as f:
            json.dump(metric_info, f, indent=4)
        mlflow.log_artifact("metric_info.json")

        # ARTEFAK 2: Actual vs Predicted (Scatter Plot)
        plt.figure(figsize=(7,7))
        plt.scatter(y_test, y_pred, color='green', alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title('Actual vs Predicted Rating')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.savefig("actual_vs_predicted.png")
        mlflow.log_artifact("actual_vs_predicted.png")

        # ARTEFAK 3: Feature Importance
        plt.figure(figsize=(10, 6))
        feat_importances = pd.Series(best_model.feature_importances_, index=X.columns)
        feat_importances.nlargest(10).plot(kind='barh', color='skyblue')
        plt.title('Top 10 Statistik Penentu Rating')
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")

        # ARTEFAK 4: Data Preview (CSV)
        df.head(10).to_csv("data_preview.csv", index=False)
        mlflow.log_artifact("data_preview.csv")
        
        # Bersihkan file lokal setelah di-upload ke MLflow
        for f in ["feature_importance.png", "data_preview.csv", "actual_vs_predicted.png", "metric_info.json"]:
            if os.path.exists(f): os.remove(f)
        
        print(f"Tuning selesai dengan total 4 Artefak. R2 Score: {val_r2:.4f}")

if __name__ == "__main__":
    run_tuning_modelling()