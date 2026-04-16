import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Inisialisasi DagsHub
dagshub.init(repo_owner='waryuu', repo_name='Eksperimen_SML_Wahyudi-Putra', mlflow=True)


def run_tuning_modelling():
    PATH_DATA = 'premier_league_complete_stats_until31thGameDayOnSeason2025-26_preprocessing.csv'
    df = pd.read_csv(PATH_DATA)
    X = df.drop(columns=['target_rating'])
    y = df['target_rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("Premier_League_Tuning_Experiment")

    with mlflow.start_run(run_name="RF_Manual_Tuning"):
        rf = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
        }
        
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # 4. Manual Logging
        val_mse = mean_squared_error(y_test, y_pred)
        val_r2 = r2_score(y_test, y_pred)


        # ARTEFAK TAMBAHAN

        # Log Params
        mlflow.log_params(grid_search.best_params_)

        # Log Metrics
        mlflow.log_metric("mse", val_mse)
        mlflow.log_metric("r2_score", val_r2)
        
        # Log Model
        mlflow.sklearn.log_model(best_model, "best_random_forest_model")

      
        # Artefak Tambahan 1: Feature Importance
        plt.figure(figsize=(10, 6))
        feat_importances = pd.Series(best_model.feature_importances_, index=X.columns)
        feat_importances.nlargest(10).plot(kind='barh', color='skyblue')
        plt.title('Top 10 Statistik Penentu Rating')
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")

        # Artefak Tambahan 2: Data Preview
        df.head(10).to_csv("data_preview.csv", index=False)
        mlflow.log_artifact("data_preview.csv")
        
        print(f"Tuning selesai. R2 Score Terbaik: {val_r2:.4f}")

if __name__ == "__main__":
    run_tuning_modelling()