#!/usr/bin/env python3
"""
Machine Learning and Deep Learning Models for Irish CSO Census & Economic Dataset.

Tasks:
1. Supervised Regression: Predict 'total_working' workforce from industrial and educational metrics.
   - Models: Linear Regression, Random Forest Regressor, Gradient Boosting, Multi-Layer Perceptron (Neural Network / Deep Learning)
2. Supervised Classification: Predict 'planning_region' based on economic & industry distribution.
   - Models: Random Forest Classifier, Logistic Regression
3. Unsupervised Clustering: K-Means Clustering for Economic Profiling
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor  # Deep Learning (Multi-Layer Perceptron)
from sklearn.metrics import (
    r2_score, mean_absolute_error, root_mean_squared_error,
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.cluster import KMeans

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(BASE_DIR, "pub_combined_view.csv")


def load_dataset():
    if not os.path.exists(CSV_PATH):
        import generate_data
        generate_data.generate_cso_data()
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.\n")
    return df


def run_regression_models(df):
    print("=" * 60)
    print(" TASK 1: SUPERVISED REGRESSION (Predict 'total_working')")
    print("=" * 60)

    feature_cols = [
        'total_agriculture_forestry_fishing',
        'total_building_construction',
        'total_manufacturing_industries',
        'total_commerce_trade',
        'total_transport_communications',
        'total_public_administration',
        'persons_field_of_study_engineering_manufacturing_construction',
        'total_dependent'
    ]
    target_col = 'total_working'

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standard scaling for features and target (for neural network stability)
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    # Models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Deep Learning (MLP Neural Network)": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
    }

    results = {}
    predictions = {}

    for name, model in models.items():
        if "Deep Learning" in name:
            model.fit(X_train_scaled, y_train_scaled)
            y_pred_scaled = model.predict(X_test_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        elif "Linear" in name:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        results[name] = {"R2_Score": round(float(r2), 4), "MAE": round(float(mae), 2), "RMSE": round(float(rmse), 2)}
        predictions[name] = y_pred

        print(f"[{name}]")
        print(f"   R² Score : {r2:.4f}")
        print(f"   MAE      : {mae:.2f}")
        print(f"   RMSE     : {rmse:.2f}\n")

    # Plot 1: Feature Importance
    rf_model = models["Random Forest Regressor"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    clean_feature_names = [f.replace('total_', '').replace('_', ' ').title() for f in feature_cols]
    sorted_features = [clean_feature_names[i] for i in indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=importances[indices],
        y=sorted_features,
        hue=sorted_features,
        palette="viridis",
        legend=False
    )
    plt.title("Feature Importance in Predicting Total Workforce (Random Forest)", fontsize=13, pad=12)
    plt.xlabel("Relative Importance Score", fontsize=11)
    plt.tight_layout()
    fi_path = os.path.join(OUTPUT_DIR, "ml_feature_importance.png")
    plt.savefig(fi_path, dpi=200)
    plt.close()
    print(f"Saved: {fi_path}")

    # Plot 2: Actual vs Predicted
    plt.figure(figsize=(9, 6))
    best_pred = predictions["Random Forest Regressor"]
    plt.scatter(y_test, best_pred, color="#3498db", edgecolors="#2980b9", s=60, alpha=0.85, label="Predictions")
    min_val = min(y_test.min(), best_pred.min())
    max_val = max(y_test.max(), best_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Fit (y = x)")
    plt.title("Actual vs. Predicted Working Population (Random Forest Regressor)", fontsize=13, pad=12)
    plt.xlabel("Actual Working Population", fontsize=11)
    plt.ylabel("Predicted Working Population", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    pred_path = os.path.join(OUTPUT_DIR, "ml_actual_vs_predicted.png")
    plt.savefig(pred_path, dpi=200)
    plt.close()
    print(f"Saved: {pred_path}\n")

    return results


def run_classification_models(df):
    print("=" * 60)
    print(" TASK 2: SUPERVISED CLASSIFICATION (Predict 'planning_region')")
    print("=" * 60)

    feature_cols = [
        'total_working',
        'total_dependent',
        'total_agriculture_forestry_fishing',
        'total_building_construction',
        'total_manufacturing_industries',
        'total_commerce_trade',
        'total_transport_communications',
        'total_public_administration',
        'persons_field_of_study_engineering_manufacturing_construction'
    ]
    target_col = 'planning_region'

    X = df[feature_cols]
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Random Forest Classifier Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix: Planning Region Classification", fontsize=13, pad=12)
    plt.xlabel("Predicted Region", fontsize=11)
    plt.ylabel("Actual Region", fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "ml_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved: {cm_path}\n")

    return {"Random_Forest_Classifier_Accuracy": round(float(acc), 4)}


def run_clustering(df):
    print("=" * 60)
    print(" TASK 3: UNSUPERVISED CLUSTERING (K-Means Economic Profiling)")
    print("=" * 60)

    cluster_features = [
        'total_working',
        'total_manufacturing_industries',
        'total_commerce_trade',
        'total_agriculture_forestry_fishing'
    ]
    X = df[cluster_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['economic_cluster'] = kmeans.fit_predict(X_scaled)
    cluster_labels = {0: "Industrial & Mixed", 1: "Major Urban Hubs", 2: "Agricultural & Rural"}
    df['cluster_name'] = df['economic_cluster'].map(cluster_labels)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='total_commerce_trade',
        y='total_manufacturing_industries',
        hue='cluster_name',
        palette='Set1',
        s=85,
        alpha=0.9
    )
    plt.title("K-Means Economic Clustering (Commerce vs Manufacturing)", fontsize=13, pad=12)
    plt.xlabel("Commerce & Trade Employment", fontsize=11)
    plt.ylabel("Manufacturing Employment", fontsize=11)
    plt.legend(title="Economic Cluster")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    km_path = os.path.join(OUTPUT_DIR, "ml_kmeans_clusters.png")
    plt.savefig(km_path, dpi=200)
    plt.close()
    print(f"Saved: {km_path}\n")


def main():
    df = load_dataset()
    reg_results = run_regression_models(df)
    clf_results = run_classification_models(df)
    run_clustering(df)

    summary = {
        "Supervised_Regression": reg_results,
        "Supervised_Classification": clf_results,
        "Unsupervised_Clustering": "K-Means (k=3)"
    }

    summary_file = os.path.join(BASE_DIR, "ml_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved model summary metrics to: {summary_file}")
    print("\nMachine Learning pipeline completed successfully!")


if __name__ == "__main__":
    main()
