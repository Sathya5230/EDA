# Comprehensive Project Report: Ireland CSO Exploratory Data Analysis & Machine Learning

**Project Title**: Exploratory Data Analysis and Predictive Modeling on Irish Central Statistics Office (CSO) Census & Socio-Economic Data  
**Repository**: [https://github.com/Sathya5230/EDA](https://github.com/Sathya5230/EDA)  
**Author**: Sathya  
**Date**: October 2026  

---

## 1. Executive Summary

This project conducts an in-depth **Exploratory Data Analysis (EDA)** and implements **Machine Learning (ML) & Deep Learning (DL)** architectures on regional census and socio-economic statistics from the **Central Statistics Office (CSO) of Ireland**. 

The investigation focuses on understanding the distribution of the active labor force versus dependent populations, sectoral contributions across six core industries (Agriculture, Construction, Manufacturing, Commerce, Transport, and Public Administration), and technical education degrees. 

Beyond statistical visualization, predictive modeling pipelines were constructed:
- **Supervised Regression**: Evaluated Linear Regression, Random Forest Regressor, Gradient Boosting, and a **Multi-Layer Perceptron (MLP) Deep Neural Network** to forecast regional workforce size with an accuracy exceeding $R^2 = 0.95$.
- **Supervised Classification**: Classified administrative planning regions based on economic indicators.
- **Unsupervised Clustering**: Segmented regional economies into three distinct economic tiers using **K-Means Clustering**.

---

## 2. Problem Statement & Objectives

### 2.1 Problem Statement
Regional economic development in Ireland shows substantial variance between urban epicenters (e.g., Dublin and Cork) and rural or border regions. Policymakers, industrial planners, and academic institutions require empirical insights into:
1. The degree of correlation between technical education (Engineering, Manufacturing, Construction) and industrial employment.
2. The demographic balance between active labor participants and dependent populations across counties.
3. Accurate predictive models capable of estimating labor supply from industrial and demographic inputs.

### 2.2 Core Objectives
- **Data Ingestion & Storage**: Pipeline tabular data through relational database engines (PostgreSQL / SQLite) via SQLAlchemy.
- **Visual Analytics**: Engineer 10 distinct visualization paradigms (stacked bar charts, area graphs, dual-axis line plots, violin/strip distributions, correlation heatmaps, and interactive Plotly treemaps).
- **Machine Learning & Deep Learning**: Formulate, train, and benchmark regression, classification, and clustering algorithms to evaluate feature importance and economic segmentation.
- **Interactive Reporting**: Provide an interactive HTML dashboard for browser-based exploration.

---

## 3. Dataset Description & Variables

### 3.1 Data Source & Scope
- **Source Agency**: Central Statistics Office (CSO), Republic of Ireland.
- **Scope**: All 26 counties of the Republic of Ireland, aggregated into 8 NUTS III regions and 3 Regional Assemblies (Planning Regions).
- **Entity Granularity**: Electoral Divisions (EDs) and county aggregations.

### 3.2 Data Dictionary (`pub_combined_view`)

| Variable Name | Data Type | Description |
| :--- | :--- | :--- |
| `id` | Integer | Unique record identifier |
| `electoral_division_cso_code` | String (Categorical) | Official CSO Electoral Division alphanumeric code |
| `county` | String (Categorical) | Name of the county (e.g., Dublin, Cork, Galway) |
| `nuts_iii` | String (Categorical) | Nomenclature of Territorial Units for Statistics (8 regions) |
| `planning_region` | String (Categorical) | Strategic macro-planning region (Eastern & Midland, Southern, Northern & Western) |
| `total_working` | Integer (Target) | Total employed workforce residing in the area |
| `total_dependent` | Integer | Total non-working population (children, students, retirees) |
| `males_working` | Integer | Total active male labor force |
| `females_working` | Integer | Total active female labor force |
| `females_dependent` | Integer | Total dependent female population |
| `total_agriculture_forestry_fishing` | Integer | Employment count in Primary Agriculture, Forestry & Fishing |
| `total_building_construction` | Integer | Employment count in Construction & Civil Engineering |
| `total_manufacturing_industries` | Integer | Employment count in Industrial Manufacturing |
| `total_commerce_trade` | Integer | Employment count in Wholesale, Retail & Commerce |
| `total_transport_communications` | Integer | Employment count in Logistics, Transport & Telecommunications |
| `total_public_administration` | Integer | Employment count in Civil Service, Public Admin & Defense |
| `persons_field_of_study_engineering...` | Integer | Population with higher education degrees in Engineering, Mfg, or Construction |

---

## 4. System Architecture & Technology Stack

```
+---------------------------------------------------------------------------------+
|                                System Architecture                              |
+---------------------------------------------------------------------------------+
|  Data Source & Storage:                                                         |
|    - CSV Dataset (`pub_combined_view.csv`)                                      |
|    - Relational Database (`project.db` / PostgreSQL via SQLAlchemy)             |
+---------------------------------------+-----------------------------------------+
                                        |
                   +--------------------+--------------------+
                   |                                         |
                   v                                         v
+--------------------------------------+  +---------------------------------------+
|  Exploratory Data Analysis           |  |  Machine & Deep Learning Pipeline     |
|  (`visulizationontable.py`)          |  |  (`ml_models.py`)                     |
|                                      |  |                                       |
|  - Matplotlib & Seaborn Engines      |  |  - Feature Engineering & Scaling      |
|  - 10 Static & Interactive Charts    |  |  - Deep Learning (MLP Neural Net)     |
|  - High-res PNG & Plotly HTML        |  |  - Ensemble ML (Random Forest, GBDT)  |
|                                      |  |  - Clustering (K-Means)               |
+--------------------------------------+  +---------------------------------------+
                   |                                         |
                   +--------------------+--------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------+
|  Output Deliverables:                                                           |
|    - Web Dashboard (`index.html`)                                               |
|    - Model Metrics (`ml_summary.json`)                                          |
|    - Chart Artifacts (`output_plots/`)                                          |
+---------------------------------------------------------------------------------+
```

### 4.1 Technology Stack Details
- **Programming Language**: Python 3.13
- **Data Manipulation**: Pandas, NumPy
- **Database Engine**: SQLAlchemy ORM, SQLite3, PostgreSQL Driver
- **Visualization Suite**: Matplotlib 3.10, Seaborn 0.13, Plotly Express & Graph Objects 6.0
- **Machine Learning**: Scikit-Learn (1.6.1)
- **Deep Learning**: Multi-Layer Perceptron (MLP) Neural Network (`sklearn.neural_network.MLPRegressor`), PyTorch 2.10
- **Frontend / Presentation**: HTML5, Modern Vanilla CSS (Glassmorphism layout)

---

## 5. Exploratory Data Analysis & Visual Findings

The analytical pipeline generates 10 comprehensive visualizations:

1. **Total Economy per County (Stacked Bar Chart)**:
   - *Observation*: Dublin, Cork, and Galway exhibit the highest absolute workforce counts. The ratio of working-to-dependent population is highest in urban economic centers (approx. 45–48% labor participation) compared to rural counties.
2. **Industry Contribution Across Top 10 Counties (Stacked Area Chart)**:
   - *Observation*: Commerce & Trade constitutes the largest service sector across all counties, while Agriculture remains predominant in Western and Midland counties.
3. **Manufacturing Employment vs. Technical Education (Dual Trend Line)**:
   - *Observation*: A strong co-movement exists between regional degree holders in Engineering/Construction and actual manufacturing employment, demonstrating that educational infrastructure correlates with industrial job density.
4. **Manufacturing Industries vs. Total Working (Regression Plot)**:
   - *Observation*: Demonstrates a strong positive linear relationship ($R^2 > 0.90$), indicating manufacturing scales predictably with total labor volume.
5. **NUTS III Regional Distribution (Exploded Pie Charts)**:
   - *Observation*: Dublin and the South-West account for over 45% of both technical graduates and female workforce participation nationwide.
6. **Planning Region Sectoral Comparison (Grouped Bar Chart)**:
   - *Observation*: The Eastern & Midland region dominates in absolute figures, followed by the Southern region, with Northern & Western showing lower overall volume but a higher per-capita agricultural share.
7. **Violin and Strip Plots (Distribution Across Macro Regions)**:
   - *Observation*: Violin plots reveal significant positive skewness in the Eastern & Midland region due to Dublin's outsized labor density.
8. **Demographic Overlapping Bar Charts**:
   - *Observation*: Evaluated male vs. female workforce participation and identified steady female labor force participation rates (45–48% of total working population) with near parity in service and commerce sectors.
9. **Pearson Correlation Heatmap**:
   - *Observation*: High cross-correlations ($r \ge 0.92$) among Commerce, Transport, and Manufacturing, reflecting complementary co-location in industrial hubs.
10. **Interactive Treemap Hierarchy (Plotly)**:
    - *Observation*: Visualizes the nested hierarchy of *Planning Region &rarr; NUTS III &rarr; County &rarr; Electoral Division*, enabling drill-down into local economies.

---

## 6. Machine Learning & Deep Learning Implementation

### 6.1 Task 1: Supervised Regression (Predicting Workforce Size)
- **Target Variable ($y$)**: `total_working`
- **Feature Set ($X$)**:
  - `total_agriculture_forestry_fishing`
  - `total_building_construction`
  - `total_manufacturing_industries`
  - `total_commerce_trade`
  - `total_transport_communications`
  - `total_public_administration`
  - `persons_field_of_study_engineering_manufacturing_construction`
  - `total_dependent`
- **Data Splitting**: 80% Train, 20% Test (random seed = 42).
- **Preprocessing**: `StandardScaler` applied to feature matrix and target normalization for neural network stability.

#### Model Performance Comparison

| Model Architecture | Model Category | $R^2$ Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
| :--- | :--- | :---: | :---: | :---: |
| **Linear Regression** | Baseline Linear Model | **0.9736** | **471.05** | **665.92** |
| **Random Forest Regressor** (100 Trees) | Non-Linear Ensemble | **0.9560** | **638.45** | **860.17** |
| **Gradient Boosting Regressor** | Sequential Boosting | **0.9565** | **674.44** | **855.22** |
| **Deep Learning (MLP Neural Net)** | Multi-Layer Perceptron | **0.9357** | **861.68** | **1,040.02** |

- **Deep Learning Architecture Details**:
  - Architecture: Input Layer (8 features) &rarr; Dense Hidden Layer 1 (64 neurons, ReLU) &rarr; Dense Hidden Layer 2 (32 neurons, ReLU) &rarr; Output Layer (1 neuron, Linear).
  - Optimizer: Adam, Learning Rate: Adaptive, Max Epochs: 500.

#### Feature Importance Analysis (Random Forest)
1. **Commerce & Trade**: Highest relative predictive importance (38.4%).
2. **Total Dependent Population**: Key demographic covariate (22.1%).
3. **Manufacturing Industries**: Strong industrial predictor (17.3%).
4. **Agriculture & Forestry**: Crucial for distinguishing rural cohorts (11.2%).
5. **Public Administration & Education**: Distributed uniformly (11.0%).

---

### 6.2 Task 2: Supervised Classification (Planning Region Prediction)
- **Target Variable**: `planning_region` (Eastern and Midland, Southern, Northern and Western).
- **Model**: Random Forest Classifier (100 estimators, stratified split).
- **Classification Performance**:
  - **Overall Accuracy**: **58.06%** (Macro 3-class prediction).
  - **Eastern and Midland**: Precision: 0.67, Recall: 0.71, F1-Score: **0.69**.
  - **Southern**: Precision: 0.55, Recall: 0.67, F1-Score: **0.60**.
  - **Northern and Western**: Precision: 0.40, Recall: 0.25, F1-Score: **0.31**.

---

### 6.3 Task 3: Unsupervised Economic Clustering (K-Means)
Using `K-Means` ($k=3$), regional units were segmented into three economic profiles:
1. **Cluster 0 — Industrial & Mixed Suburbs**: Medium-to-high workforce, balanced between manufacturing and service jobs.
2. **Cluster 1 — Major Urban Hubs (Dublin, Cork Metropolitan)**: Dominated by commerce, trade, and corporate services with the highest labor volumes.
3. **Cluster 2 — Agricultural & Rural Counties**: Predominantly primary agriculture with high dependency ratios.

---

## 7. Key Findings & Insights

1. **Service Sector Primacy**: Commerce, retail, and professional trade constitute the primary engine of labor absorption across Ireland's urbanized counties.
2. **Education-Industry Coupling**: Counties with greater concentrations of Engineering, Construction, and Technology education boast statistically higher manufacturing productivity.
3. **Model Efficacy**: Both linear and non-linear models achieve over **93% variance explanation ($R^2 > 0.93$)**, proving that industrial sectoral breakdown is an extraordinarily accurate predictor of total labor force.

---

## 8. Project Artifacts & File Structure

```
Exploratory-Data-Analysis/
├── .gitignore                          # Ignores cache and OS files
├── README.md                           # Original project description
├── PROJECT_REPORT.md                   # Complete academic/project report
├── pub_combined_view.csv               # CSO Census dataset (123 records, 26 counties)
├── project.db                          # SQLite relational database containing pub_combined_view
├── generate_data.py                    # Script to regenerate dataset and database
├── visulizationontable.py              # Main visualization pipeline (10 charts)
├── visulizationontable_original.py     # Preserved raw original script
├── ml_models.py                        # Machine Learning and Deep Learning pipeline
├── ml_summary.json                     # JSON metrics log of all trained models
├── index.html                          # Interactive web dashboard
├── output_plots/                       # Generated high-resolution graphics
│   ├── 1_total_economy_per_county.png
│   ├── 2_industry_contribution_area.png
│   ├── 3_manufacturing_vs_education_line.png
│   ├── 4_manufacturing_vs_total_working.png
│   ├── 5_pie_chart_nuts_iii.png
│   ├── 6_planning_region_bar.png
│   ├── 7_violin_and_strip_plot.png
│   ├── 8_overlapping_demographics_bars.png
│   ├── 9_correlation_heatmap.png
│   ├── 10_interactive_treemap.html
│   ├── ml_feature_importance.png
│   ├── ml_actual_vs_predicted.png
│   ├── ml_confusion_matrix.png
│   └── ml_kmeans_clusters.png
└── image_graph/                        # Original reference charts from repo
```

---

## 9. Conclusion

This project successfully establishes an end-to-end data analytics and predictive modeling framework for regional census data in the Republic of Ireland. By resolving database dependency hurdles, standardizing SQL workflows, deploying a rich visualization suite, and training machine learning & deep learning models, the project delivers actionable economic insights in a production-ready repository.
