# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis on CSO (Central Statistics Office) data for Republic of Ireland.
Visualizations using Pandas, SQLAlchemy, Matplotlib, Seaborn, and Plotly.
"""

import os
import sys
import pandas as pd
import sqlalchemy
import matplotlib
matplotlib.use('Agg')  # Headless backend for reliable server/terminal execution
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Create output folder for generated charts
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Database / Data Loading with SQLAlchemy
# ---------------------------------------------------------
pg_conn_str = os.environ.get('DATABASE_URL', 'postgresql://postgres:Password@localhost:5432/project')
sqlite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "project.db")
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pub_combined_view.csv")

df = None

# Attempt 1: Try PostgreSQL
try:
    engine = sqlalchemy.create_engine(pg_conn_str, connect_args={'connect_timeout': 2})
    with engine.connect() as conn:
        df = pd.read_sql_table("pub_combined_view", conn)
        print("Connected to PostgreSQL database successfully.")
except Exception:
    pass

# Attempt 2: Try SQLite
if df is None:
    if os.path.exists(sqlite_path):
        try:
            engine = sqlalchemy.create_engine(f"sqlite:///{sqlite_path}")
            df = pd.read_sql_table("pub_combined_view", engine)
            print(f"Connected to SQLite database ({sqlite_path}) successfully.")
        except Exception as e:
            print(f"Could not read from SQLite: {e}")

# Attempt 3: Try CSV or generate if missing
if df is None:
    if not os.path.exists(csv_path):
        print("Generating CSO dataset...")
        try:
            import generate_data
            df = generate_data.generate_cso_data()
        except Exception as e:
            print(f"Failed to generate dataset: {e}")
            sys.exit(1)
    else:
        df = pd.read_csv(csv_path)
        print(f"Loaded dataset from {csv_path}")

print(f"Dataset ready: {len(df)} rows across {df['county'].nunique()} counties.\n")
print(df.head())

# ---------------------------------------------------------
# Visualization 1: Stacked Bar Chart - Total economy per county
# ---------------------------------------------------------
print("\n[1/10] Generating Total Economy per County (Bar Chart)...")
plt.figure(figsize=(14, 7))
df3 = df.groupby(['county'])[['total_dependent', 'total_working']].sum()
ax = df3.plot(kind='bar', stacked=True, figsize=(14, 7), color=['#e74c3c', '#2ecc71'])
plt.title("Total Economy per County: Working vs Dependent Population", fontsize=14, pad=12)
plt.xlabel("County", fontsize=12)
plt.ylabel("Population Count", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
p1 = os.path.join(OUTPUT_DIR, "1_total_economy_per_county.png")
plt.savefig(p1, dpi=200)
plt.close('all')
print(f"   Saved: {p1}")

# ---------------------------------------------------------
# Visualization 2: Area Graph - Total contribution across industries
# ---------------------------------------------------------
print("[2/10] Generating Industry Contribution across Counties (Area Chart)...")
plt.figure(figsize=(14, 8))
industry_cols = [
    'total_agriculture_forestry_fishing',
    'total_building_construction',
    'total_manufacturing_industries',
    'total_commerce_trade',
    'total_transport_communications',
    'total_public_administration'
]
df2 = df.groupby(['county'])[industry_cols].sum()
# Focus on top 10 populated counties for clean presentation as shown in author's README
top_counties = df.groupby('county')['total_working'].sum().nlargest(10).index
df2_top = df2.loc[top_counties]
df2_top.plot(kind='area', stacked=True, figsize=(14, 8), alpha=0.85, colormap='Spectral')
plt.title("Total Contribution Across Industries for Top 10 Counties", fontsize=14, pad=12)
plt.xlabel("County", fontsize=12)
plt.ylabel("Employment Count", fontsize=12)
plt.xticks(range(len(df2_top.index)), df2_top.index, rotation=45, ha='right')
plt.legend(title="Industry Sector", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
p2 = os.path.join(OUTPUT_DIR, "2_industry_contribution_area.png")
plt.savefig(p2, dpi=200)
plt.close('all')
print(f"   Saved: {p2}")

# ---------------------------------------------------------
# Visualization 3: Line Chart - Manufacturing vs Field of Study
# ---------------------------------------------------------
print("[3/10] Generating Manufacturing vs Education (Line Chart)...")
plt.figure(figsize=(14, 7))
df_mfg = df.groupby(['county'])[['total_manufacturing_industries', 'persons_field_of_study_engineering_manufacturing_construction']].sum()
df_mfg.plot(kind='line', marker='o', figsize=(14, 7))
plt.title("Manufacturing Employment vs Engineering/Manufacturing Education by County", fontsize=14, pad=12)
plt.xlabel("County", fontsize=12)
plt.ylabel("Number of Persons", fontsize=12)
plt.xticks(range(len(df_mfg.index)), df_mfg.index, rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
p3 = os.path.join(OUTPUT_DIR, "3_manufacturing_vs_education_line.png")
plt.savefig(p3, dpi=200)
plt.close('all')
print(f"   Saved: {p3}")

# ---------------------------------------------------------
# Visualization 4: Scatter / Trend Line - Manufacturing vs Total Working
# ---------------------------------------------------------
print("[4/10] Generating Manufacturing vs Total Working Correlation...")
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='total_manufacturing_industries', y='total_working',
            scatter_kws={'alpha': 0.7, 'color': '#e74c3c'}, line_kws={'color': '#2c3e50'})
plt.title("Manufacturing Industries vs Total Working Population", fontsize=14, pad=12)
plt.xlabel("Total Manufacturing Industries", fontsize=12)
plt.ylabel("Total Working Population", fontsize=12)
plt.tight_layout()
p4 = os.path.join(OUTPUT_DIR, "4_manufacturing_vs_total_working.png")
plt.savefig(p4, dpi=200)
plt.close('all')
print(f"   Saved: {p4}")

# ---------------------------------------------------------
# Visualization 5: Exploded Pie Chart - Education & Females Working by NUTS III
# ---------------------------------------------------------
print("[5/10] Generating NUTS III Regional Distribution (Pie Charts)...")
df_nuts = df.groupby(['nuts_iii'])[['persons_field_of_study_engineering_manufacturing_construction', 'females_working']].sum().iloc[:10]
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
explode = [0.05] * len(df_nuts)
colors = sns.color_palette("Set2", len(df_nuts))

axes[0].pie(df_nuts['persons_field_of_study_engineering_manufacturing_construction'],
            labels=df_nuts.index, autopct='%1.1f%%', explode=explode, colors=colors, startangle=140)
axes[0].set_title("Engineering & Construction Study by NUTS III", fontsize=13)

axes[1].pie(df_nuts['females_working'],
            labels=df_nuts.index, autopct='%1.1f%%', explode=explode, colors=colors, startangle=140)
axes[1].set_title("Females Working by NUTS III", fontsize=13)

plt.suptitle("Regional Demographics Across NUTS III Regions", fontsize=16, y=1.02)
plt.tight_layout()
p5 = os.path.join(OUTPUT_DIR, "5_pie_chart_nuts_iii.png")
plt.savefig(p5, dpi=200, bbox_inches='tight')
plt.close('all')
print(f"   Saved: {p5}")

# ---------------------------------------------------------
# Visualization 6: Planning Region Comparison Bar Chart
# ---------------------------------------------------------
print("[6/10] Generating Planning Region Comparison (Bar Chart)...")
plt.figure(figsize=(10, 6))
df_plan = df.groupby(['planning_region'])[['persons_field_of_study_engineering_manufacturing_construction', 'females_working']].sum()
df_plan.plot(kind='bar', figsize=(10, 6), colormap='viridis')
plt.title("Planning Region: Education in Engineering/Mfg vs Females Working", fontsize=13, pad=12)
plt.xlabel("Planning Region", fontsize=12)
plt.ylabel("Number of Persons", fontsize=12)
plt.xticks(rotation=15, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
p6 = os.path.join(OUTPUT_DIR, "6_planning_region_bar.png")
plt.savefig(p6, dpi=200)
plt.close('all')
print(f"   Saved: {p6}")

# ---------------------------------------------------------
# Visualization 7: Violin Plot & Strip Plot - Planning Regions
# ---------------------------------------------------------
print("[7/10] Generating Violin & Strip Plots by Planning Region...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

sns.violinplot(ax=axes[0], x='planning_region', y='total_working', data=df, hue='planning_region', palette="Blues", legend=False)
axes[0].set_title("Violin Plot: Total Working by Planning Region", fontsize=13)
axes[0].tick_params(axis='x', rotation=15)

sns.stripplot(ax=axes[1], x='planning_region', y='persons_field_of_study_engineering_manufacturing_construction',
              data=df, hue='planning_region', palette="Set1", jitter=0.25, size=6, legend=False)
axes[1].set_title("Strip Plot: Degree/Study Level across Planning Regions", fontsize=13)
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
p7 = os.path.join(OUTPUT_DIR, "7_violin_and_strip_plot.png")
plt.savefig(p7, dpi=200)
plt.close('all')
print(f"   Saved: {p7}")

# ---------------------------------------------------------
# Visualization 8: Overlapping Bar Charts
# ---------------------------------------------------------
print("[8/10] Generating Demographic Overlapping Bar Charts...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Overlap 1: Working vs Males Working
top_working = df.groupby('county')[['total_working', 'males_working']].sum().sort_values(by='total_working', ascending=False).head(10)
top_working['total_working'].plot(kind='bar', ax=axes[0], color='#3498db', width=0.75, label='Total Working', alpha=0.8)
top_working['males_working'].plot(kind='bar', ax=axes[0], color='#e91e63', width=0.45, label='Males Working', alpha=0.95)
axes[0].set_title("Top 10 Counties: Total Working vs Males Working", fontsize=13)
axes[0].set_ylabel("Count")
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend()

# Overlap 2: Females Working vs Females Dependent
top_females = df.groupby('county')[['females_working', 'females_dependent']].sum().sort_values(by='females_working', ascending=False).head(10)
top_females['females_working'].plot(kind='bar', ax=axes[1], color='#9b59b6', width=0.75, label='Females Working', alpha=0.8)
top_females['females_dependent'].plot(kind='bar', ax=axes[1], color='#f39c12', width=0.45, label='Females Dependent', alpha=0.95)
axes[1].set_title("Top 10 Counties: Females Working vs Females Dependent", fontsize=13)
axes[1].set_ylabel("Count")
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend()

plt.tight_layout()
p8 = os.path.join(OUTPUT_DIR, "8_overlapping_demographics_bars.png")
plt.savefig(p8, dpi=200)
plt.close('all')
print(f"   Saved: {p8}")

# ---------------------------------------------------------
# Visualization 9: Correlation Heatmap
# ---------------------------------------------------------
print("[9/10] Generating Correlation Heatmap...")
plt.figure(figsize=(14, 11))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
cor = numeric_df.corr(method='pearson')
sns.heatmap(cor, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True, linewidths=0.5)
plt.title("Pearson Correlation Heatmap of CSO Demographic & Industry Metrics", fontsize=14, pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
p9 = os.path.join(OUTPUT_DIR, "9_correlation_heatmap.png")
plt.savefig(p9, dpi=200)
plt.close('all')
print(f"   Saved: {p9}")

# ---------------------------------------------------------
# Visualization 10: Interactive Regional Summary Dashboard
# ---------------------------------------------------------
print("[10/10] Generating Interactive Plotly Regional Visualization...")
county_agg = df.groupby(['county', 'nuts_iii', 'planning_region'], as_index=False)[
    ['total_working', 'total_dependent', 'females_working', 'total_manufacturing_industries']
].sum()

fig = px.treemap(
    county_agg,
    path=['planning_region', 'nuts_iii', 'county'],
    values='total_working',
    color='females_working',
    color_continuous_scale='RdBu_r',
    title="Interactive Treemap: Ireland Population & Workforce Structure (Planning Region > NUTS III > County)"
)
fig.update_layout(margin=dict(t=50, l=10, r=10, b=10))
p10 = os.path.join(OUTPUT_DIR, "10_interactive_treemap.html")
fig.write_html(p10)
print(f"   Saved: {p10}")

print("\n" + "="*60)
print(f" All 10 visualizations generated successfully in:")
print(f" {OUTPUT_DIR}")
print("="*60)
