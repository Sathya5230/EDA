# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 18:38:39 2019

@author: nandita
"""

import pandas as pd
import sqlalchemy
from matplotlib import pyplot as plt
import seaborn as sns
ax = plt.gca()

# Connect Postgres with Python for visualization
engine = sqlalchemy.create_engine('postgresql://postgres:Password@localhost:5432/project')
df = pd.read_sql_table("pub_combined_view", engine, index_col = 1)
df.head()

#1
df3 = df.groupby(['county'])['total_dependent','total_working'].sum()
df3.plot(kind='bar',stacked=True,figsize=(13,7), title="Total economy per county")

#2
df2=df.groupby(['county'])['total_agriculture_forestry_fishing','total_building_construction','total_manufacturing_industries', 'total_manufacturing_industries','total_commerce_trade','total_transport_communications','total_public_administration'].sum()
df2.plot(kind='area', stacked=True, figsize=(10,12))
plt.show()
plt.savefig('df2.png', dpi=300)

#3
df2=df.groupby(['county'])['total_manufacturing_industries', 'persons_field_of_study_engineering_manufacturing_construction'].sum()
df2.plot(kind='line', figsize=(10,12))
plt.show()

#4
df.plot(kind='line',x='total_manufacturing_industries',y='total_working',color='red')
plt.show()


#5
df2=df.groupby(['nuts_iii'])['persons_field_of_study_engineering_manufacturing_construction','females_working'].sum().iloc[:10]
df2.plot(kind='pie', figsize=(13,16), subplots=True, legend=False, autopct='%1.1f%%',title="", fontsize=15, )
plt.show()

#6
    df2=df.groupby(['planning_region'])['persons_field_of_study_engineering_manufacturing_construction','females_working'].sum().iloc[:10]
    df2.plot(kind='bar', figsize=(7,5), legend=True,title="", fontsize=15 )
    plt.show()

#7
plt.figure(figsize=(10,12)) 
ax = sns.violinplot(x=df.planning_region, y=df.total_working, hue=df.planning_region, data=df, palette="Blues")

#try with cloumn value select
plt.figure(figsize=(10,12)) 
ax = sns.violinplot(x=.value_counts()[:20], y=df.total_working, data=df,hue=df.planning_region, palette="Blues")

#8Overlap bar gharp
plt.figure(figsize=(14,16)) 
Q=df['total_working'].value_counts().head(10).plot(kind='bar', color='blue', width=.75, legend=True, alpha=0.8)
df['males_working'].value_counts().head(10).plot(kind='bar', color='pink', width=.5, alpha=1, legend=True)

#geolocation
import plotly.figure_factory as ff
import geopandas
df_sample_r = df['county']
values = df['nuts_iii'].tolist()
fips = df['electoral_division_cso_code'].tolist()
colorscale = [
    'rgb(193, 193, 193)',
    'rgb(239,239,239)',
    'rgb(195, 196, 222)',
    'rgb(144,148,194)',
    'rgb(101,104,168)',
    'rgb(65, 53, 132)'
]
fig = ff.create_choropleth(
    fips=fips, colorscale=colorscale, values=values,
    county_outline={'color': 'rgb(255,255,255)', 'width': 0.5}, round_legend_values=True,
    legend_title='Population by County', title='Country Ireland'
)
fig.layout.template = None
fig.show()

conda install geopandas


import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs
data= dict(type='choropleth',colorscale='YIOrd', locations=df['county'],z=df['females_working'],
           locationmode='USA-states',text=df['Planning Region'],marker=dic(line=dict(color='rgb(255,255,255)'),
           colorbar={'title':"Millions USD"})
           
layout = dict(title='country ireland',
            geo = dict(scope='USA'))

choromap=go.Figure(data=[data],layout=layout)
iplot(choromap)

import plotly.graph_objects as go
fig = go.Figure(data=go.Choropleth(
    locations=df['county'], # Spatial coordinates
    z = df['females_working'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "Millions USD",
))

fig.update_layout(
    title_text = 'ireland country working',
    geo_scope='usa', # limite map scope to USA
)

fig.show()

#heatmap
plt.figure(figsize=(20,25))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#correlation
df.corr(method ='pearson') 

#
plt.figure(figsize=(10,12)) 
Q=df['females_working'].value_counts().head(10).plot(kind='bar', color='blue', width=.75, legend=True, alpha=0.8, x='females_working',rot=0)
df['females_dependent'].value_counts().head(10).plot(kind='bar', color='pink', width=.5, alpha=1, legend=True, y='females_dependent',rot=0)



