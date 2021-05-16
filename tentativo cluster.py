# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:45:13 2021

@author: pierl
"""

# Import Libraries and csv files

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


hol=pd.read_csv('holiday_17_18_19.csv')
ist_ene_17=pd.read_csv('IST_South_Tower_2017_Ene_Cons.csv')
ist_ene_18=pd.read_csv('IST_South_Tower_2018_Ene_Cons.csv')
ist_meteo=pd.read_csv('IST_meteo_data_2017_2018_2019.csv')

# Data preparation 

#### Set the date 

ist_meteo=ist_meteo.rename(columns={'yyyy-mm-dd hh:mm:ss':'Date'})
ist_ene_17=ist_ene_17.rename(columns={'Date_start':'Date'})
ist_ene_18=ist_ene_18.rename(columns={'Date_start':'Date'})

ist_meteo_1=ist_meteo[~ist_meteo.Date.str.contains('2019')]
ist_meteo_1.index = np.arange(0, len(ist_meteo_1))

ist_ene_17['Date'] = pd.to_datetime(ist_ene_17['Date'])
ist_ene_18['Date'] = pd.to_datetime(ist_ene_18['Date'])
ist_meteo_1['Date'] = pd.to_datetime(ist_meteo_1['Date'])

ist_ene_1=pd.concat([ist_ene_17,ist_ene_18])

ist_ene_1 = ist_ene_1.set_index ('Date', drop = True)
ist_meteo_1 = ist_meteo_1.set_index ('Date', drop = True).resample('H').mean().bfill()

ist_ene_1.index = pd.to_datetime(ist_ene_1.index, format='%y/%m/%d %H:%M:%S').strftime('%d/%m/%y %H:%M:%S')
ist_meteo_1.index = pd.to_datetime(ist_meteo_1.index, format='%m/%d/%y %H:%M:%S').strftime('%d/%m/%y %H:%M:%S')

# Merge the data into one csv file

tot=pd.merge(ist_meteo_1,ist_ene_1,on='Date')
tot['day'] = pd.to_datetime(tot.index,format='%d/%m/%y %H:%M:%S').weekday

fig1=px.line(x=tot.index, y=tot['Power_kW'])
fig1.update_traces(mode='markers+lines', line = dict(color='royalblue', width=1),
                                                     marker=dict(color='firebrick',symbol='octagon',size=3))
fig1.update_layout(margin_pad=20)
fig1.update_yaxes(title_text='Power kW')

tot['day']=tot['day'].shift(-24) # Previous hour consumption
tot['day'].fillna(1,inplace=True)

hol['Date'] = pd.to_datetime(hol['Date'])
hol = hol.set_index('Date',drop= True).resample('H').mean()
hol['Datetime']=hol.index
hol.index = pd.to_datetime(hol.index, format='%y/%m/%d  %H:%M:%S').strftime('%d/%m/%y')
hol['Holiday'] = hol.groupby([hol.index])['Holiday'].ffill()

hol['Holiday']=hol['Holiday'].fillna(0)
hol['Datetime'] = pd.to_datetime(hol['Datetime'])
hol=hol.set_index('Datetime',drop= True)
hol.index = pd.to_datetime(hol.index, format='%y/%m/%d %H:%M:%S').strftime('%d/%m/%y %H:%M:%S')

hol=hol[~hol.index.str.contains('/19')]
tot['Holiday']=hol['Holiday']
tot['Holiday'] = np.where(tot['Holiday'] == 0, 1,0)

tot['time_hour'] = pd.to_datetime(tot.index).hour

#Remove outliers
tot_clean3 = tot[tot['Power_kW'] >tot['Power_kW'].quantile(0.25) ]

fig2=px.line(x=tot_clean3.index, y=tot_clean3['Power_kW'])
fig2.update_traces(mode='markers+lines', line = dict(color='royalblue', width=1),
                                                     marker=dict(color='firebrick',symbol='octagon',size=3))

fig2.update_layout(margin_pad=20)
fig2.update_yaxes(title_text='Power kW')

#########################

# Clustering

# import KMeans that is a clustering method 
from sklearn.cluster import KMeans
from pandas import DataFrame #to manipulate data frame

# create kmeans object
model = KMeans(n_clusters=3).fit(tot_clean3) #fit means to develop the model on the given data
pred = model.labels_
tot_clean3['Cluster']=pred
pred

df1 = tot_clean3[tot_clean3.isna().any(axis=1)]

## Graphical representations of the clusters

cluster_0=tot_clean3[pred==0]
cluster_1=tot_clean3[pred==1]
cluster_2=tot_clean3[pred==2]

#ax1=tot_clean3.plot.scatter(x='temp_C',y='Power_kW',c='Cluster',colormap='summer', sharex=False)
ax1=px.scatter(x=tot_clean3['temp_C'], y=tot_clean3['Power_kW'], color=tot_clean3['Cluster'])
#do the plot using plotlyexpress

#tot_clean3.plot.scatter(x='time_hour',y='Power_kW',c='Cluster',colormap='summer', sharex=False)
ax2=px.scatter(x=tot_clean3['time_hour'], y=tot_clean3['Power_kW'], color=tot_clean3['Cluster'])

#ax3=tot_clean3.plot.scatter(x='day',y='Power_kW',c='Cluster',colormap='summer', sharex=False)
ax3=px.scatter(x=tot_clean3['day'], y=tot_clean3['Power_kW'], color=tot_clean3['Cluster'])

ax1.update_yaxes(title_text='Power kW')
ax1.update_xaxes(title_text='temp_C')
ax2.update_yaxes(title_text='Power kW')
ax2.update_xaxes(title_text='time_hour')
ax3.update_yaxes(title_text='Power kW')
ax3.update_xaxes(title_text='day')

#pathways
df=tot_clean3
df=df.drop(columns=['temp_C','HR','windSpeed_m/s','windGust_m/s','pres_mbar','solarRad_W/m2','rain_mm/h','rain_day','Holiday','day'])
df=df.rename(columns = {'Power_kW':'Power'})
df.index = pd.to_datetime(df.index, format='%d/%m/%y %H:%M:%S').strftime('%d/%m/%y')
#Create a pivot table creating a table where une column becomes a pivot  
df_pivot = df.pivot_table(values='Power', index=[df.index], columns=['time_hour'])
df_pivot = df_pivot.dropna()

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler # it normalize the data to find the euclidian distance or it doesn't make sense 
from sklearn.metrics import silhouette_score

sillhoute_scores = []
n_cluster_list = np.arange(2,10).astype(int)

X = df_pivot.values.copy()
    
# Very important to scale!
sc = MinMaxScaler()
X = sc.fit_transform(X)

for n_cluster in n_cluster_list:
    
    kmeans = KMeans(n_clusters=n_cluster)
    cluster_found = kmeans.fit_predict(X)
    sillhoute_scores.append(silhouette_score(X, kmeans.labels_))

kmeans = KMeans(n_clusters=3)
cluster_found = kmeans.fit_predict(X)
cluster_found_sr = pd.Series(cluster_found, name='cluster')
df_pivot = df_pivot.set_index(cluster_found_sr, append=True )

fig3, ax= plt.subplots(1,1, figsize=(18,10))
color_list = ['blue','red','green']
cluster_values = sorted(df_pivot.index.get_level_values('cluster').unique())
for cluster, color in zip(cluster_values, color_list):
    df_pivot.xs(cluster, level=1).T.plot(    #xs prende solo cluster leve=1 prende i numeri 1 Tplot serve per plottare dopo aver fatto operazioni
        ax=ax, legend=False, alpha=0.01, color=color, label= f'Cluster {cluster}')
    df_pivot.xs(cluster, level=1).median().plot(
        ax=ax, color=color, alpha=0.9, ls='--'    )

ax.set_xticks(np.arange(1,25))
ax.set_ylabel('kilowatts')
ax.set_xlabel('hour')
fig3.savefig('fig3.png')
#ax.legend() the difference between green and red is the chang between solare e legale ora 

tot_clean3=tot_clean3.drop(columns=['Cluster'])

## Feature Extraction/Engineering 

#Power of the prevous hour
tot_clean3['Power-1_kW']=tot_clean3['Power_kW'].shift(1) # Previous hour consumption
tot_clean3=tot_clean3.dropna()

#Log of temperature
tot_clean3['logtemp']=np.log(tot_clean3['temp_C'])
#The difference between high T and low T is enhanced in the logaritmic scale

# Weekday square
tot_clean3['day2']=np.square(tot_clean3['day'])
#The difference between the days is enhanced

#Holiday/weekday
tot_clean3['Holtimesweek']=tot_clean3['day2']*tot_clean3['Holiday']
#It takes into account the holidays as if they were Sundays

tot_clean3['workday'] = tot_clean3['day']*tot_clean3['Holiday']
#With this feature all the Sundays corresond to 0, while the all the other days are 1

#dayweektimeholiday
tot_clean3['workday']=tot_clean3['workday']*tot_clean3['Holiday']

#Hour parabolic shape
tot_clean3['hour_par']=-((np.square(tot_clean3['time_hour'])/2)-14*tot_clean3['time_hour']+26)
#With this engineered feature the hour became parabolic with the vertex at 2 p.m. which is not only the hottest hour of the day but it is also very central in the tipical day of IST

fig4=go.Figure()
fig4.add_scatter(name="Power",x=tot_clean3.index, y=tot_clean3['Power_kW'], mode='lines')
fig4.add_scatter(name="temp_C",x=tot_clean3.index, y=tot_clean3['temp_C']*30, mode='lines')
fig4.add_scatter(name="logtemp",x=tot_clean3.index, y=tot_clean3['logtemp']*100, mode='lines')
fig4.update_layout(xaxis_range=[4100,4200],showlegend=True)
  
fig4.update_layout(xaxis_range=[4100,4200],showlegend=True)
fig4.update_layout(legend=dict(
    orientation="h",
   yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

fig5=go.Figure()
fig5.add_scatter(name="Power",x=tot_clean3.index, y=tot_clean3['Power_kW'], mode='lines')
fig5.add_scatter(name="day",x=tot_clean3.index, y=tot_clean3['day']*100, mode='lines')
fig5.add_scatter(name="Holiday",x=tot_clean3.index, y=tot_clean3['Holiday']*100, mode='lines')
fig5.add_scatter(name="timehour",x=tot_clean3.index, y=tot_clean3['time_hour']*10, mode='lines')
fig5.add_scatter(name="Holtimesweek",x=tot_clean3.index, y=tot_clean3['Holtimesweek']*8, mode='lines')
fig5.add_scatter(name="hour_par",x=tot_clean3.index, y=tot_clean3['hour_par']*20, mode='lines')
fig5.update_layout(xaxis_range=[4100,4200],showlegend=True)  

fig5.update_layout(xaxis_range=[4100,4200],showlegend=True)  
fig5.update_layout(legend=dict(
    orientation="h",
 yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))


tot_clean4=tot_clean3.drop(columns=['HR','solarRad_W/m2','pres_mbar','windSpeed_m/s','windGust_m/s','rain_mm/h','rain_day','day2'])
tot_clean4.columns

tot_clean4=tot_clean4.drop(columns=['Holtimesweek','day','Holiday'])

# Clustering

# create kmeans object
model = KMeans(n_clusters=3).fit(tot_clean4)
pred = model.labels_
tot_clean4['Cluster']=pred
pred

df1 = tot_clean4[tot_clean4.isna().any(axis=1)]
print (df1)

# each line is a data point so first line belong to class 2 line to belong to class 2 3line to 0
Nc = range(1, 20)
#creating a matrix with cluster increasing
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
# score is an array where there is calculate the "score " that is the square distance between different point 
# 
score = [kmeans[i].fit(tot_clean4).score(tot_clean4) for i in range(len(kmeans))]

cluster_0=tot_clean4[pred==0]
cluster_1=tot_clean4[pred==1]
cluster_2=tot_clean4[pred==2]

#aax1=tot_clean4.plot.scatter(x='logtemp',y='Power_kW',c='Cluster',colormap='summer', sharex=False)
aax1=px.scatter(x=tot_clean4['logtemp'], y=tot_clean4['Power_kW'], color=tot_clean4['Cluster'])


#tot_clean4.plot.scatter(x='time_hour',y='Power_kW',c='Cluster',colormap='summer', sharex=False)
aax2=px.scatter(x=tot_clean4['time_hour'], y=tot_clean4['Power_kW'], color=tot_clean4['Cluster'])
#The light green points appearing at night could be due to the fact that some devices may have taken advantage of the low price of the electricity at night (for example to cool down the temperature of the building).

#aax3=tot_clean4.plot.scatter(x='workday',y='Power_kW',c='Cluster',colormap='summer', sharex=False)
aax3=px.scatter(x=tot_clean4['workday'], y=tot_clean4['Power_kW'], color=tot_clean4['Cluster'])

aax1.update_yaxes(title_text='Power kW')
aax1.update_xaxes(title_text='logtemp')
aax2.update_yaxes(title_text='Power kW')
aax2.update_xaxes(title_text='time_hour')
aax3.update_yaxes(title_text='Power kW')
aax3.update_xaxes(title_text='workday')

df=tot_clean4
df=df.drop(columns=[ 'Power-1_kW','workday','logtemp', 'hour_par'])
df=df.rename(columns = {'Power_kW':'Power'})
df.index = pd.to_datetime(df.index, format='%d/%m/%y %H:%M:%S').strftime('%d/%m/%y')
#Create a pivot table creating a table where une column becomes a pivot  
df_pivot = df.pivot_table(values='Power', index=[df.index],columns=['time_hour'])
df_pivot = df_pivot.dropna()


sillhoute_scores = []
n_cluster_list = np.arange(2,10).astype(int)

X = df_pivot.values.copy()
    
# Very important to scale!
sc = MinMaxScaler()
X = sc.fit_transform(X)

for n_cluster in n_cluster_list:
    
    kmeans = KMeans(n_clusters=n_cluster)
    cluster_found = kmeans.fit_predict(X)
    sillhoute_scores.append(silhouette_score(X, kmeans.labels_))

kmeans = KMeans(n_clusters=3)
cluster_found = kmeans.fit_predict(X)
cluster_found_sr = pd.Series(cluster_found, name='cluster')
df_pivot = df_pivot.set_index(cluster_found_sr, append=True )

fig6, ax= plt.subplots(1,1, figsize=(18,10))
color_list = ['blue','red','green']
cluster_values = sorted(df_pivot.index.get_level_values('cluster').unique())
for cluster, color in zip(cluster_values, color_list):
    df_pivot.xs(cluster, level=1).T.plot(    #xs prende solo cluster leve=1 prende i numeri 1 Tplot serve per plottare dopo aver fatto operazioni
        ax=ax, legend=False, alpha=0.01, color=color, label= f'Cluster {cluster}')
    df_pivot.xs(cluster, level=1).median().plot(
        ax=ax, color=color, alpha=0.9, ls='--'    )

ax.set_xticks(np.arange(1,25))
ax.set_ylabel('kilowatts')
ax.set_xlabel('hour')
fig6.savefig('fig6.png')
tot_clean4=tot_clean4.drop(columns=['Cluster'])

#Regression
tot_clean4.columns
from sklearn.model_selection import train_test_split
from sklearn import  metrics

# recurrent
X=tot_clean4.values
Y=X[:,1]
X=X[:,[0,2,3,4,5,6]] 

X_train, X_test, y_train, y_test = train_test_split(X,Y)

#Linear Regression
from sklearn import  linear_model

regr = linear_model.LinearRegression()

regr.fit(X_train,y_train)

y_pred_LR = regr.predict(X_test)

MAE_LR=metrics.mean_absolute_error(y_test,y_pred_LR) 
MSE_LR=metrics.mean_squared_error(y_test,y_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y_test)

err= {'Methods':['LR'], 'Error':[cvRMSE_LR]}
err=pd.DataFrame(err)

fig7=go.Figure()
fig7.add_scatter(name="Ytest",y=y_test, mode='lines')
fig7.add_scatter(name="y_pred_LR",y=y_pred_LR, mode='lines')
fig8=px.scatter(x=y_test, y=y_pred_LR)

fig7.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

fig8.update_yaxes(title_text='y_pred_LR')
fig8.update_xaxes(title_text='y_test')

#Random Forest
from sklearn.ensemble import RandomForestRegressor

parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 200, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 20,
              'max_leaf_nodes': None}
RF_model = RandomForestRegressor(**parameters)

RF_model.fit(X_train, y_train)
y_pred_RF = RF_model.predict(X_test)

MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)

new_row = {'Methods':'RF', 'Error':cvRMSE_RF}
err = err.append(new_row, ignore_index=True)

fig9=go.Figure()
fig9.add_scatter(name="Ytest",y=y_test, mode='lines')
fig9.add_scatter(name="y_pred_RF",y=y_pred_RF, mode='lines')
fig10=px.scatter(x=y_test, y=y_pred_RF)

fig9.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

fig10.update_yaxes(title_text='y_pred_RF')
fig10.update_xaxes(title_text='y_test')

#Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor

#params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
#          'learning_rate': 0.01, 'loss': 'ls'}
#GB_model = GradientBoostingRegressor(**params)
GB_model = GradientBoostingRegressor()
GB_model.fit(X_train, y_train)
y_pred_GB =GB_model.predict(X_test)

MAE_GB=metrics.mean_absolute_error(y_test,y_pred_GB) 
MSE_GB=metrics.mean_squared_error(y_test,y_pred_GB)  
RMSE_GB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_GB))
cvRMSE_GB=RMSE_GB/np.mean(y_test)

new_row = {'Methods':'GB', 'Error':cvRMSE_GB}
err = err.append(new_row, ignore_index=True)

fig11=go.Figure()
fig11.add_scatter(name="Ytest",y=y_test, mode='lines')
fig11.add_scatter(name="y_pred_GB",y=y_pred_GB, mode='lines')
fig12=px.scatter(x=y_test, y=y_pred_GB)

fig11.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

fig12.update_yaxes(title_text='y_pred_GB')
fig12.update_xaxes(title_text='y_test')


err=err.set_index('Methods',drop = False)
fig13=px.bar(err, x="Methods", y="Error")
fig13.update_xaxes(title_text='Forecasting Methods')
fig13.update_yaxes(title_text='Error')
###########################

import dash
import dash_core_components as dcc
import dash_html_components as html
import base64
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

image_filename ='fig3.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

image_filename2 ='fig6.png'
encoded_image2 = base64.b64encode(open(image_filename2, 'rb').read())


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Img(src=app.get_asset_url('IST_logo.png')),
    html.H2('by Pierluigi Ciasullo ist1100826'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Exploratory Data Analysis', value='tab-1'),
        dcc.Tab(label='Clustering', value='tab-2'),
        dcc.Tab(label='Feature Selection', value='tab-3'),
        dcc.Tab(label='Regression', value='tab-4'),
        
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
             

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Exploratory Data Analysis'),
            dcc.RadioItems(
        id='radio',
        options=[
            {'label': 'Raw Data', 'value': 1},
            {'label': 'Clean Data', 'value': 2}
        ], 
        value=1
        ),
         html.Div(id='EDA_html'),
                    ],style={'textAlign': 'center'}) 
    
    elif tab == 'tab-2':
        return html.Div([
            html.H2('Clustering'),
            dcc.Dropdown( 
        id='dropdown',
        options=[
            {'label': 'Power vs Temperature', 'value': 1},
            {'label': 'Power vs Hour', 'value': 2},
            {'label': 'Power vs Day', 'value': 3},
            {'label': 'Silhouettes Score', 'value': 4},
        ], 
        value=1
        ),
        html.Div([
            html.Div([
                html.H3('Clustering performed on the complete database'),
                html.Div(id='cluster1')
                ], 
            className="six columns"
            ),
           
            html.Div([
                 html.H3('Clustering performed after Feature Selection'),
                 html.Div(id='cluster2')
                ], 
            className="six columns"
            ),
    ], 
        className="row"
        )
],style={'textAlign': 'center'})
            
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Feature selection'),
             dcc.Dropdown( 
        id='dropdown2',
        options=[
            {'label': 'Temperature', 'value': 1},
            {'label': 'Day and Hour', 'value': 2},
        ], 
        value=1
        ),
        html.Div(id='Featureselection_html'),
            
        ],style={'textAlign': 'center'})
    
    elif tab == 'tab-4':
        return html.Div([
            html.H3('Regression'),
                dcc.Dropdown( 
        id='dropdown3',
        options=[
            {'label': 'Linear', 'value': 1},
            {'label': 'Random Forest', 'value': 2},
            {'label': 'Gradient Boosting', 'value': 3},
            {'label': 'Errors of the Forcasting Models', 'value': 4},
        ], 
        value=1
        ),
          html.Div([
            html.Div([
                html.Div(id='Regression1')
                ], 
            className="six columns"
            ),
            html.Div([
                 html.Div(id='Regression2')
                ], 
            className="six columns"
            ),
    ], 
        className="row"
        )
],style={'textAlign': 'center'})

@app.callback(Output('EDA_html', 'children'), 
              Input('radio', 'value'))

def render_figure_html(EDA_RI):
    
    if EDA_RI == 1:
        return html.Div([dcc.Graph(figure=fig1),])
    elif EDA_RI == 2:
        return html.Div([dcc.Graph(figure=fig2),])
    
    
@app.callback(Output('cluster1', 'children'), 
              Input('dropdown', 'value'))


def render_figure_html(cluster_RI):
    
    if cluster_RI == 1:
        return html.Div([dcc.Graph(figure=ax1),])
    elif cluster_RI == 2:
        return html.Div([dcc.Graph(figure=ax2),])
    elif cluster_RI == 3:
        return html.Div([dcc.Graph(figure=ax3),])       
    elif cluster_RI == 4:
        return html.Div([html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),style={'height':'100%','width':'100%'})
        ])
    
@app.callback(Output('cluster2', 'children'), 
              Input('dropdown', 'value'))


def render_figure_html(cluster_RI2):
    
   if cluster_RI2 == 1:
            return html.Div([dcc.Graph(figure=aax1),])
   elif cluster_RI2 == 2:
            return html.Div([dcc.Graph(figure=aax2),])
   elif cluster_RI2 == 3:
            return html.Div([dcc.Graph(figure=aax3),])
   elif cluster_RI2 == 4:
        return html.Div([html.Img(src='data:image/png;base64,{}'.format(encoded_image2.decode()),style={'height':'100%','width':'100%'})
        ])
    
@app.callback(Output('Featureselection_html', 'children'), 
              Input('dropdown2', 'value'))

def render_figure_html(FS_RI):
    
    if FS_RI == 1:
        return html.Div([dcc.Graph(figure=fig4),])
    elif FS_RI == 2:
        return html.Div([dcc.Graph(figure=fig5),])
    
@app.callback(Output('Regression1', 'children'), 
              Input('dropdown3', 'value'))

def render_figure_html(Regression_RI1):
    
    if Regression_RI1 == 1:
        return html.Div([dcc.Graph(figure=fig7),])
    elif Regression_RI1 == 2:
        return html.Div([dcc.Graph(figure=fig9),])
    elif Regression_RI1 == 3:
        return html.Div([dcc.Graph(figure=fig11),])
    elif Regression_RI1 == 4:
        return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in err.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(err.iloc[i][col]) for col in err.columns
            ]) for i in range(len(err))
        ])
    ], style={'marginLeft': 'auto', 'marginRight': 'auto', 'marginTop': '7vw'})
    
@app.callback(Output('Regression2', 'children'), 
              Input('dropdown3', 'value'))

def render_figure_html(Regression_RI2):
    
    if Regression_RI2 == 1:
        return html.Div([dcc.Graph(figure=fig8),])
    elif Regression_RI2 == 2:
        return html.Div([dcc.Graph(figure=fig10),])
    elif Regression_RI2 == 3:
        return html.Div([dcc.Graph(figure=fig12),])
    elif Regression_RI2 == 4:
        return html.Div([dcc.Graph(figure=fig13),])
    
    
    
    
if __name__ == '__main__':
    app.run_server(debug=False)