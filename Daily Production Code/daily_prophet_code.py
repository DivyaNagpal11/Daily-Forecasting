# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 01:36:31 2020

@author: SP075598
"""

from itertools import product
from tqdm import tqdm
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
import os
from fbprophet import Prophet
import warnings
warnings.filterwarnings('ignore')

def mean_absolute_percentage_error_daily(y_true,y_pred):
    y_true=pd.Series(y_true)
    y_pred=pd.Series(y_pred)
    true_pred = pd.DataFrame(zip(y_true,y_pred),columns=['y_true','y_pred'])
    true_pred.drop(true_pred[true_pred['y_pred'] == 0].index, axis=0, inplace=True)
    true_pred.drop(true_pred[true_pred['y_true'] == 0].index, axis=0, inplace=True)
    return np.mean(np.abs(np.subtract(true_pred.y_true,true_pred.y_pred)/true_pred.y_true))*100

def median_absolute_percentage_error_daily(y_true,y_pred):
    y_true=pd.Series(y_true)
    y_pred=pd.Series(y_pred)
    true_pred = pd.DataFrame(zip(y_true,y_pred),columns=['y_true','y_pred'])
    true_pred.drop(true_pred[true_pred['y_pred'] == 0].index, axis=0, inplace=True)
    true_pred.drop(true_pred[true_pred['y_true'] == 0].index, axis=0, inplace=True)
    return np.median(np.abs(np.subtract(true_pred.y_true,true_pred.y_pred)/true_pred.y_true))*100
#    return np.mean(np.abs(np.subtract(y_true,y_pred)/y_true))*100
  
def optimize_prophet(parameters_list,train_dataset,val_dataset,steps):  
    results=[]
    best_adj_mape=float('inf')
    for i in tqdm(parameters_list):
        forecast=pd.DataFrame()
        future=pd.DataFrame()
        
        prophet_basic = Prophet(growth='linear',daily_seasonality=False,weekly_seasonality=True,yearly_seasonality=True,holidays_prior_scale=10,n_changepoints=i[0],changepoint_prior_scale=i[1])
#        prophet_basic = Prophet(growth='linear',daily_seasonality=False,weekly_seasonality=True,yearly_seasonality=True,n_changepoints=20,changepoint_prior_scale=0.3)
        prophet_basic.add_regressor('is_extended')
        prophet_basic.add_regressor('Is_Month_End')
        prophet_basic.add_regressor('LastButOneDay')
        prophet_basic.add_regressor('LastSecDay')
#        prophet_basic.add_regressor('6th_Day')
        prophet_basic.add_regressor('13th_Day')
#        prophet_basic.add_regressor('20th_Day')
#        prophet_basic.add_regressor('27th_Day')
        prophet_basic.add_regressor('31st_Day')
        prophet_basic.add_country_holidays(country_name='US')
#        prophet_basic.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        prophet_basic.fit(train_dataset)
        
        future= prophet_basic.make_future_dataframe(periods=len(val_dataset))
        x=train_dataset.append(val_dataset)
        future['is_extended'] =pd.Series(x['is_extended'].values)
        future['Is_Month_End'] =pd.Series(x['Is_Month_End'].values)
        future['LastButOneDay'] =pd.Series(x['LastButOneDay'].values)
        future['LastSecDay'] =pd.Series(x['LastSecDay'].values)
#        future['6th_Day'] =pd.Series(x['6th_Day'].values)
        future['13th_Day'] =pd.Series(x['13th_Day'].values)
#        future['20th_Day'] =pd.Series(x['20th_Day'].values)
#        future['27th_Day'] =pd.Series(x['27th_Day'].values)
        future['31st_Day'] =pd.Series(x['31st_Day'].values)
        forecast=prophet_basic.predict(future)
        
        y_true=np.array(list(train_dataset['y']))
        y_pred=np.array(list(forecast.yhat[:-steps]))
        val_predicted=np.array(list(forecast.yhat[-steps:]))
        train_mape=round((median_absolute_percentage_error_daily(y_true[-365:],y_pred[-365:])),2)
        val_mape=round((median_absolute_percentage_error_daily(val_dataset["y"],val_predicted)),2)
        adj_mape = train_mape*len(y_true)/(len(y_true)+len(val_dataset))+val_mape*len(val_dataset)/(len(y_true)+len(val_dataset))
        
        if adj_mape <= best_adj_mape:
            best_adj_mape=adj_mape
            best_model = prophet_basic
            
        results.append([i,train_mape,val_mape,adj_mape])
        
    result_table=pd.DataFrame(results,columns=['parameters','train_mape','val_mape','adj_mape'])
    result_table=result_table.sort_values(by='adj_mape',ascending=True).reset_index(drop=True)
    return result_table, best_model


def daily_prophet(train_set,val_set,test_set,be_name):
#    train_set,val_set,test_set,be_name=train[["total_charge","is_extended","Is_Month_End","LastButOneDay",
#                                              "LastSecDay",'13th_Day','31st_Day']],val[["total_charge","is_extended","Is_Month_End","LastButOneDay","LastSecDay",'13th_Day','31st_Day']],test[["total_charge","is_extended","Is_Month_End","LastButOneDay","LastSecDay",'13th_Day','31st_Day']],be_name
    train_dataset= pd.DataFrame()
    val_dataset= pd.DataFrame()
    train_set=train_set.reset_index()
    val_set=val_set.reset_index()
    train_dataset['ds'] = train_set["posted_date"]
    train_dataset['y']=train_set["total_charge"]
    train_dataset['is_extended']=train_set["is_extended"]
    train_dataset['Is_Month_End']=train_set["Is_Month_End"]
    train_dataset['LastButOneDay']=train_set["LastButOneDay"]
    train_dataset['LastSecDay']=train_set["LastSecDay"]
#    train_dataset['6th_Day']=train_set["6th_Day"]
    train_dataset['13th_Day']=train_set["13th_Day"]
#    train_dataset['20th_Day']=train_set["20th_Day"]
#    train_dataset['27th_Day']=train_set["27th_Day"]
    train_dataset['31st_Day']=train_set["31st_Day"]
    val_dataset['ds'] = val_set["posted_date"]
    val_dataset['y']=val_set["total_charge"]
    val_dataset['is_extended']=val_set["is_extended"]
    val_dataset['Is_Month_End']=val_set["Is_Month_End"]
    val_dataset['LastButOneDay']=val_set["LastButOneDay"]
    val_dataset['LastSecDay']=val_set["LastSecDay"]
#    val_dataset['6th_Day']=val_set["6th_Day"]
    val_dataset['13th_Day']=val_set["13th_Day"]
#    val_dataset['20th_Day']=val_set["20th_Day"]
#    val_dataset['27th_Day']=val_set["27th_Day"]
    val_dataset['31st_Day']=val_set["31st_Day"]
    steps = len(val_set)
      
    cp=(.001,.005,0.01,0.05,0.1,0.3)
    ncp=(30,40)
    
    parameters=product(ncp,cp)
    parameters_list=list(parameters)
    result_table, best_model = optimize_prophet(parameters_list,train_dataset,val_dataset,steps)
    future2= best_model.make_future_dataframe(periods=steps)
    x=train_dataset.append(val_dataset)
    future2['is_extended'] =pd.Series(x['is_extended'].values)
    future2['Is_Month_End'] =pd.Series(x['Is_Month_End'].values)
    future2['LastButOneDay'] =pd.Series(x['LastButOneDay'].values)
    future2['LastSecDay'] =pd.Series(x['LastSecDay'].values)
#    future2['6th_Day'] =pd.Series(x['6th_Day'].values)
    future2['13th_Day'] =pd.Series(x['13th_Day'].values)
#    future2['20th_Day'] =pd.Series(x['20th_Day'].values)
#    future2['27th_Day'] =pd.Series(x['27th_Day'].values)
    future2['31st_Day'] =pd.Series(x['31st_Day'].values)
    forcast_val=best_model.predict(future2).yhat[-steps:]
    
#    overall_train=pd.concat([train_set,val_set])
#    overall_train['ds'] = overall_train.posted_date
#    overall_train['y'] = overall_train["total_charge"]
#    fitted_val_list=[]
    
    overall_train=pd.concat([train_set,val_set])
    overall_train['ds'] = overall_train.posted_date
    overall_train['y'] = overall_train["total_charge"]
    fitted_val_list=[]
    fitted_val_list_up=[]
    fitted_val_list_down=[]
    
    c=1
    for ncp,cp in result_table.parameters:
        try:
            if c > 3:
                break
            prophet_basic1 = Prophet(growth='linear',daily_seasonality=False,weekly_seasonality=True,yearly_seasonality=True,holidays_prior_scale=10,n_changepoints=ncp,changepoint_prior_scale=cp)
#            prophet_basic1.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            prophet_basic1.add_regressor('is_extended')
            prophet_basic1.add_regressor('Is_Month_End')
            prophet_basic1.add_regressor('LastButOneDay')
            prophet_basic1.add_regressor('LastSecDay')
#            prophet_basic1.add_regressor('6th_Day')
            prophet_basic1.add_regressor('13th_Day')
#            prophet_basic1.add_regressor('20th_Day')
#            prophet_basic1.add_regressor('27th_Day')
            prophet_basic1.add_regressor('31st_Day')
            prophet_basic1.add_country_holidays(country_name='US')
            prophet_basic1.fit(overall_train)
            #if prophet_basic1.aic < res_aic_avg:
            future1= prophet_basic1.make_future_dataframe(periods=len(test_set))
            x=overall_train.append(test_set)
            future1['is_extended'] =pd.Series(x['is_extended'].values)
            future1['Is_Month_End'] =pd.Series(x['Is_Month_End'].values)
            future1['LastButOneDay'] =pd.Series(x['LastButOneDay'].values)
            future1['LastSecDay'] =pd.Series(x['LastSecDay'].values)
#            future1['6th_Day'] =pd.Series(x['6th_Day'].values)
            future1['13th_Day'] =pd.Series(x['13th_Day'].values)
#            future1['20th_Day'] =pd.Series(x['20th_Day'].values)
#            future1['27th_Day'] =pd.Series(x['27th_Day'].values)
            future1['31st_Day'] =pd.Series(x['31st_Day'].values)
            forecast=prophet_basic1.predict(future1).yhat[-(len(test_set)):]
            fore_test_set_down=prophet_basic1.predict(future1).yhat_lower[-(len(test_set)):]
            fore_test_set_up=prophet_basic1.predict(future1).yhat_upper[-(len(test_set)):]
            
            c=c+1
            get_list=[]
            for i in range(len(forecast)):
                get_list.append(forecast.iloc[i])
            fitted_val_list.append(get_list)
            
            get_list=[]
            for i in range(len(fore_test_set_down)):
                get_list.append(fore_test_set_down.iloc[i])
            fitted_val_list_up.append(get_list)
            
            get_list=[]
            for i in range(len(fore_test_set_up)):
                get_list.append(fore_test_set_up.iloc[i])
            fitted_val_list_down.append(get_list)

        except:
            continue
        
    fitted_val=pd.DataFrame(fitted_val_list,columns=[x for x in range(1,len(test_set)+1)])
    fitted_val_up=pd.DataFrame(fitted_val_list_up,columns=[x for x in range(1,len(test_set)+1)])
    fitted_val_down=pd.DataFrame(fitted_val_list_down,columns=[x for x in range(1,len(test_set)+1)])
    
    fitted_mean=[]
    for i in range(1,len(test_set)+1):
        fitted_mean.append(fitted_val[i].mean())
    
    fitted_mean_up=[]
    for i in range(1,len(test_set)+1):
        fitted_mean_up.append(fitted_val_up[i].mean())
    
    fitted_mean_down=[]
    for i in range(1,len(test_set)+1):
        fitted_mean_down.append(fitted_val_down[i].mean())
        
#    test_set1=np.array(list(test_set["total_charge"]))
#    test_results=round(median_absolute_percentage_error_daily(test_set1,fitted_mean),2)
    

    
    graph_path_tso = r"feb/"
    if not os.path.exists(graph_path_tso):
        os.makedirs(graph_path_tso)
    trace1 = go.Scatter(x=train_set["posted_date"], y=train_set['total_charge'], mode='lines+markers',name="Actual values: Train", marker=dict(color="blue",size=9, line=dict(width=1)))
    trace2 = go.Scatter(x=train_set["posted_date"], y=best_model.predict(future2).yhat[:-steps], mode='lines+markers',name="Fitted values: Train", marker=dict(color="red",size=9,line=dict(width=1)))
    trace3 = go.Scatter(x=val_set['posted_date'], y=val_set['total_charge'], mode='lines+markers',name="Actual values: Val", marker=dict(color="blue",size=9,line=dict(width=1)))
    trace4 = go.Scatter(x=val_set['posted_date'], y=forcast_val, mode='lines+markers',name="Predicted values: Val", marker=dict(color="rgb(44, 160, 44)",size=9,line=dict(width=1))) 
#    trace5 = go.Scatter(x=test_set.index, y=test_set['total_charge'], mode='lines+markers',name="Actual values: Test", marker=dict(color="blue",size=9,line=dict(width=1)))
#    trace6 = go.Scatter(x=test_set.index, y=fitted_mean, mode='lines+markers',name="Predicted values: Test", marker=dict(color="orange",size=9,line=dict(width=1)))
#    data = [trace1, trace2, trace3, trace4, trace5, trace6]
    trace6 = go.Scatter(x=test_set.index, y=fitted_mean, mode='lines+markers',name="Forecasted values", marker=dict(color="orange",size=9,line=dict(width=1))) 
    trace7 = go.Scatter(x=test_set.index,y=fitted_mean_down,  mode='none', fill='tonexty',fillcolor='rgba(131, 90, 241,0.15)', showlegend=False, marker=dict(line=dict(width=1)))
    trace8 = go.Scatter(x=test_set.index,y=fitted_mean_up,  mode='none', fill='tonexty',fillcolor='rgba(131, 90, 241,0.15)', showlegend=False, marker=dict(line=dict(width=1)))
    data = [trace1, trace2, trace3, trace4,trace6,trace7,trace8]
    layout = go.Layout(go.Layout(title='Billing Entity : {} <br>Train MAPE: {} Val MAPE: {} <br>PROPHET'.format(be_name,round(result_table.train_mape[0],2),round(result_table.val_mape[0],2)),yaxis=dict(title="Daily Charges", zeroline=False),
                                 xaxis=go.layout.XAxis(
                                     title="Month-Year",
                                 ), boxmode='group'))
    fig = go.Figure(data=data,layout=layout)
    plotly.offline.plot(fig, filename=graph_path_tso+be_name, image='png')
    return round(result_table.train_mape[0],2),round(result_table.val_mape[0],2),test_set.index,fitted_mean,fitted_mean_down,fitted_mean_up
