# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:52:13 2020

@author: DN067571
"""

from itertools import product
from tqdm import tqdm_notebook
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')
import os
from sklearn.preprocessing import MinMaxScaler

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

    
def optimize_SARIMAX_daily(parameters_list,train_set,val_set,cols):      #train_set_transformed
    results=[]
    best_adj_mape = float('inf')
    for param in tqdm_notebook(parameters_list):
        try: 
            model=sm.tsa.statespace.SARIMAX(train_set["total_charge"],exog=train_set[cols],order=(param[0],param[1],param[2]),seasonal_order=(param[3],param[4],param[5],param[6])).fit(disp=-1)
            fore1=model.forecast(exog=val_set[cols],steps=len(val_set))#start=train_set_transformed.shape[0],end=train_set_transformed.shape[0]+len(val_set)-1)
            fore1=np.array(fore1)
#            fore_updated=val_set.copy()
#            fore_updated["total_charge"]=fore1
#            fore=scaler.inverse_transform(fore_updated)[:,0]
            
            y_true=np.array(list(train_set['total_charge']))
            y_pred=np.array(list(model.fittedvalues))
#            y_true=scaler.inverse_transform(train_set_transformed)[:,0]
#            
#            fitted_df=train_set_transformed.copy()
#            fitted_df["total_charge"]=model.fittedvalues.values.reshape(-1,1)
#            y_pred=scaler.inverse_transform(fitted_df)[:,0]
            
            train_mape=round((median_absolute_percentage_error_daily(y_true[-365:],y_pred[-365:])),2)
            val_set_charges=np.array(list(val_set["total_charge"]))
            val_mape=round((median_absolute_percentage_error_daily(val_set_charges,fore1)),2)
            adj_mape = train_mape*len(y_true)/(len(y_true)+len(val_set))+val_mape*len(val_set)/(len(y_true)+len(val_set))
            if adj_mape <= best_adj_mape:
                best_adj_mape=adj_mape
                best_model = model    
            results.append([param,model.aic,train_mape,val_mape,adj_mape])
        except:
            continue
        
    result_table=pd.DataFrame(results)
    result_table.columns=['parameters','aic','train_mape','val_mape','adj_mape']
    result_table=result_table.sort_values(by='adj_mape',ascending=True).reset_index(drop=True)
    return result_table, best_model

    
def sarimax_model_daily(train_set,val_set,test_set,be_name,set_no): 
#    train_set,val_set,test_set,be_name=train,val,test,be_name
    cols=val_set.columns
    cols=cols.drop('total_charge')
    p=range(0,2)
    d=range(0,2)
    q=range(0,2)
    P=range(0,3)
    D=range(0,2)
    Q=range(0,3)
    s=range(7,8)
    
    parameters=product(p,d,q,P,D,Q,s)
    parameters_list=list(parameters)
    
#    scaler = MinMaxScaler()
#    
#    scaler.fit(train_set)
#    train_set_transformed = pd.DataFrame(scaler.transform(train_set))
#    train_set_transformed.columns=train_set.columns
#    train_set_transformed.index=train_set.index 
    
    result_table, best_model = optimize_SARIMAX_daily(parameters_list,train_set,val_set,cols)
   
    p, d, q, P, D, Q, s=result_table.parameters[0]
    
#    fore1=best_model.forecast(exog=val_set[cols],steps=len(val_set))#start=train_set_transformed.shape[0],end=train_set_transformed.shape[0]+len(val_set)-1)
#    fore_updated=val_set.copy()
#    fore_updated["total_charge"]=fore1
#    fore_val=scaler.inverse_transform(fore_updated)[:,0]
    
    fore_val=best_model.predict(exog=val_set[cols],start=train_set.shape[0],end=train_set.shape[0]+len(val_set)-1)

#    fitted_df=train_set_transformed.copy()
#    fitted_df["total_charge"]=best_model.fittedvalues.values.reshape(-1,1)
#    fitted_values=scaler.inverse_transform(fitted_df)[:,0]
#            
    overall_train=pd.concat([train_set,val_set])
#    scaler.fit(overall_train)
#    overall_train_transformed = pd.DataFrame(scaler.transform(overall_train))
#    overall_train_transformed.columns=overall_train.columns
#    overall_train_transformed.index=overall_train.index
    fitted_val_list=[]
    res_aic_avg = 2* np.abs(result_table.aic.mean())
    
    c=1
    for p1, d1, q1, P1, D1, Q1, s1 in result_table.parameters:
        try:
            if c > 5:
                break
            best_model_overall=sm.tsa.statespace.SARIMAX(overall_train["total_charge"],overall_train[cols],order=(p1,d1,q1),seasonal_order=(P1,D1,Q1,s1)).fit(disp=-1)
            if best_model_overall.aic < res_aic_avg:
                c=c+1
                fore_test=best_model_overall.forecast(exog=test_set[cols],steps=len(test_set))#start=overall_train.shape[0],end=overall_train.shape[0]+len(test_set)-1)
                fore_test_set=np.array(list(fore_test))
#                forecasted_df=test_set.copy()
#                forecasted_df["total_charge"]=fore_test
#                fore_test_set=scaler.inverse_transform(forecasted_df)[:,0]
                
                get_list=[]
                for i in range(len(fore_test_set)):
                    get_list.append(fore_test_set[i])
                fitted_val_list.append(get_list)
            
        except:
            continue
    
    p, d, q, P, D, Q, s = result_table.parameters[0]
    
    fitted_val=pd.DataFrame(fitted_val_list,columns=[x for x in range(1,len(test_set)+1)])

    fitted_mean=[]
    for i in range(1,len(test_set)+1):
        fitted_mean.append(fitted_val[i].mean())

    test_set1=np.array(list(test_set["total_charge"]))
    test_results=round(median_absolute_percentage_error_daily(test_set1,fitted_mean),2)
    
    graph_path_tso = r"daily_sarimax_walk_forward_val_set_46_new_regressors/"+be_name+"/"
    if not os.path.exists(graph_path_tso):
        os.makedirs(graph_path_tso)
        
    trace1 = go.Scatter(x=train_set[s+d:].index, y=train_set[s+d:]['total_charge'], mode='lines+markers',name="Actual values: Train", marker=dict(color="blue",size=9, line=dict(width=1)))
    trace2 = go.Scatter(x=train_set[s+d:].index, y=best_model.fittedvalues[s+d:].values, mode='lines+markers',name="Fitted values: Train", marker=dict(color="red",size=9,line=dict(width=1)))
    trace3 = go.Scatter(x=val_set['total_charge'].index, y=val_set['total_charge'], mode='lines+markers',name="Actual values: Val", marker=dict(color="blue",size=9,line=dict(width=1)))
    trace4 = go.Scatter(x=val_set['total_charge'].index, y=fore_val.values, mode='lines+markers',name="Predicted values: Val", marker=dict(color="rgb(44, 160, 44)",size=9,line=dict(width=1))) 
    trace5 = go.Scatter(x=test_set['total_charge'].index, y=test_set['total_charge'], mode='lines+markers',name="Actual values: Test", marker=dict(color="blue",size=9,line=dict(width=1)))
    trace6 = go.Scatter(x=test_set['total_charge'].index, y=fitted_mean, mode='lines+markers',name="Predicted values: Test", marker=dict(color="orange",size=9,line=dict(width=1))) 
    data = [trace1, trace2, trace3, trace4, trace5, trace6]
    layout = go.Layout(go.Layout(title='Billing Entity : {} <br>Train MAPE: {} Val MAPE: {} Test MAPE: {} <br>SARIMAX(p,d,q)(P,D,Q,s):({},{},{})({},{},{},{})'.format(be_name,round(result_table.train_mape[0],2),round(result_table.val_mape[0],2),test_results,p, d, q, P, D, Q, s),yaxis=dict(title="Daily Charges", zeroline=False),
                                 xaxis=go.layout.XAxis(
                                     title="Day-Month-Year",
                                 ), boxmode='group'))
    fig = go.Figure(data=data,layout=layout)
    plotly.offline.plot(fig, filename=graph_path_tso+str(set_no)+"_"+be_name, image='png')
    return round(result_table.train_mape[0],2),round(result_table.val_mape[0],2),test_set.index,fitted_mean,test_results