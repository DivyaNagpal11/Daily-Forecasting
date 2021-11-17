# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:00:40 2020

@author: DN067571
"""

#Import files


import statsmodels.api as sm
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None


#Model class
class SARIMAX:

    def __init__(self,**kwargs):

        #input parameters
        self.train,self.test = kwargs.get('train'),kwargs.get('test')       
        self.random_state = kwargs.get('random_state',1)
        self.target_dates = kwargs.get('target_dates')
        self.target_value = kwargs.get('target_value')
        self.target_item = kwargs.get('ti')
        self.target_items = kwargs.get('target_items')
        self.set_no = kwargs.get('set_no')
        self.parameters_list = kwargs.get('parameters_list',None)
        self.walk_forward = kwargs.get('wf')
        self.n_periods = int(kwargs.get('n_periods'))

        #output values
        self.fitted_values = pd.DataFrame()
        self.test_prediction = pd.DataFrame()
        self.unseen_prediction = pd.DataFrame()        
        self.apes = []
        self.results=[]
        self.mape_res=pd.DataFrame()
        self.run_model()
        del self.train,self.test

    def fit(self,data,exog_data,param):
        return sm.tsa.statespace.SARIMAX(data,exog=exog_data,order=(param[0],param[1],param[2]),
                                                seasonal_order=(param[3],param[4],param[5],param[6])).fit(disp=-1)
        
    def fitted_data(self,model,param=None):
        return model.fittedvalues 

    def forecast(self,model,exog_data,n_periods):
        return model.forecast(exog=exog_data,steps=n_periods) 

    def mean_absolute_percentage_error(self,y_true,y_pred):
        return np.mean(np.abs(np.subtract(y_true,y_pred)/y_true))*100  
    
    def median_absolute_percentage_error_daily(self,y_true,y_pred):
        y_true=pd.Series(y_true)
        y_pred=pd.Series(y_pred)
        true_pred = pd.DataFrame(zip(y_true,y_pred),columns=['y_true','y_pred'])
        true_pred.drop(true_pred[true_pred['y_pred'] == 0].index, axis=0, inplace=True)
        true_pred.drop(true_pred[true_pred['y_true'] == 0].index, axis=0, inplace=True)
        return np.median(np.abs(np.subtract(true_pred.y_true,true_pred.y_pred)/true_pred.y_true))*100


    def calculate_apes(self):
        for i,j in zip(self.test[self.target_value].values.flatten(),self.test_prediction.values.flatten()):            
            self.apes.append(self.mean_absolute_percentage_error(i,j))


    def optimize_SARIMAX(self,parameters_list,train_set,val_set,exog_columns):
        results=[]
        best_adj_mape = float('inf')
        for param in parameters_list:
            try:
                model=self.fit(train_set[self.target_value],train_set[exog_columns],param)
                fore1=self.forecast(model,val_set[exog_columns],len(val_set))
                fore1=np.array(fore1)
                y_true=np.array(list(train_set[self.target_value]))
                y_pred=np.array(list(model.fittedvalues))
                val_set_charges=np.array(list(val_set[self.target_value]))
                train_mape=round(self.median_absolute_percentage_error_daily(y_true[-365:],y_pred[-365:]),2)
                val_mape=round((self.median_absolute_percentage_error_daily(val_set_charges,fore1)),2)
                adj_mape = train_mape*len(y_true)/(len(y_true)+len(val_set))+val_mape*len(val_set)/(len(y_true)+len(val_set))
                if adj_mape <= best_adj_mape:
                    best_adj_mape=adj_mape
                    best_model = model    
                results.append([param,model.aic,train_mape,val_mape,adj_mape])
            except:
                continue    
        result_table=pd.DataFrame(results)
        result_table.columns=['parameters','aic','train_mape','val_mape','adj_mape']
        result_table = result_table.sort_values(by='adj_mape',ascending=True).reset_index(drop=True)
        return result_table, best_model
    
    
    def run_model(self):
        print("in sarimax")
        self.train=self.train.set_index(self.target_dates)
        self.test=self.test.set_index(self.target_dates)
        exog_columns=self.train.columns
        exog_columns=exog_columns.drop(self.target_value)
        result_table, best_model = self.optimize_SARIMAX(self.parameters_list,self.train[:-(int(self.n_periods/2))],self.train[-(int(self.n_periods/2)):],exog_columns)
        print(result_table.head())
        fitted_val_list=[]

        actual_data = self.train
        res_aic_avg = 2* np.abs(result_table.aic.mean())
        
        c=1
        for param in result_table.parameters:
            try:
                if c > 5:
                    break
                model = self.fit(actual_data[self.target_value],actual_data[exog_columns],param)
                if model.aic < res_aic_avg:
                    c=c+1
                    fore_test=self.forecast(model,self.test[exog_columns],len(self.test))
                    fore_test_set=np.array(list(fore_test))
                    
                    get_list=[]
                    for i in range(len(fore_test_set)):
                        get_list.append(fore_test_set[i])
                    fitted_val_list.append(get_list)
                
            except:
                continue

        fitted_val=pd.DataFrame(fitted_val_list,columns=np.arange(1,self.n_periods+1))
        fitted_mean=[]
        for i in np.arange(1,self.n_periods+1):
            fitted_mean.append(fitted_val[i].mean())
        if self.walk_forward == "True":
            test_set1=np.array(list(self.test[self.target_value]))
            test_results=round(self.median_absolute_percentage_error_daily(test_set1,fitted_mean),2)
            self.results.append([self.target_item,"SARIMAX",round(result_table.train_mape[0],2),round(result_table.val_mape[0],2),test_results,list(self.test.index),fitted_mean,self.set_no])
            self.mape_res=pd.DataFrame(self.results,columns=[self.target_items,'model','train_mape','val_mape','test_mape','test_dates','test_predicted','set_no'])
        else:
            self.results.append([self.target_item,round(result_table.train_mape[0],2),round(result_table.val_mape[0],2),list(self.test.index),fitted_mean])
            self.mape_res=pd.DataFrame(self.results,columns=[self.target_items,'train_mape','val_mape','test_dates','test_predicted'])