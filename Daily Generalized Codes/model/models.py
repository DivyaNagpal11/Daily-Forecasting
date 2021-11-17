# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:34:11 2020

@author: DN067571
"""

# -*- coding: utf-8 -*-


import pandas as pd
import time
from itertools import product
import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

from SARIMAX import SARIMAX
from PROPHET import PROPHET

class Univariate():
    
    def split_train_test(self,subset_df,n_peroids):
        return subset_df[:-n_peroids],subset_df[-n_peroids:]
    
    
    def run_model(self,**kwargs):   
        if self.model_name == 'sarimax':
            try:
                starttime = time.time()
                list=self.model_param[self.model_name].values()
                parameters_list = [a for a in product(*list)]
                df=pd.DataFrame()
                df = SARIMAX(train = self.train,test = self.test,parameters_list=parameters_list,ti=self.target_item,**kwargs)
                self.models_ran=pd.concat([self.models_ran,df.mape_res])
                self.models_timetaken['sarimax'] = time.time()-starttime
            except:
                print('Dint run sarimax model for:',self.target_item)
                
        elif self.model_name == 'prophet':
            try:
                starttime = time.time()
                list=self.model_param[self.model_name].values()
                parameters_list = [a for a in product(*list)]
                df=pd.DataFrame
                df = PROPHET(train = self.train,test = self.test,parameters_list=parameters_list,ti=self.target_item,**kwargs)
                self.models_ran=pd.concat([self.models_ran,df.mape_res])
                self.models_timetaken['prophet'] = time.time()-starttime
            except:
                print('Dint run prophet model for:',self.target_item)
            
    def __init__(self,target_item,model_params,**kwargs):
        self.random_state = kwargs.get('random_state',1)
        self.target_dates = kwargs.get('target_dates')
        self.target_value = kwargs.get('target_value')
        self.n_periods = int(kwargs.get('n_periods'))
        self.model_name = kwargs.get('model_name','All')
        self.set_no = kwargs.get('set_no',0)
        self.walk_forward=kwargs.get('wf',"True")
        self.train,self.test = self.split_train_test(kwargs.get('data'),self.n_periods)
        self.target_item = target_item
        self.target_items = kwargs.get("target_items")
        self.model_param = model_params
        self.models_ran = pd.DataFrame()
        self.models_timetaken = {}

        if self.model_name == 'All':
            model_names=self.model_param.keys()
            for i in model_names:
                self.model_name=i
                self.run_model(**kwargs)
        else:
            self.run_model(**kwargs)
                
               
        
        
             
           