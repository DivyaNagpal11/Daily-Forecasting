# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:57:07 2020

@author: DN067571
"""
import pandas as pd

def generate_best_model():
    wf_file=pd.read_csv("daily_walk_forward_for_all_be2.csv")
    be_names=list(wf_file.billing_entity.unique())
    model_names=list(wf_file.model.unique())
    best_model_list=[]
    
    for m in be_names:
        subset_df=wf_file[wf_file.billing_entity == m]
        models_avg=[]
        for n in model_names:
            model_df=subset_df[subset_df.model == n]
            test_mape_avg = model_df.test_mape.mean()
            models_avg.append(test_mape_avg)
        min_mape=min(models_avg)
        min_mape_index=models_avg.index(min_mape)
        best_model=model_names[min_mape_index]
        best_model_list.append([m,best_model,round(min_mape,2)])
        
    best_model_list=pd.DataFrame(best_model_list,columns=["be_name","best_model","mape_avg_min"])
    best_model_list.to_csv("daily_best_model_feb_with_reg.csv")
    return best_model_list    
