# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:16:03 2020

@author: DN067571
"""

import pandas as pd
import numpy as np
from datetime import datetime
from plotly.offline import plot
import collections
from itertools import product
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import time
import holidays
import daily_sarima_code as DSA
import daily_sarimax_code as DSX
import daily_prophet_code as DPH
import warnings
warnings.filterwarnings('ignore')
from dateutil.relativedelta import relativedelta
import plotly
import plotly.graph_objs as go
import sys
import os
from sklearn import preprocessing
import tqdm
from itertools import chain

########### Get Data #################################################
def get_data(data_path,not_include_be_list):
    data_charges = pd.read_csv(data_path)
    data_charges = data_charges[~(data_charges['billing_entity'].isin(not_include_be_list))]
    return data_charges    


##################### Make data Consistent ##########################
    
def data_consistency_date(subset_df,BE):
    subset_df['posted_date']=pd.to_datetime(subset_df['posted_date'])
    subset_df=subset_df.set_index("posted_date")
    subset_df=subset_df.resample('D').agg({'total_charge': 'sum'})
    subset_df['billing_entity']=BE
    return subset_df

############## Create Overall BE ################################
    
def create_overall_be(new_df,overall_be,not_include_be_list):
    new_df1 = new_df[~(new_df['billing_entity'].isin(not_include_be_list))]
    new_df1['billing_entity']= overall_be
    full_df=new_df.append(new_df1)
    return full_df


############# Create  Granular Columns ########################
    
def create_granular_col(df):
    df['posted_date']=pd.to_datetime(df['posted_date'])
    df= df.sort_values('posted_date').reset_index(drop=True)
    df['Month']= df['posted_date'].dt.month
    df['Year']= df['posted_date'].dt.year
    df['Day']= df['posted_date'].dt.day
    df['Weekday']= df['posted_date'].dt.weekday_name
    df['Weekday_num']= df['posted_date'].dt.weekday
    df['MonthYear']= df['Month'].astype('str')+ '-' + df['Year'].astype('str')
    df['WeekNum']= df['posted_date'].dt.week
    return df


################# Remove Partial Data ##########################
def remove_partial_data(modified_df):
    year = np.sort(list(set(modified_df.Year)))
    # Cut off Dates
    cut_off_day_start = 10 #if the Day is greater than cut_off_day_start then we are conisdering it is partial data
    cut_off_day_end = 28   #if the Day is greater than cut_off_day_end then we are conisdering it is partial data
    
    #Subset data with respect to year
    start_year_data = modified_df[modified_df.Year == year[0]].sort_values('posted_date').reset_index(drop=True) # start year data
    end_year_data = modified_df[modified_df.Year == year[-1]].sort_values('posted_date').reset_index(drop=True) # end year data
    
    start_date = start_year_data.posted_date.iloc[0]
    if (len(start_year_data.Month.unique())>1) & (start_date.month !=12):
        if start_date.day > cut_off_day_start:
            start_date = start_date+pd.offsets.MonthBegin()
    else:
        if start_date.day > cut_off_day_start:
            start_date = start_date+pd.offsets.MonthBegin()
        
    end_date = end_year_data.posted_date.iloc[-1]
    
    if end_date.month !=1:
        if end_date.day < cut_off_day_end:
            end_date = end_date-pd.offsets.MonthEnd()
    else:
        if end_date.day < cut_off_day_end:
            end_date = end_date-pd.offsets.MonthEnd()
       
    return start_date,end_date


def subset_data(full_df):
    df_date = pd.DataFrame()
    new_full_df = pd.DataFrame()
    all_be_list = list(sorted(set(full_df.billing_entity)))
    for i in all_be_list:
        modified_df = full_df[full_df.billing_entity==i]
        start_date,end_date = remove_partial_data(modified_df)    
        mask = (modified_df['posted_date'] >= start_date) & (modified_df['posted_date'] <= end_date)
        subset_df= modified_df.loc[mask]
        subset_df = subset_df.sort_values(by='posted_date').reset_index(drop=True)
        if len(subset_df)!=0:
            df_date = df_date.append(pd.DataFrame([i,subset_df.iloc[0]['posted_date'],subset_df.iloc[-1]['posted_date']]).T)
            new_full_df = new_full_df.append(subset_df)
    df_date.columns=['billing_entity','start_date','end_date']
    df_date.reset_index(drop=True,inplace=True)
    new_full_df = new_full_df.sort_values(by='posted_date').reset_index(drop=True)
    return new_full_df,df_date


#################### Trnasformation ###############
    
def transformation(subset_df,trans_str,i): 
    transformed_sum=subset_df.resample(trans_str).agg({'total_charge': 'sum'})
    transformed_sum['billing_entity']=i
    return transformed_sum
   
    
    
def get_monthly_agg(new_full_df,choice):
    transformed_data = pd.DataFrame()
    for i in new_full_df.billing_entity.unique():   
        subset_df = new_full_df[new_full_df.billing_entity==i][['posted_date','total_charge']]
        subset_df.set_index('posted_date',inplace=True)
        trans_str=''
        if choice == "Monthly":
            trans_str='M'
        elif choice == "Daily":
            trans_str='D'
        transformed_sum = transformation(subset_df,trans_str,i)
        transformed_data = transformed_data.append(transformed_sum)
    return transformed_data

    
##################### Finding the type of BE ###########################     
    
def be_type(new_full_df, df_date):
    df = pd.DataFrame()
    df = new_full_df.groupby(['billing_entity', 'MonthYear'])['total_charge'].sum().reset_index()
    df = pd.DataFrame(df.groupby('billing_entity')['total_charge'].mean())
    df.drop('All Adventist W', axis=0, inplace=True)
    df.rename(columns={'total_charge': 'monthly_sum'}, inplace=True)
    df.reset_index(inplace=True)
    
    df1 = pd.DataFrame(df.describe())
    df1.columns = ['monthly_sum_desc']
    df1.reset_index(inplace=True)

    all_be_monthly_res = df1.values.tolist()
    
    df_date['type_be'] = ""
    all_be_list = list(sorted(set(df.billing_entity)))   
    for i in all_be_list:
        subset_sum = df[df.billing_entity == i]['monthly_sum'].values[0]
        # If sum of be greater than q3 then large
        if subset_sum >= all_be_monthly_res[6][1]:
            df_date.loc[df_date['billing_entity'] == i, ['type_be']] = 'Large'
        # If sum of be between q2 & q3 then medium
        elif all_be_monthly_res[4][1] < subset_sum <= all_be_monthly_res[6][1]:
             df_date.loc[df_date['billing_entity'] == i, ['type_be']] = 'Medium'
        # If sum of be less than q2 then small
        else:
             df_date.loc[df_date['billing_entity'] == i, ['type_be']] = 'Small'
                 
    df_date.sort_values('type_be', inplace=True)
    df_date.reset_index(inplace=True)
    df_date.drop("index",axis=1,inplace=True) 
    return df_date


################## Finding the BE with missing recent months & Count of months missing #################
    
def diff_month(d1, d2):
    return (d1 - d2).days


def recent_days_missing(df_date):
    recent = df_date.end_date.max()
    days_miss = []
    for i in df_date.end_date:
        a=diff_month(recent,i)
        days_miss.append(a)
    df_date['days_missing']=days_miss
    return df_date


################# Negative charges count ##########################
    
def negative_charges_count(data_agg,df_date):
    neg_charges_be=data_agg[data_agg["total_charge"]<0]
    neg_be=list(neg_charges_be.billing_entity)
    counter=collections.Counter(neg_be)
    df_date["Count_neg"]=''
    df_date["Count_neg"]=[ counter[x] if x in counter.keys()  else 0 for x in df_date.billing_entity ]
    return df_date


def zero_data_percentage(data_agg,df_date):
    zero_charges = data_agg[data_agg["total_charge"] == 0]
    zero_charges_be=list(zero_charges.billing_entity)
    counter=collections.Counter(zero_charges_be)
    df_date["zero_data_percent"]=''
    df_date["zero_data_percent"]=[ round(counter[x]/df_date[df_date.billing_entity == x]["days_count"].values[0],2)*100 if x in counter.keys()  else 0 for x in df_date.billing_entity ]
    return df_date

################################ Outlier Rejection ##############################  

def outlier_detection(temp_df, k):
    weekday_df = temp_df[(temp_df.posted_date.dt.weekday == k) & (temp_df.all_holidays!=1) & (temp_df.is_last_3_days!=1)]
    q1 = weekday_df["total_charge"].quantile(0.25)
    q3 = weekday_df["total_charge"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 2 * iqr
    upper_bound = q3 + 2 * iqr   
    outlier_indices_list = list(weekday_df.index[weekday_df["total_charge"] > upper_bound])
    outlier_indices_list.extend(list(weekday_df.index[weekday_df["total_charge"] < lower_bound]))   
    if(len(outlier_indices_list)>0):
        temp_df.loc[outlier_indices_list, 'Out_Flag'] = 1
        temp_df = outlier_imputation(temp_df, k)
    return temp_df

 
def outlier_imputation(temp_df, k):
    weekday_df = temp_df[(temp_df.posted_date.dt.weekday == k) & (temp_df.all_holidays!=1) & (temp_df.is_last_3_days!=1) ]
    median = weekday_df[weekday_df["Out_Flag"] == 0]["total_charge"].quantile(0.50)
    temp_df.loc[temp_df["Out_Flag"]==1, "total_charge"] = median
    temp_df["Out_Flag"] = 0
    return temp_df


def outliers_helper(subset_df):
    subset_df = subset_df.reset_index()
    subset_df["Out_Flag"] = 0
    
    print(subset_df.columns)
    num_of_splits = round(len(subset_df)/365)
    num_of_splits = num_of_splits if num_of_splits > 0 else 1
    
    for j in range(num_of_splits):
        if(j!=num_of_splits-1):
            temp_df = subset_df[j*365 : (j+1)*365]
            for k in range(7):
                temp_df = outlier_detection(temp_df, k)
                
        else:
            temp_df = subset_df[j*365:]
            for k in range(7):
                temp_df = outlier_detection(temp_df, k)
    return subset_df


########################### Last Day Outlier Deetection & Imputation #################

def last_day_outlier_detection(new_df_monthly):
    new_df_monthly=new_df_monthly.reset_index()
    last_day_data=new_df_monthly[new_df_monthly.is_last_3_days == 1][["total_charge"]]
    q1 = last_day_data["total_charge"].quantile(0.25)
    q3 = last_day_data["total_charge"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 2 * iqr
    upper_bound = q3 + 2 * iqr
    new_df_monthly["Out_Flag"]=0
    last_day_data["Out_Flag"]=0
    outlier_indices_list = list(last_day_data.index[last_day_data["total_charge"] > upper_bound])
    outlier_indices_list.extend(list(last_day_data.index[last_day_data["total_charge"] < lower_bound]))   
    if(len(outlier_indices_list)>0):
        new_df_monthly.loc[outlier_indices_list, 'Out_Flag'] = 1
        last_day_data.loc[outlier_indices_list, 'Out_Flag'] = 1
        median = last_day_data[last_day_data["Out_Flag"] == 0]["total_charge"].quantile(0.50)
        new_df_monthly.loc[new_df_monthly["Out_Flag"]==1, "total_charge"] = median
    return new_df_monthly


def holiday_outlier_detection(new_df_monthly):
    new_df_monthly=new_df_monthly.reset_index(drop=True)
    holiday_data=new_df_monthly[new_df_monthly.all_holidays == 1][["total_charge"]]
    q1 = holiday_data["total_charge"].quantile(0.25)
    q3 = holiday_data["total_charge"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 2 * iqr
    upper_bound = q3 + 2 * iqr
    new_df_monthly["Out_Flag"]=0
    holiday_data["Out_Flag"]=0
    outlier_indices_list = list(holiday_data.index[holiday_data["total_charge"] > upper_bound])
    outlier_indices_list.extend(list(holiday_data.index[holiday_data["total_charge"] < lower_bound]))   
    if(len(outlier_indices_list)>0):
        new_df_monthly.loc[outlier_indices_list, 'Out_Flag'] = 1
        holiday_data.loc[outlier_indices_list, 'Out_Flag'] = 1
        median = holiday_data[holiday_data["Out_Flag"] == 0]["total_charge"].quantile(0.50)
        new_df_monthly.loc[new_df_monthly["Out_Flag"]==1, "total_charge"] = median
    return new_df_monthly
############################ Make extended holidays ############################
    
def make_extended(new_df_monthly,us_holidays):
    new_df_monthly=new_df_monthly.set_index('posted_date')
    new_df_monthly['is_extended']=0
    for i in new_df_monthly.index:
        if i in list(us_holidays.posted_date):
            if i.weekday()==1:
                j=i-pd.tseries.offsets.Day(1)
                new_df_monthly['is_extended'].loc[j]=1
            elif i.weekday()==3 :
                j=i+pd.tseries.offsets.Day(1)
                new_df_monthly['is_extended'].loc[j]=1
    return new_df_monthly
    

################ Mark All Holidays ############################################

def mark_all_holidays(new_df_monthly,us_holidays):
    holiday_dates=new_df_monthly[new_df_monthly.is_extended == 1].index
    us_holidays=us_holidays.set_index('posted_date')
    holiday_dates=holiday_dates.append(us_holidays.index)
    new_df_monthly['all_holidays']=[1 if i in holiday_dates else 0 for i in new_df_monthly.index]
    return new_df_monthly


def mark_all_last_days(new_df_monthly):
    new_df_monthly['is_last_3_days']=[1 if (new_df_monthly.Is_Month_End.loc[i]==1 or new_df_monthly.LastButOneDay.loc[i]==1 or new_df_monthly.LastSecDay.loc[i]==1)  else 0 for i in new_df_monthly.index]
    return new_df_monthly
        

def remove_last_days(new_df_monthly):
    for i in new_df_monthly.index:
        if (new_df_monthly.is_last_3_days.loc[i]==1 and new_df_monthly.major_holiday.loc[i]==1):
            new_df_monthly.Is_Month_End.loc[i]=0
            new_df_monthly.LastSecDay.loc[i]=0
            new_df_monthly.LastButOneDay.loc[i]=0
    return new_df_monthly


##############################################################################
    
def all_model_combined(new_df_monthly,be_name):

    sarimax_model_res=[]
    prophet_model_res=[]
    train=[]
    val=[]
    test=[]
    
    count_data_points=len(new_df_monthly)
    n_steps_in, n_steps_out =int(count_data_points-155),92
    
    c=0
    i=0
    while True:
        end_ix=i+n_steps_in
        out_end_ix=end_ix+n_steps_out
        if out_end_ix>len(new_df_monthly):
            break
        c=c+1

        outlier_data,test = new_df_monthly[i:end_ix], new_df_monthly[end_ix:out_end_ix]

        outlier_data=outliers_helper(outlier_data)
        outlier_data=outlier_data.set_index("posted_date")
        outlier_data=remove_last_days(outlier_data)
        outlier_data=last_day_outlier_detection(outlier_data)
        outlier_data=holiday_outlier_detection(outlier_data)
        outlier_data=outlier_data.set_index("posted_date")
        train,val=outlier_data[:-46],outlier_data[-46:]
        
        train=train.drop(['all_holidays','Out_Flag','is_last_3_days'],axis=1)
        val=val.drop(['all_holidays','Out_Flag','is_last_3_days'],axis=1)
        test=test.drop(['all_holidays','is_last_3_days'],axis=1)
         
        data=pd.concat([train,val,test])
        cols=data.columns
        cols=cols.drop('total_charge')
        
        min_max_scaler=preprocessing.MinMaxScaler()
        normalized_data = min_max_scaler.fit_transform(data[cols])
        normalized_data=pd.DataFrame(normalized_data)
        normalized_data.columns=cols
        
        data=data.drop(cols,axis=1)
        data=data.reset_index()
        normalized_data["posted_date"]=data["posted_date"]
        
        transformed_data=pd.merge(data,normalized_data,on="posted_date").set_index("posted_date")
        
        train,val,test=transformed_data[:-138],transformed_data[-138:-92],transformed_data[-92:]

        train_mape_sarimax,val_mape_sarimax,test_dates,fitted_mean_sarimax,test_mape_sarimax=DSX.sarimax_model_daily(train,val,test,be_name,c)
        sarimax_model_res.append([be_name,'SARIMAX',train_mape_sarimax,val_mape_sarimax,test_dates,fitted_mean_sarimax,test_mape_sarimax,c])
            
        train_mape_prophet,val_mape_prophet,test_dates,fitted_mean_prophet,test_mape_prophet=DPH.daily_prophet(train[["total_charge","is_extended","Is_Month_End","LastButOneDay","LastSecDay",'13th_Day','31st_Day']],val[["total_charge","is_extended","Is_Month_End","LastButOneDay","LastSecDay",'13th_Day','31st_Day']],test[["total_charge","is_extended","Is_Month_End","LastButOneDay","LastSecDay",'13th_Day','31st_Day']],be_name,c)
        prophet_model_res.append([be_name,'PROPHET',train_mape_prophet,val_mape_prophet,test_dates,fitted_mean_prophet,test_mape_prophet,c])
        
        i=i+7
        
    return sarimax_model_res,prophet_model_res 


################ Report Generation ####################
def chainer(s):
    return list(chain.from_iterable(s.str.split(',')))
    
def report_file(mape_res,new_df_monthly,be_name):
    res=pd.DataFrame()
    test_predicted_values=pd.DataFrame()
    models=list(mape_res.model.unique())
    for m in models:
        subset_df=mape_res[mape_res.model == m]
        for k in range(0,10):
            start_date=subset_df["test_dates"][k][0]
            end_date=subset_df["test_dates"][k][-1]
            mask = (new_df_monthly.index >= start_date) & (new_df_monthly.index <= end_date)
            new_subset_df= ['{:0,.0f}'.format(i) for i in new_df_monthly['total_charge'].loc[mask]]
            a=subset_df["test_dates"][k][:].strftime("%d-%b-%Y")
            b=['{:0,.0f}'.format(i) for i in subset_df['test_predicted'][k][:]]
            test_predicted_values=test_predicted_values.append(b)
            lens=len(b)
            res = res.append(pd.DataFrame({'billing_entity': np.repeat(be_name, lens),
                                'set_no': np.repeat(k+1, lens),
                                'model': np.repeat(m, lens),
                                'predicted_dates':np.repeat(a[0],lens),
                                'test_dates': chainer(a),
                                'actual_charges':new_subset_df}))  #'test_predicted': chainer(b1)
    res["test_predicted"]=test_predicted_values
                            
    return res
    
########################## Main Function ####################################
def main(data_path):
    not_include_be_list=["Feather River Hospital"]
    overall_be = "All Adventist W"
    
    data_charges = get_data(data_path,not_include_be_list)
    all_be_list = list(sorted(set(data_charges.billing_entity)))
    
    new_df = pd.DataFrame()
    for i in all_be_list:
        a = data_consistency_date(data_charges[data_charges.billing_entity==i],i)
        new_df = new_df.append(a)
    new_df=new_df.reset_index()
    
    
    full_df = create_overall_be(new_df,overall_be, not_include_be_list)
    new_df = create_granular_col(full_df)
    new_full_df, df_date = subset_data(new_df)

    
    choice="Daily"
    data_agg = get_monthly_agg(new_full_df,choice)
    df_date = be_type(new_full_df,df_date)
    df_date = recent_days_missing(df_date)
    df_date = negative_charges_count(data_agg, df_date)
    df_date['days_count']=[len(data_agg[data_agg.billing_entity ==i]) for i in df_date.billing_entity]
    df_date=zero_data_percentage(data_agg,df_date)
    
    current_data_summary=df_date.copy().set_index("billing_entity")
    new_full_df=new_full_df.set_index("billing_entity")
    
    #Condition 1:
    new_full_df.drop(list(df_date[df_date.days_count < 276]['billing_entity']), axis=0, inplace=True)
    current_data_summary.drop(list(df_date[df_date.days_count < 276]['billing_entity']), axis=0, inplace=True)
    current_data_summary=current_data_summary.reset_index()
    
    #Condition 2:
    new_full_df.drop(list(current_data_summary[current_data_summary.days_missing > 1]['billing_entity']), axis=0, inplace=True)
    drop_list=list(current_data_summary[current_data_summary.days_missing > 1]['billing_entity'])
    current_data_summary=current_data_summary.set_index("billing_entity")
    current_data_summary.drop(drop_list, axis=0, inplace=True)
    current_data_summary=current_data_summary.reset_index()
    
    #Condition 3:
    new_full_df.drop(list(current_data_summary[current_data_summary.zero_data_percent > 90]['billing_entity']), axis=0, inplace=True)
    drop_list=list(current_data_summary[current_data_summary.zero_data_percent > 90]['billing_entity'])
    current_data_summary=current_data_summary.set_index("billing_entity")
    current_data_summary.drop(drop_list, axis=0, inplace=True)
    
    new_full_df.reset_index(inplace=True)
    
    
    us_holidays=[]
    for date in holidays.UnitedStates(years=[2015,2016,2017,2018,2019,2020,2021]).items():
        us_holidays.append([str(date[0]),date[1]])
    us_holidays=pd.DataFrame(us_holidays,columns=['posted_date','holiday'])
    us_holidays['posted_date']=pd.to_datetime(us_holidays["posted_date"])
    us_holidays.holiday=us_holidays.holiday.astype(str)
    us_holidays['flag']=0
    us_holidays['flag']=[1 if (i=='Martin Luther King, Jr. Day' or i=="Washington's Birthday" or i=='Columbus Day' or i=='Veterans Day') else 0 for i in us_holidays.holiday.astype("str").values]
    us_holidays['flag1']=0
    us_holidays['flag1']=[1 if (i=="New Year's Day" or i=="Christmas Day" or i=='Thanksgiving' or i=='Memorial Day' or i=='Labor Day' or i=='Independence Day') else 0 for i in us_holidays.holiday.astype("str").values]
    us_holidays['flag2']=0
    us_holidays['flag2']=[1 if (i.endswith("(Observed)")) else 0 for i in us_holidays.holiday.astype("str").values]
    
    
    csv_folder=r"csv_daily_sarimax_results/"
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
    walk_forward_file=pd.DataFrame()
    report_file_generated=pd.DataFrame()
  
    for i in new_full_df.billing_entity.unique():
        new_df_monthly=new_full_df[new_full_df.billing_entity == i]
        new_df_monthly=new_df_monthly[["posted_date","total_charge"]]
        new_df_monthly=new_df_monthly.set_index("posted_date")
        new_df_monthly = new_df_monthly.resample("D").sum()
        
        new_df_monthly=new_df_monthly.reset_index()
        new_df_monthly=new_df_monthly.sort_values('posted_date')
        
        new_df_monthly["minor_holiday"]=[1 if (val in list(us_holidays.posted_date) and us_holidays[us_holidays.posted_date==val]["flag"].values[0]==1) else 0 for val in new_df_monthly.posted_date] 
        new_df_monthly["major_holiday"]=[1 if (val in list(us_holidays.posted_date) and us_holidays[us_holidays.posted_date==val]["flag1"].values[0]==1) else 0 for val in new_df_monthly.posted_date]
        new_df_monthly["observed_holiday"]=[1 if (val in list(us_holidays.posted_date) and us_holidays[us_holidays.posted_date==val]["flag2"].values[0]==1) else 0 for val in new_df_monthly.posted_date]
        new_df_monthly["Is_Month_End"] = new_df_monthly["posted_date"].dt.is_month_end
        new_df_monthly["LastButOneDay"] = new_df_monthly["posted_date"].dt.days_in_month - new_df_monthly["posted_date"].dt.day
        new_df_monthly["LastSecDay"] = new_df_monthly["posted_date"].dt.days_in_month - new_df_monthly["posted_date"].dt.day
        new_df_monthly["Is_Month_End"] = [1 if new_df_monthly.loc[i]["Is_Month_End"]==True else 0 for i in range(len(new_df_monthly))]
        new_df_monthly["LastButOneDay"] = [1 if new_df_monthly.loc[i]["LastButOneDay"]==1 else 0 for i in range(len(new_df_monthly))]
        new_df_monthly["LastSecDay"] = [1 if new_df_monthly.loc[i]["LastSecDay"]==2 else 0 for i in range(len(new_df_monthly))]
        
        new_df_monthly=make_extended(new_df_monthly,us_holidays)
        new_df_monthly=mark_all_holidays(new_df_monthly,us_holidays)
        new_df_monthly=mark_all_last_days(new_df_monthly)
       
        new_df_monthly['sin365'] = np.sin(2 * np.pi * new_df_monthly.index.dayofyear / 365.25)
        new_df_monthly['cos365'] = np.cos(2 * np.pi * new_df_monthly.index.dayofyear / 365.25)
#        new_df_monthly['sin365_2'] = np.sin(4 * np.pi * new_df_monthly.index.dayofyear / 365.25)
#        new_df_monthly['cos365_2'] = np.cos(4 * np.pi * new_df_monthly.index.dayofyear / 365.25)
#        new_df_monthly['sin365_3'] = np.sin(6 * np.pi * new_df_monthly.index.dayofyear / 365.25)
#        new_df_monthly['cos365_3'] = np.cos(6 * np.pi * new_df_monthly.index.dayofyear / 365.25)
        
#        new_df_monthly['6th_Day']= [1 if x==6 else 0 for x in new_df_monthly.index.day]
        new_df_monthly['13th_Day']= [1 if x==13 else 0 for x in new_df_monthly.index.day]
#        new_df_monthly['20th_Day']= [1 if x==20 else 0 for x in new_df_monthly.index.day]
#        new_df_monthly['27th_Day']= [1 if x==27 else 0 for x in new_df_monthly.index.day]
        new_df_monthly['31st_Day']= [1 if x==31 else 0 for x in new_df_monthly.index.day]
        
        sarimax_model_res,prophet_model_res=all_model_combined(new_df_monthly,i)
        sarimax_model_res=pd.DataFrame(sarimax_model_res,columns=['billing_entity','model','train_mape','val_mape','test_dates','test_predicted','test_mape','set_no'])
        prophet_model_res=pd.DataFrame(prophet_model_res,columns=['billing_entity','model','train_mape','val_mape','test_dates','test_predicted','test_mape','set_no'])
        mape_res=pd.concat([sarimax_model_res,prophet_model_res])
        report_res=report_file(mape_res,new_df_monthly,i)
        report_file_generated=report_file_generated.append(report_res)
        walk_forward_file=walk_forward_file.append(mape_res)   
    walk_forward_file.to_csv("daily_walk_forward_for_all_be_june"+".csv")
    report_file_generated.to_csv("daily_june_report1.csv")
    
if __name__ == '__main__':
    data_path="new_feb_3y_charges.csv"
    main(data_path)
    