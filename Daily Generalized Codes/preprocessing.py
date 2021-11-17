import json
import pickle
import holidays
import collections
import numpy as np
import pandas as pd
from os import path,mkdir
from sklearn import preprocessing
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from datetime import datetime

class DataPreprocessing:
    
    """
    data_path: specify path of the file
    not_include_be_list: list of Billing Entity that we should ommit
    overall_be: specify name of overall billing entity
    aggregate_choice: specify whether you need transformation Weekly or Monthly by default its daily
    dates: date column
    self.target_value: target column for which we need to predict
    items: granural columns by which we have to do prediction
    eg:- 
    target_dates = 'posted_date'
    target_value = 'total_charge'
    target_items = 'billing_entity'
    """
    
    # STEP1: Data Initialization 
    def __init__(self,data_path,not_include_be_list,overall_be,target_dates,target_value,target_items,target_file,other_target_value,aggregate_choice,minor_holidays,major_holidays,n_periods,wf):
        self.data_path = data_path
        self.target_file = target_file
        self.other_target_value = other_target_value
        self.data = pd.DataFrame()
        self.not_include_be_list = not_include_be_list
        self.overall_be = overall_be
        self.summary_df = pd.DataFrame()
        self.transform_data = pd.DataFrame()
        self.choice = aggregate_choice
        self.target_dates = target_dates
        self.target_value = target_value
        self.target_items = target_items 
        self.minor_holidays = minor_holidays
        self.major_holidays = major_holidays
        self.n_periods = n_periods
        self.walk_forward = wf
    
    #STEP 2: Data Conistency 
    def data_consistency_date(self, data):
        resample_data = data.set_index(pd.to_datetime(data[self.target_dates])).groupby([self.target_items]).resample('D').sum().reset_index()
        return resample_data
    
    
    #STEP 2.a: combine files
    def combine_data(self):
        data = self.data_consistency_date(pd.read_csv(self.data_path))
        data[self.target_dates] = pd.to_datetime(data[self.target_dates])
        data.sort_values(self.target_dates).reset_index(drop=True,inplace=True)       
        self.data = data.copy()
        return data
    
    
    #STEP 3: Create Overall Billing Entity
    def create_overall_be(self, data_update):
        new_df = data_update[~(data_update[self.target_items].isin(self.not_include_be_list))]
        all_advh = new_df.copy()
        all_advh[self.target_items] = self.overall_be 
        full_df = new_df.append(all_advh)
        return full_df
    
    
    #STEP 4: Create Granurlar columns
    def create_granular_col(self, overall_be_data):
        overall_be_data[self.target_dates] = pd.to_datetime(overall_be_data[self.target_dates])
        overall_be_data = overall_be_data.sort_values(self.target_dates).reset_index(drop=True)
        overall_be_data['Month'] = overall_be_data[self.target_dates].dt.month
        overall_be_data['Year'] = overall_be_data[self.target_dates].dt.year
        overall_be_data['Day'] = overall_be_data[self.target_dates].dt.day
        overall_be_data['Weekday'] = overall_be_data[self.target_dates].dt.weekday_name
        overall_be_data['Weekday_num'] = overall_be_data[self.target_dates].dt.weekday
        overall_be_data['MonthYear'] = overall_be_data['Month'].astype('str')+ '-' + overall_be_data['Year'].astype('str')
        overall_be_data['WeekNum'] = overall_be_data[self.target_dates].dt.week
        return overall_be_data


    
    #Update summary file
    def update_summary(self, data, summary_df):
        #Sum of total charges per billing entity
        sum_data  = data.groupby([self.target_items, 'MonthYear'])[self.target_value].sum().reset_index()
        mean_data = pd.DataFrame(sum_data.groupby(self.target_items)[self.target_value].mean())    
         
        
        mean_data.drop(self.overall_be, axis=0, inplace=True)
        mean_data.rename(columns={self.target_value: 'monthly_sum'}, inplace=True)
        mean_data.reset_index(inplace=True)
    

        q25 =  mean_data.quantile(0.25)[0]
        q75 =  mean_data.quantile(0.75)[0]

        
        summary_df['type_be'] = ""
        
        all_be_list = list(mean_data[self.target_items].unique())
        
        for BE in all_be_list:
            subset_sum = mean_data[mean_data[self.target_items] == BE]['monthly_sum'].values[0]
            
            # If sum of be greater than q3 then large
            if subset_sum > q75:
                summary_df.loc[summary_df[self.target_items] == BE, ['type_be']] = 'Large'
            
            # Else if sum of be between q2 & q3 then medium
            elif q25 < subset_sum <= q75:
                summary_df.loc[summary_df[self.target_items] == BE, ['type_be']] = 'Medium'
            # Else: sum of be less than q2 then small
            else:
                summary_df.loc[summary_df[self.target_items] == BE, ['type_be']] = 'Small'
           
        summary_df.sort_values('type_be', inplace=True)
        summary_df.reset_index(inplace = True, drop=True)
        return data, summary_df
    
    
    #STEP 5: Check for partial data
    def remove_partial_data(self, modified_df, cut_off_day_start = 10 ):
        """
        Here we check for first and last month that has partial data, in case
        of partial data we will remove them and move it next month for first date
        
        if the Day is greater than cut_off_day_start then we are conisdering it is partial data
        """
       
        year = np.sort(list(set(modified_df.Year)))

        # Cut off Dates
        cut_off_day_start = 10            
        
        #Subset data with respect to year
        start_year_data = modified_df[modified_df.Year == year[0]].sort_values(self.target_dates).reset_index(drop=True) # start year data
        
        start_date = start_year_data[self.target_dates].iloc[0]      
        if start_date.day > cut_off_day_start:
            if (len(start_year_data.Month.unique())>1) & (start_date.month !=12):            
                start_date = start_date+pd.offsets.MonthBegin()
            else:
                start_date = start_date+pd.offsets.MonthBegin()
               
        return start_date
    
    
    def subset_data(self, data):
        summary_df = pd.DataFrame()
        new_full_df = pd.DataFrame()
        all_be_list = data[self.target_items].unique().tolist()
        
        for BE in all_be_list:
            subset_df = data[data[self.target_items]==BE]
            start_date = self.remove_partial_data(subset_df)    
            if (start_date != subset_df[self.target_dates].min()):
                mask = subset_df[self.target_dates] >= start_date
                subset_df = subset_df.loc[mask].sort_values(by=self.target_dates).reset_index(drop=True)
            if len(subset_df)!=0:
                summary_df = summary_df.append(pd.DataFrame([BE, subset_df.iloc[0][self.target_dates]]).T)
                new_full_df = new_full_df.append(subset_df)
        
        summary_df.columns = [self.target_items,'start_date']
        summary_df.reset_index(drop=True,inplace=True)
        new_full_df = new_full_df.sort_values(by=self.target_dates).reset_index(drop=True)
        data = new_full_df
        
        return data, summary_df
    
    
    def transformation(self,subset_df, trans_str, BE): 
        transformed_sum = subset_df.resample(trans_str).agg({self.target_value: 'sum'})
        transformed_sum[self.target_items] = BE
        return transformed_sum


    def get_monthly_agg(self, data, choice):
        transformed_data = pd.DataFrame()
        for BE in data[self.target_items].unique().tolist():   
            subset_df = data[data[self.target_items]==BE][[self.target_dates, self.target_value]]
            subset_df.set_index(self.target_dates, inplace=True)
            trans_str = ''
            if choice == "Monthly":
                trans_str = 'M'
            elif choice == "Daily":
                trans_str = 'D'
            transformed_sum = self.transformation(subset_df, trans_str, BE)
            transformed_data = transformed_data.append(transformed_sum)
        return transformed_data


    # DIFF MONTH
    def diff_month(self,d1, d2):
        return (d1 - d2).days


    # RECENT DAYS MISSING
    def recent_days_missing(self, df_date):
        recent = df_date.end_date.max()    
        days_miss = []
        for i in df_date.end_date:
            a = self.diff_month(recent, i)
            days_miss.append(a)
        df_date['days_missing'] = days_miss
        return df_date


    # NEG CHARGES COUNT
    def negative_charges_count(self, data_agg, data):
        neg_charges_be = data_agg[data_agg[self.target_value]<0]
        neg_be = list(neg_charges_be[self.target_items])
        counter = collections.Counter(neg_be)
        data["Count_neg"] = ''
        data["Count_neg"] = [counter[x] if x in counter.keys()  else 0 for x in data[self.target_items]]
        return data


    #ZERO DATA PERCENTAGE
    def zero_data_percentage(self, data_agg, data):
        zero_charges = data_agg[data_agg[self.target_value] == 0]
        zero_charges_be = list(zero_charges[self.target_items])
        counter = collections.Counter(zero_charges_be)
        data["zero_data_percent"] = ''
        data["zero_data_percent"] = [round(counter[x]/data[data[self.target_items] == x]["days_count"].values[0],2)*100 if x in counter.keys()  else 0 for x in data[self.target_items]]
        return data

    #Condition 1:
    def condition1(self,zero_data_df, all_be_df, current_data_summary):
        all_be_df.drop(list(zero_data_df[zero_data_df.days_count < 230][self.target_items]), axis=0, inplace=True)
        current_data_summary.drop(list(zero_data_df[zero_data_df.days_count < 230][self.target_items]), axis=0, inplace=True)
        current_data_summary = current_data_summary.reset_index()
        return all_be_df, current_data_summary
    
    
    #Condition 2:
    def condition2(self,all_be_df, current_data_summary):
        all_be_df.drop(list(current_data_summary[current_data_summary.days_missing > 0][self.target_items]), axis=0, inplace=True)
        drop_list = list(current_data_summary[current_data_summary.days_missing > 0][self.target_items])
        current_data_summary = current_data_summary.set_index(self.target_items)
        current_data_summary.drop(drop_list, axis=0, inplace=True)
        current_data_summary = current_data_summary.reset_index()
        return all_be_df, current_data_summary
    
    
    #Condition 3:
    def condition3(self,all_be_df, current_data_summary):
        all_be_df.drop(list(current_data_summary[current_data_summary.zero_data_percent > 90][self.target_items]), axis=0, inplace=True)
        drop_list = list(current_data_summary[current_data_summary.zero_data_percent > 90][self.target_items])
        current_data_summary = current_data_summary.set_index(self.target_items)
        current_data_summary.drop(drop_list, axis=0, inplace=True)
        all_be_df.reset_index(inplace=True)
        return all_be_df, current_data_summary


    #US Holidays
    def usa_holidays(self):
        us_holidays=[]
        for date in holidays.UnitedStates(years=[2017,2018,2019,2020]).items():
            us_holidays.append([str(date[0]),date[1]])
        us_holidays = pd.DataFrame(us_holidays,columns=[self.target_dates,'holiday'])
        us_holidays[self.target_dates] = pd.to_datetime(us_holidays[self.target_dates])
        us_holidays.holiday = us_holidays.holiday.astype(str)
        us_holidays['flag'] = 0
        us_holidays['flag'] = [1 if (i in self.minor_holidays) else 0 for i in us_holidays.holiday.astype("str").values]#'Martin Luther King, Jr. Day' or i=="Washington's Birthday" or i=='Columbus Day' or i=='Veterans Day'
        us_holidays['flag1'] = 0
        us_holidays['flag1'] = [1 if (i in self.major_holidays) else 0 for i in us_holidays.holiday.astype("str").values]#=="New Year's Day" or i=="Christmas Day" or i=='Thanksgiving' or i=='Memorial Day' or i=='Labor Day' or i=='Independence Day'
        us_holidays['flag2'] = 0
        us_holidays['flag2'] = [1 if (i.endswith("(Observed)")) else 0 for i in us_holidays.holiday.astype("str").values]
        return us_holidays


    #MAKE EXTENDED HOLIDAYS
    def make_extended(self, new_df_monthly, us_holidays):
        new_df_monthly = new_df_monthly.set_index(self.target_dates)
        new_df_monthly['is_extended'] = 0
        for i in new_df_monthly.index:
            if i in list(us_holidays[self.target_dates]):
                if i.weekday()==1:
                    j = i-pd.tseries.offsets.Day(1)
                    new_df_monthly['is_extended'].loc[j] = 1
                elif i.weekday()==3 :
                    j = i+pd.tseries.offsets.Day(1)
                    new_df_monthly['is_extended'].loc[j] = 1
        return new_df_monthly
    
    
    #MARK ALL HOLIDAYS
    def mark_all_holidays(self, new_df_monthly, us_holidays):
        holiday_dates = new_df_monthly[new_df_monthly.is_extended == 1].index
        us_holidays = us_holidays.set_index(self.target_dates)
        holiday_dates = holiday_dates.append(us_holidays.index)
        new_df_monthly['all_holidays'] = [1 if i in holiday_dates else 0 for i in new_df_monthly.index]
        return new_df_monthly     


    def mark_all_last_days(self, new_df_monthly):
        new_df_monthly['is_last_3_days'] = [1 if (new_df_monthly.Is_Month_End.loc[i]==1 or new_df_monthly.LastButOneDay.loc[i]==1 
                                                  or new_df_monthly.LastSecDay.loc[i]==1)  else 0 for i in new_df_monthly.index]
        return new_df_monthly



    def outlier_detection(self, temp_df, k):
        weekday_df = temp_df[(temp_df[self.target_dates].dt.weekday == k) & (temp_df.all_holidays!=1) & (temp_df.is_last_3_days!=1)]
        q1 = weekday_df[self.target_value].quantile(0.25)
        q3 = weekday_df[self.target_value].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 2 * iqr
        upper_bound = q3 + 2 * iqr   
        outlier_indices_list = list(weekday_df.index[weekday_df[self.target_value] > upper_bound])
        outlier_indices_list.extend(list(weekday_df.index[weekday_df[self.target_value] < lower_bound]))   
        if(len(outlier_indices_list)>0):
            temp_df.loc[outlier_indices_list, 'Out_Flag'] = 1
            temp_df = self.outlier_imputation(temp_df, k)
        return temp_df

 
    def outlier_imputation(self, temp_df, k):
        weekday_df = temp_df[(temp_df[self.target_dates].dt.weekday == k) & (temp_df.all_holidays!=1) & (temp_df.is_last_3_days!=1) ]
        median = weekday_df[weekday_df["Out_Flag"] == 0][self.target_value].quantile(0.50)
        temp_df.loc[temp_df["Out_Flag"]==1, self.target_value] = median
        temp_df["Out_Flag"] = 0
        return temp_df


    def outliers_helper(self, subset_df):
        subset_df = subset_df.reset_index()
        subset_df["Out_Flag"] = 0
        
        num_of_splits = round(len(subset_df)/365)
        num_of_splits = num_of_splits if num_of_splits > 0 else 1
    
        for j in range(num_of_splits):
            if(j!=num_of_splits-1):
                temp_df = subset_df[j*365 : (j+1)*365]
                for k in range(7):
                    temp_df = self.outlier_detection(temp_df, k)
                
            else:
                temp_df = subset_df[j*365:]
                for k in range(7):
                    temp_df = self.outlier_detection(temp_df, k)
        return subset_df


    def remove_last_days(self, new_df_monthly):
        for i in new_df_monthly.index:
            if (new_df_monthly.is_last_3_days.loc[i]==1 and new_df_monthly.major_holiday.loc[i]==1):
                new_df_monthly.Is_Month_End.loc[i]=0
                new_df_monthly.LastSecDay.loc[i]=0
                new_df_monthly.LastButOneDay.loc[i]=0
        return new_df_monthly


    def last_day_outlier_detection(self, new_df_monthly):
        new_df_monthly=new_df_monthly.reset_index()
        last_day_data=new_df_monthly[new_df_monthly.is_last_3_days == 1][[self.target_value]]
        q1 = last_day_data[self.target_value].quantile(0.25)
        q3 = last_day_data[self.target_value].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 2 * iqr
        upper_bound = q3 + 2 * iqr
        new_df_monthly["Out_Flag"]=0
        last_day_data["Out_Flag"]=0
        outlier_indices_list = list(last_day_data.index[last_day_data[self.target_value] > upper_bound])
        outlier_indices_list.extend(list(last_day_data.index[last_day_data[self.target_value] < lower_bound]))   
        if(len(outlier_indices_list)>0):
            new_df_monthly.loc[outlier_indices_list, 'Out_Flag'] = 1
            last_day_data.loc[outlier_indices_list, 'Out_Flag'] = 1
            median = last_day_data[last_day_data["Out_Flag"] == 0][self.target_value].quantile(0.50)
            new_df_monthly.loc[new_df_monthly["Out_Flag"]==1, self.target_value] = median
        return new_df_monthly

    def holiday_outlier_detection(self, new_df_monthly):
        new_df_monthly=new_df_monthly.reset_index(drop=True)
        holiday_data=new_df_monthly[new_df_monthly.all_holidays == 1][[self.target_value]]
        q1 = holiday_data[self.target_value].quantile(0.25)
        q3 = holiday_data[self.target_value].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 2 * iqr
        upper_bound = q3 + 2 * iqr
        new_df_monthly["Out_Flag"]=0
        holiday_data["Out_Flag"]=0
        outlier_indices_list = list(holiday_data.index[holiday_data[self.target_value] > upper_bound])
        outlier_indices_list.extend(list(holiday_data.index[holiday_data[self.target_value] < lower_bound]))   
        if(len(outlier_indices_list)>0):
            new_df_monthly.loc[outlier_indices_list, 'Out_Flag'] = 1
            holiday_data.loc[outlier_indices_list, 'Out_Flag'] = 1
            median = holiday_data[holiday_data["Out_Flag"] == 0][self.target_value].quantile(0.50)
            new_df_monthly.loc[new_df_monthly["Out_Flag"]==1, self.target_value] = median
        return new_df_monthly

    
    # GETS RAW DATA
    def  get_raw_data(self):
        if len(self.data)==0:
            return self.combine_data()
        return self.data

    
    # GETS Transformed data
    def get_transformed_data(self):
        """Gets transformed data returns summary_df & transformed data<br>
        
        transformed data: contains cleaned and transformed data
        summary_df: contains summary of data for each billing entity"""
        if len(self.transform_data)==0:
            self.transform()
        return self.summary_df,self.transform_data
    
    
    # Aggregate data monthly or weekly    
    def aggregate_data(self,data):
        if self.choice == "Monthly":
            trans_str='M'
        elif self.choice == "Weekly":
            trans_str='W'
        else:
            trans_str='D'
        return data[[self.target_dates, self.target_items, self.target_value]+self.other_target_value].set_index(pd.to_datetime(data[self.target_dates])).groupby([ self.target_items]).resample(trans_str).sum().reset_index()
    
    
    # Outlier detection & Negative charge count
    def count_outlier_negative(self,transformed_data,summary_df):
        g = transformed_data.groupby(by=self.target_items).apply(lambda row: row.quantile([.75,.25])).reset_index()
        a = g.pivot(index=self.target_items,columns='level_1',values=self.target_value)
        a.columns = ['Q1','Q3']
        a['IQR'] = a.Q3-a.Q1
        range_array = np.arange(1.5,2.5,.5)
        for i in range_array:     
            a['lower_bound_'+str(i)] = a.Q1-(a.IQR*i)
            a['upper_bound_'+str(i)] = a.Q3+(a.IQR*i)
        g = transformed_data.groupby(by=self.target_items)
        b =pd.merge(a,g[self.target_value].apply(lambda row: row.values),how='left',on=self.target_items).reset_index()
        b['negative_counts'] = b[self.target_value].apply(lambda row: len(row[row<0]))
        summary_df = pd.merge(summary_df,b[[self.target_items,'negative_counts']],how='left',on=self.target_items)
        for i in range_array:               
            b['count_outliers_'+str(i)] = b[[self.target_value,'upper_bound_'+str(i),'lower_bound_'+str(i)]].apply(lambda row: len(np.where(row[self.target_value]>row['upper_bound_'+str(i)])[0])+len(np.where(row[self.target_value]<row['lower_bound_'+str(i)])[0]),axis=1)
            summary_df = pd.merge(summary_df,b[[self.target_items,'count_outliers_'+str(i)]],how='left',on=self.target_items)
        
        return summary_df
    
    #create config file
    def create_config(self,obj_file):
        with open('data-config.json') as f:
            config_file = json.load(f)
        config_file['raw_data_path'] = self.data_path.replace("\\","/")
        config_file['processed_data_path'] = obj_file
        with open('model-config.json',"w") as f:
            f.write(json.dumps(config_file))
        f.close()
            
    
    #Save files
    def save_transformed_files(self):
        if len(self.transform_data)==0:
            self.transform()
        
        if self.walk_forward == "True":
            save_path ='data/wf_data/'
        else:
            save_path ='data/prod_data/'
            
        data_date = self.summary_df.end_date.max().month_name()+'-'+str(self.summary_df.end_date.max().year)
        if path.exists(save_path+self.overall_be)==False:
            mkdir(save_path+self.overall_be)    
        obj_file = save_path+self.overall_be+'/'+data_date+'.pkl'
        file =  open(obj_file,'wb')
        pickle.dump(self,file)
        file.close()
        self.create_config(obj_file)
   
    
    def all_model_combined_pd(self, BE_df, BE):
        print("in all_pd")
        outlier_data, test = BE_df[:-self.n_periods], BE_df[-self.n_periods:]
        outlier_data = self.outliers_helper(outlier_data)
        outlier_data = outlier_data.set_index(self.target_dates)
        outlier_data = self.remove_last_days(outlier_data)
        outlier_data = self.last_day_outlier_detection(outlier_data)
        outlier_data = self.holiday_outlier_detection(outlier_data)
        outlier_data = outlier_data.set_index(self.target_dates)

        outlier_data=outlier_data.drop(['all_holidays','Out_Flag','is_last_3_days'],axis=1)
        test=test.drop(['all_holidays','is_last_3_days'],axis=1)
         
        data=pd.concat([outlier_data,test])
        cols=data.columns
        cols=cols.drop(self.target_value)
        min_max_scaler=preprocessing.MinMaxScaler()
        normalized_data = min_max_scaler.fit_transform(data[cols])
        normalized_data=pd.DataFrame(normalized_data)
        normalized_data.columns=cols
        
        data=data.drop(cols,axis=1)
        data=data.reset_index()
        normalized_data[self.target_dates]=data[self.target_dates]
        
        transformed_data=pd.merge(data,normalized_data,on=self.target_dates).set_index(self.target_dates)        
        return transformed_data
     
    def daterange(self,date1, date2):
        for n in range(int ((date2 - date1).days)):
            yield date1 + timedelta(n+1)
            
    def main_method_pd(self, all_be_df, us_holidays):
        print("in main_pd")
        final_df = pd.DataFrame()
        for BE in all_be_df[self.target_items].unique().tolist():    
            BE_df = all_be_df[all_be_df[self.target_items] == BE]
            BE_df = BE_df[[self.target_dates, self.target_value]].set_index(self.target_dates).resample("D").sum().reset_index().sort_values(self.target_dates)       
            
            last_month = BE_df[self.target_dates].loc[BE_df.index[-1]].month
            last_year = BE_df[self.target_dates].loc[BE_df.index[-1]].year
            last_day = BE_df[self.target_dates].loc[BE_df.index[-1]].day
            date1=datetime(last_year,last_month,last_day)
            date2=date1+relativedelta(days=self.n_periods)
            dates_list=[]
            for dt in self.daterange(date1, date2):
                dates_list.append(dt.strftime("%Y-%m-%d"))
            BE_df=BE_df.append(pd.DataFrame(dates_list,columns=["posted_date"]),ignore_index=True)
            BE_df['posted_date']=pd.to_datetime(BE_df['posted_date'])
            
            BE_df["minor_holiday"] = [1 if (val in list(us_holidays[self.target_dates]) and us_holidays[us_holidays[self.target_dates]==val]["flag"].values[0]==1) else 0 for val in BE_df[self.target_dates]] 
            BE_df["major_holiday"] = [1 if (val in list(us_holidays[self.target_dates]) and us_holidays[us_holidays[self.target_dates]==val]["flag1"].values[0]==1) else 0 for val in BE_df[self.target_dates]]
            BE_df["observed_holiday"] = [1 if (val in list(us_holidays[self.target_dates]) and us_holidays[us_holidays[self.target_dates]==val]["flag2"].values[0]==1) else 0 for val in BE_df[self.target_dates]]
    
            BE_df["Is_Month_End"] = BE_df[self.target_dates].dt.is_month_end
            BE_df["LastButOneDay"] = BE_df[self.target_dates].dt.days_in_month - BE_df[self.target_dates].dt.day
            BE_df["LastSecDay"] = BE_df[self.target_dates].dt.days_in_month - BE_df[self.target_dates].dt.day
            BE_df["Is_Month_End"] = [1 if BE_df.loc[i]["Is_Month_End"]==True else 0 for i in range(len(BE_df))]
            BE_df["LastButOneDay"] = [1 if BE_df.loc[i]["LastButOneDay"]==1 else 0 for i in range(len(BE_df))]
            BE_df["LastSecDay"] = [1 if BE_df.loc[i]["LastSecDay"]==2 else 0 for i in range(len(BE_df))]
        
            BE_df = self.make_extended(BE_df, us_holidays)
            BE_df = self.mark_all_holidays(BE_df, us_holidays)
            BE_df = self.mark_all_last_days(BE_df)
        
            BE_df['sin365'] = np.sin(2 * np.pi * BE_df.index.dayofyear / 365.25)
            BE_df['cos365'] = np.cos(2 * np.pi * BE_df.index.dayofyear / 365.25)
            
            BE_df = BE_df.reset_index()
            BE_df.set_index(self.target_dates, inplace=True)
        
            BE_df['13th_Day']= [1 if x==13 else 0 for x in BE_df.index.day]
            BE_df['31st_Day']= [1 if x==31 else 0 for x in BE_df.index.day]
            print(BE)
            BE_df = self.all_model_combined_pd(BE_df, BE)
            BE_df[self.target_items] = BE
            final_df = final_df.append(BE_df)
        return final_df
    
    
    def all_model_combined_wf(self, BE_df, BE):
        print("in all_wf")
        BE_df = BE_df.reset_index()
    
        count_data_points = len(BE_df)
        n_steps_in, n_steps_out = int(count_data_points-155), 92
    
        c = 0
        i = 0

        while True:
            
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
        
            if out_end_ix > len(BE_df):
                break
        
            c = c + 1
        
            #detect and remove outliers
            outlier_data, test = BE_df[i:end_ix], BE_df[end_ix:out_end_ix]
            outlier_data = self.outliers_helper(outlier_data)
            outlier_data = outlier_data.set_index(self.target_dates)
            outlier_data = self.remove_last_days(outlier_data)
            outlier_data = self.last_day_outlier_detection(outlier_data)
            outlier_data = self.holiday_outlier_detection(outlier_data)
            outlier_data = outlier_data.append(test)
            outlier_data[self.target_dates] = pd.to_datetime(outlier_data[self.target_dates])
            outlier_data = outlier_data.sort_values(self.target_dates, ascending=True)

            BE_df["Set_"+str(c)] = 0
            start_date = outlier_data[self.target_dates].min()
            end_date = outlier_data[self.target_dates].max()
            
            mask1 = (BE_df[self.target_dates] >= start_date) & (BE_df[self.target_dates] <= end_date)
            BE_df.loc[mask1, "Set_"+str(int(c))] = list(outlier_data[self.target_value])
            print(c)        
            i=i+7
            
        return  BE_df
      

    def main_method_wf(self, all_be_df, us_holidays):
        print("in main_wf")
        final_df = pd.DataFrame()
        for BE in all_be_df[self.target_items].unique().tolist():    
            BE_df = all_be_df[all_be_df[self.target_items] == BE]
            BE_df = BE_df[[self.target_dates, self.target_value]].set_index(self.target_dates).resample("D").sum().reset_index().sort_values(self.target_dates)       
        
            BE_df["minor_holiday"] = [1 if (val in list(us_holidays[self.target_dates]) and us_holidays[us_holidays[self.target_dates]==val]["flag"].values[0]==1) else 0 for val in BE_df[self.target_dates]] 
            BE_df["major_holiday"] = [1 if (val in list(us_holidays[self.target_dates]) and us_holidays[us_holidays[self.target_dates]==val]["flag1"].values[0]==1) else 0 for val in BE_df[self.target_dates]]
            BE_df["observed_holiday"] = [1 if (val in list(us_holidays[self.target_dates]) and us_holidays[us_holidays[self.target_dates]==val]["flag2"].values[0]==1) else 0 for val in BE_df[self.target_dates]]
    
            BE_df["Is_Month_End"] = BE_df[self.target_dates].dt.is_month_end
            BE_df["LastButOneDay"] = BE_df[self.target_dates].dt.days_in_month - BE_df[self.target_dates].dt.day
            BE_df["LastSecDay"] = BE_df[self.target_dates].dt.days_in_month - BE_df[self.target_dates].dt.day
            BE_df["Is_Month_End"] = [1 if BE_df.loc[i]["Is_Month_End"]==True else 0 for i in range(len(BE_df))]
            BE_df["LastButOneDay"] = [1 if BE_df.loc[i]["LastButOneDay"]==1 else 0 for i in range(len(BE_df))]
            BE_df["LastSecDay"] = [1 if BE_df.loc[i]["LastSecDay"]==2 else 0 for i in range(len(BE_df))]
        
            BE_df = self.make_extended(BE_df, us_holidays)
            BE_df = self.mark_all_holidays(BE_df, us_holidays)
            BE_df = self.mark_all_last_days(BE_df)
        
            BE_df['sin365'] = np.sin(2 * np.pi * BE_df.index.dayofyear / 365.25)
            BE_df['cos365'] = np.cos(2 * np.pi * BE_df.index.dayofyear / 365.25)
            
            BE_df = BE_df.reset_index()
            BE_df.set_index(self.target_dates, inplace=True)
        
            BE_df['13th_Day']= [1 if x==13 else 0 for x in BE_df.index.day]
            BE_df['31st_Day']= [1 if x==31 else 0 for x in BE_df.index.day]
            print(BE)
            BE_df = self.all_model_combined_wf(BE_df, BE)
            BE_df[self.target_items] = BE
            BE_df=BE_df.drop(['all_holidays','is_last_3_days'],axis=1)
            final_df = final_df.append(BE_df)
        return final_df
  
        
    # Transforms raw Data
    def transform(self):
        
        data_dc = self.combine_data()
        
        #Create overall billing entity
        overall_be_data = self.create_overall_be(data_dc)
        del data_dc
        
        #Create granular columns
        granular_data = self.create_granular_col(overall_be_data)
        del overall_be_data
        
        #Remove partial data
        partial_removed_data, summary_df = self.subset_data(granular_data)
        del granular_data
        
        summary_df['days_count'] = [len(partial_removed_data[partial_removed_data[self.target_items] == BE]) for BE in summary_df[self.target_items]]
        summary_df['end_date'] = [partial_removed_data[partial_removed_data[self.target_items] == BE][self.target_dates].max() for BE in summary_df[self.target_items]]
        
        #Update Summary file with billing entity types and check for each of billing entity for minimum of 2 years
        data, summary_df = self.update_summary(partial_removed_data, summary_df)
        
        #Recent days missing
        summary_df = self.recent_days_missing(summary_df)
        
        #Zero data percentage
        zero_data_df = self.zero_data_percentage(data, summary_df)
        
        current_data_summary = zero_data_df.copy().set_index(self.target_items)
        data = data.set_index(self.target_items)
        
        #Condition-1
        data, current_data_summary = self.condition1(zero_data_df, data, current_data_summary)
        
        #Condition-2
        data, current_data_summary = self.condition2(data, current_data_summary)
        
        #Condition-3
        data, current_data_summary = self.condition3(data, current_data_summary)
        
        us_holidays = self.usa_holidays()
        
        print("wf_cond",self.walk_forward)
        if self.walk_forward == "True":
            final_df = self.main_method_wf(data, us_holidays)
        else:
            final_df = self.main_method_pd(data, us_holidays)


        self.summary_df = current_data_summary
        self.transform_data = final_df
        del partial_removed_data
        del summary_df