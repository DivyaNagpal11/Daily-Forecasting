#Import files


from tqdm import tqdm
import pandas as pd
import numpy as np
from fbprophet import Prophet
import warnings
warnings.filterwarnings('ignore')


#Model class
class PROPHET:

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


    def fit(self,model,train_dataset):
        return model.fit(train_dataset)


    def predict(self,model,data):
        return model.predict(data) 


    def split_train_val(self,subset_df):
        return subset_df[:(-self.n_periods/2)],subset_df[(-self.n_periods/2):]

    def mean_absolute_percentage_error(self,y_true,y_pred):
        return np.mean(np.abs(np.subtract(y_true,y_pred)/y_true))*100  

    def calculate_apes(self):
        for i,j in zip(self.test[self.target_value].values.flatten(),self.test_prediction.values.flatten()):            
            self.apes.append(self.mean_absolute_percentage_error(i,j))

    def median_absolute_percentage_error_daily(self,y_true,y_pred):
        y_true=pd.Series(y_true)
        y_pred=pd.Series(y_pred)
        true_pred = pd.DataFrame(zip(y_true,y_pred),columns=['y_true','y_pred'])
        true_pred.drop(true_pred[true_pred['y_pred'] == 0].index, axis=0, inplace=True)
        true_pred.drop(true_pred[true_pred['y_true'] == 0].index, axis=0, inplace=True)
        return np.median(np.abs(np.subtract(true_pred.y_true,true_pred.y_pred)/true_pred.y_true))*100
        
    def optimize_prophet(self,parameters_list,train_dataset,val_dataset,steps):  
        results=[]
        best_adj_mape=float('inf')
        for i in tqdm(parameters_list):
            forecast=pd.DataFrame()
            future=pd.DataFrame()
            
            prophet_basic = Prophet(growth='linear',daily_seasonality=False,weekly_seasonality=True,yearly_seasonality=True,holidays_prior_scale=10,n_changepoints=i[0],changepoint_prior_scale=i[1])
            prophet_basic.add_regressor('is_extended')
            prophet_basic.add_regressor('Is_Month_End')
            prophet_basic.add_regressor('LastButOneDay')
            prophet_basic.add_regressor('LastSecDay')
            prophet_basic.add_regressor('13th_Day')
            prophet_basic.add_regressor('31st_Day')
            prophet_basic.add_country_holidays(country_name='US')
            prophet_basic=self.fit(prophet_basic,train_dataset)
            
            future= prophet_basic.make_future_dataframe(periods=len(val_dataset))
            x=train_dataset.append(val_dataset)
            future['is_extended'] =pd.Series(x['is_extended'].values)
            future['Is_Month_End'] =pd.Series(x['Is_Month_End'].values)
            future['LastButOneDay'] =pd.Series(x['LastButOneDay'].values)
            future['LastSecDay'] =pd.Series(x['LastSecDay'].values)
            future['13th_Day'] =pd.Series(x['13th_Day'].values)
            future['31st_Day'] =pd.Series(x['31st_Day'].values)
            forecast=self.predict(prophet_basic,future)
            
            y_true=np.array(list(train_dataset['y']))
            y_pred=np.array(list(forecast.yhat[:-steps]))
            val_predicted=np.array(list(forecast.yhat[-steps:]))
            train_mape=round((self.median_absolute_percentage_error_daily(y_true[-365:],y_pred[-365:])),2)
            val_mape=round((self.median_absolute_percentage_error_daily(val_dataset["y"],val_predicted)),2)
            adj_mape = train_mape*len(y_true)/(len(y_true)+len(val_dataset))+val_mape*len(val_dataset)/(len(y_true)+len(val_dataset))
            
            if adj_mape <= best_adj_mape:
                best_adj_mape=adj_mape
                best_model = prophet_basic
                
            results.append([i,train_mape,val_mape,adj_mape])
            
        result_table=pd.DataFrame(results,columns=['parameters','train_mape','val_mape','adj_mape'])
        result_table=result_table.sort_values(by='adj_mape',ascending=True).reset_index(drop=True)
        return result_table, best_model
      


    def run_model(self):
        print("In prophet")
        train_set=self.train[:-(int(self.n_periods/2))]#.reset_index()
        val_set=self.train[-(int(self.n_periods/2)):]
        train_dataset= pd.DataFrame()
        val_dataset= pd.DataFrame()
        train_dataset['ds'] = train_set[self.target_dates]
        train_dataset['y']=train_set[self.target_value]
        train_dataset['is_extended']=train_set["is_extended"]
        train_dataset['Is_Month_End']=train_set["Is_Month_End"]
        train_dataset['LastButOneDay']=train_set["LastButOneDay"]
        train_dataset['LastSecDay']=train_set["LastSecDay"]
        train_dataset['13th_Day']=train_set["13th_Day"]
        train_dataset['31st_Day']=train_set["31st_Day"]
        val_dataset['ds'] = val_set[self.target_dates]
        val_dataset['y']=val_set[self.target_value]
        val_dataset['is_extended']=val_set["is_extended"]
        val_dataset['Is_Month_End']=val_set["Is_Month_End"]
        val_dataset['LastButOneDay']=val_set["LastButOneDay"]
        val_dataset['LastSecDay']=val_set["LastSecDay"]
        val_dataset['13th_Day']=val_set["13th_Day"]
        val_dataset['31st_Day']=val_set["31st_Day"]
         
        result_table, best_model = self.optimize_prophet(self.parameters_list,train_dataset,val_dataset,int(self.n_periods/2))
        future2= best_model.make_future_dataframe(periods=int(self.n_periods/2))
        x=train_dataset.append(val_dataset)
        future2['is_extended'] =pd.Series(x['is_extended'].values)
        future2['Is_Month_End'] =pd.Series(x['Is_Month_End'].values)
        future2['LastButOneDay'] =pd.Series(x['LastButOneDay'].values)
        future2['LastSecDay'] =pd.Series(x['LastSecDay'].values)
        future2['13th_Day'] =pd.Series(x['13th_Day'].values)
        future2['31st_Day'] =pd.Series(x['31st_Day'].values)
        forecast_val=self.predict(best_model,future2)
        forecast_val=forecast_val.yhat[-(int(self.n_periods/2)):]

        overall_train=self.train
        overall_train['ds'] = self.train[self.target_dates]
        overall_train['y'] = self.train[self.target_value]
        fitted_val_list=[]

        c=1
        for ncp,cp in result_table.parameters:
            try:
                if c > 3:
                    break
                prophet_basic1 = Prophet(growth='linear',daily_seasonality=False,weekly_seasonality=True,yearly_seasonality=True,holidays_prior_scale=10,n_changepoints=ncp,changepoint_prior_scale=cp)
                prophet_basic1.add_regressor('is_extended')
                prophet_basic1.add_regressor('Is_Month_End')
                prophet_basic1.add_regressor('LastButOneDay')
                prophet_basic1.add_regressor('LastSecDay')
                prophet_basic1.add_regressor('13th_Day')
                prophet_basic1.add_regressor('31st_Day')
                prophet_basic1.add_country_holidays(country_name='US')
                prophet_basic1.fit(overall_train)
                future1= prophet_basic1.make_future_dataframe(periods=len(self.test))
                x=overall_train.append(self.test)
                future1['is_extended'] =pd.Series(x['is_extended'].values)
                future1['Is_Month_End'] =pd.Series(x['Is_Month_End'].values)
                future1['LastButOneDay'] =pd.Series(x['LastButOneDay'].values)
                future1['LastSecDay'] =pd.Series(x['LastSecDay'].values)
                future1['13th_Day'] =pd.Series(x['13th_Day'].values)
                future1['31st_Day'] =pd.Series(x['31st_Day'].values)
                forecast=self.predict(prophet_basic1,future1)
                forecast=forecast.yhat[-(len(self.test)):]
                c=c+1
                get_list=[]
                for i in range(len(forecast)):
                    get_list.append(forecast.iloc[i])
                fitted_val_list.append(get_list)

            except:
                continue
            
        fitted_val=pd.DataFrame(fitted_val_list,columns=[x for x in range(1,self.n_periods+1)])
        fitted_mean=[]
        for i in range(1,self.n_periods+1):
            fitted_mean.append(fitted_val[i].mean()) 
        if self.walk_forward == "True":
            test_set1=np.array(list(self.test[self.target_value]))
            test_results=round(self.median_absolute_percentage_error_daily(test_set1,fitted_mean),2)
            self.results.append([self.target_item,"PROPHET",round(result_table.train_mape[0],2),round(result_table.val_mape[0],2),test_results,list(self.test[self.target_dates]),fitted_mean,self.set_no])
            self.mape_res=pd.DataFrame(self.results,columns=[self.target_items,'model','train_mape','val_mape','test_mape','test_dates','test_predicted','set_no'])
        else:
            self.results.append([self.target_item,round(result_table.train_mape[0],2),round(result_table.val_mape[0],2),list(self.test[self.target_dates]),fitted_mean])
            self.mape_res=pd.DataFrame(self.results,columns=[self.target_items,'train_mape','val_mape','test_dates','test_predicted'])