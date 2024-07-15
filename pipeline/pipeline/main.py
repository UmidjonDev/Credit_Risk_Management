import dill

import pandas as pd 

from fastapi import FastAPI
from pydantic import BaseModel 

from sklearn.pipeline import Pipeline

#Feature names
coded_features = ['enc_paym_0', 'enc_paym_1', 'enc_paym_2', 'enc_paym_3', 'enc_paym_4', 'enc_paym_5', 'enc_paym_6', 'enc_paym_7',
              'enc_paym_8', 'enc_paym_9', 'enc_paym_10', 'enc_paym_11', 'enc_paym_12', 'enc_paym_13', 'enc_paym_14',
              'enc_paym_15', 'enc_paym_16', 'enc_paym_17', 'enc_paym_18', 'enc_paym_19', 'enc_paym_20', 'enc_paym_21',
              'enc_paym_22', 'enc_paym_23', 'enc_paym_24', 'enc_loans_account_holder_type', 'enc_loans_credit_status',
              'enc_loans_credit_type', 'enc_loans_account_cur']

binary_cols = ['pre_since_opened', 'pre_since_confirmed', 'pre_pterm', 'pre_fterm', 'pre_till_pclose', 'pre_till_fclose',
               'pre_loans_credit_limit', 'pre_loans_next_pay_summ', 'pre_loans_outstanding', 'pre_loans_total_overdue',
               'pre_loans_max_overdue_sum', 'pre_loans_credit_cost_rate', 'pre_loans5', 'pre_loans530', 'pre_loans3060',
               'pre_loans6090', 'pre_loans90', 'pre_util', 'pre_over2limit','pre_maxover2limit']

flag_cols = ['is_zero_loans5', 'is_zero_loans530', 'is_zero_loans3060', 'is_zero_loans6090', 'is_zero_loans90', 'is_zero_util',
             'is_zero_over2limit', 'is_zero_maxover2limit', 'pclose_flag', 'fclose_flag']

ohe_cols = binary_cols + coded_features

app = FastAPI()
with open('./model/credit_risk_management.pkl', 'rb') as f:
    model_data = dill.load(f)
    pipeline = model_data['model']
    ohe_transformation = pipeline.named_steps['preprocessor'].named_steps['column_transform'].named_transformers_['ohe']
    final_pipeline = pipeline.named_steps['preprocessor'].named_steps['aggregation']
    encoded_features = ohe_transformation.named_steps['ohe'].get_feature_names_out()

class Form(BaseModel):
    id : list[int]
    rn : list[int]
    pre_since_opened : list[int]
    pre_since_confirmed : list[int]
    pre_pterm : list[int]                    
    pre_fterm : list[int]                      
    pre_till_pclose : list[int]               
    pre_till_fclose : list[int]                
    pre_loans_credit_limit : list[int]         
    pre_loans_next_pay_summ : list[int]        
    pre_loans_outstanding : list[int]     
    pre_loans_total_overdue : list[int]       
    pre_loans_max_overdue_sum : list[int]     
    pre_loans_credit_cost_rate : list[int]    
    pre_loans5 : list[int]                     
    pre_loans530 : list[int]                   
    pre_loans3060 : list[int]                 
    pre_loans6090 : list[int]                 
    pre_loans90 : list[int]                    
    is_zero_loans5 : list[int]                 
    is_zero_loans530 : list[int]              
    is_zero_loans3060 : list[int]             
    is_zero_loans6090 : list[int]              
    is_zero_loans90 : list[int]               
    pre_util : list[int]                       
    pre_over2limit : list[int]              
    pre_maxover2limit : list[int]             
    is_zero_util : list[int]                  
    is_zero_over2limit : list[int]            
    is_zero_maxover2limit : list[int]         
    enc_paym_0 : list[int]                   
    enc_paym_1 : list[int]                    
    enc_paym_2 : list[int]                    
    enc_paym_3 : list[int]                    
    enc_paym_4 : list[int]                     
    enc_paym_5 : list[int]                     
    enc_paym_6 : list[int]                     
    enc_paym_7 : list[int]                    
    enc_paym_8 : list[int]                     
    enc_paym_9 : list[int]                     
    enc_paym_10 : list[int]                   
    enc_paym_11 : list[int]                   
    enc_paym_12 : list[int]                   
    enc_paym_13 : list[int]                    
    enc_paym_14 : list[int]                   
    enc_paym_15 : list[int]                    
    enc_paym_16 : list[int]                   
    enc_paym_17 : list[int]                   
    enc_paym_18 : list[int]                   
    enc_paym_19 : list[int]                   
    enc_paym_20 : list[int]                   
    enc_paym_21 : list[int]                   
    enc_paym_22 : list[int]                   
    enc_paym_23 : list[int]                   
    enc_paym_24 : list[int]                   
    enc_loans_account_holder_type : list[int]
    enc_loans_credit_status : list[int]       
    enc_loans_credit_type : list[int]         
    enc_loans_account_cur : list[int]          
    pclose_flag : list[int]                   
    fclose_flag : list[int] 

class Prediction(BaseModel): 
    id : str 
    Result : int                  

@app.get('/status')
def status():
    return "I'm OK"

@app.get('/version')
def version():
    return model_data['metadata']

@app.post('/predict', response_model = Prediction)
async def predict(form : Form):
    data_dict = form.dict()
    df = pd.DataFrame.from_dict(data_dict, orient='columns')
    print(df)
    y = pipeline.predict(df)
    return {
        'id' : str(form.id[0]),
        'Result' : y[0]
    }