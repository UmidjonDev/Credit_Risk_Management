import dill
import joblib
import pandas as pd
from datetime import datetime
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from catboost import CatBoostClassifier, Pool
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

#Scoring metric
scoring_metric = 'roc_auc'

#Feature's categories
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

#Converting the result of ohe transformation back to dataframe
class DataFrameTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        encoded_features = ohe_transformation.named_steps['ohe'].get_feature_names_out()
        return pd.DataFrame(X, columns=encoded_features)

"""
#Creating the PoolCreator class for transforming out padnas dataframes to Pool objects
class PoolCreator(BaseEstimator, TransformerMixin):
    def __init__(self, class_weights):
        self.class_weights = class_weights

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        if y is None:
            return Pool(data = X)
        else:
            sample_weights = np.array([self.class_weights[label] for label in y])
            return Pool(data = X, label = y, weight = sample_weights)
"""
                 
def pipeline() -> None :
    print('Credit Risk Management System Model')

     #Reading dataset from parquet files
    data = []
    for i in range(0, 12, 1) :
        df_sample = pd.read_parquet(path = f'/content/drive/MyDrive/data/train_data_{i}.pq')
        data.append(df_sample)

    X = pd.concat(data)
    y = pd.read_csv(filepath_or_buffer = '/content/drive/MyDrive/data/train_target.csv')['flag']

    #Calculating weights for each class
    class_counts = y.value_counts().reset_index()
    class_counts.columns = ['class', 'count']
    class_counts['class_weight'] = len(y) / class_counts['count']
    class_weights = class_counts.set_index('class')['class_weight'].to_dict()
    sample_weights = np.array([class_weights[label] for label in y])

    #Feature engineering
    ohe_transformation = Pipeline(steps = [
        ('ohe', OneHotEncoder(min_frequency = 250, sparse_output = False, dtype = 'int8')),
        ('to_dataframe', DataFrameTransformer())
    ])

    class ToDataFrame:
        def __init__(self, flag_cols, ohe_transformation):
            self.flag_cols = flag_cols
            self.ohe_transformation = ohe_transformation

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            encoded_features = self.ohe_transformation.named_steps['ohe'].get_feature_names_out()
            columns = list(encoded_features) + self.flag_cols + ['id', 'rn']
            return pd.DataFrame(X, columns=columns)

    ohe_transformation.fit(X[ohe_cols])
    encoded_features = ohe_transformation.named_steps['ohe'].get_feature_names_out()

    #aggregator function
    def aggregator_df(df : pd.DataFrame) -> pd.DataFrame:
        category_df = df.groupby(['id'], as_index=False)[encoded_features].sum()
        flag_df = df.groupby(['id'], as_index=False)[flag_cols].mean()
        rn_df = df.groupby(['id'], as_index=False)[['rn']].count()

        df = pd.merge(left=rn_df, right=flag_df, on='id')
        df = pd.merge(left=df, right=category_df, on='id')

        return df
    
    #Pipelines
    flag_transformation = Pipeline(steps = [
        ('flag', FunctionTransformer(lambda x : x))
    ])

    column_transformer = ColumnTransformer(transformers = [
        ('ohe', ohe_transformation, ohe_cols),
        ('flag', flag_transformation, flag_cols + ['id', 'rn'])
    ])

    to_df_transformer = ToDataFrame(flag_cols, ohe_transformation)

    final_pipeline = Pipeline(steps=[
        ('column_transform', column_transformer),
        ('to_dataframe', to_df_transformer),
        ('aggregation', FunctionTransformer(aggregator_df))
    ])

    #Modelling
    cat_model = CatBoostClassifier(
        iterations = 1500,
        learning_rate = 0.1,
        depth = 6,
        eval_metric = 'AUC',
        random_seed = 1,
        class_weights = class_weights
    )

    #Overall model pipeline
    pipe = Pipeline([
        ('preprocessor', final_pipeline),
        #('pool_creator', PoolCreator(class_weights = class_weights)),
        ('classifier', cat_model)
    ])

    #Fitting perfect pipeline for whole dataset
    pipe.fit(X = X, y = y)

    #Evaluation of the model
    predictions = pipe.predict_proba(X = X)[:, 1]
    y_pred = pipe.predict(X)

    roc_auc_cat = roc_auc_score(y, predictions)
    conf_matrix = confusion_matrix(y, y_pred)

    print(f"The model is ready and the roc_auc_score of the model is equal to : {roc_auc_cat}")
    print("Confusion Matrix:\n", conf_matrix)

    model_filename = f'/content/drive/MyDrive/credit_risk_management.pkl'
    dill.dump({'model' : pipe,
        'metadata' :{
            'name' : 'Credit Risk Management',
            'author' : 'Umidjon Sattorov',
            'version' : 1,
            'date' : datetime.now(),
            'type' : type(pipe.named_steps['classifier']).__name__,
            'accuracy' : roc_auc_cat
        }
    }, open('/content/drive/MyDrive/credit_risk_management.pkl', 'wb'))

    print(f'Model is saved as {model_filename} in models directory')

if __name__ == '__main__' : 
    pipeline()
