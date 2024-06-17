#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd




categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df



def load_model(model_file):
    with open(model_file, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model
    
def apply_model(input_file, model_file, year, month, output_file):
    print(f'reading the data from {input_file}...')
    df = read_data(input_file)
    dicts = df[categorical].to_dict(orient='records')
    
    print('loading the model...')

    dv, model = load_model(model_file)
    X_val = dv.transform(dicts)
    print('applying the model...')

    y_pred = model.predict(X_val)
    print(f'The mean of the predicted duration is {y_pred.mean():.2f}')
    print(f'The standard deviation of the predicted duration is {y_pred.std():.2f}')
    
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame(df['ride_id'])
    df_result['predicted_duration'] = y_pred
    
    print(f'saving the result to {output_file}...')
    df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False)

def run():
    
    year = int(sys.argv[1]) # 2023
    month = int(sys.argv[2]) # 3
    taxi_type = 'yellow'
    model_file = 'model.bin'
    
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'


    apply_model(input_file = input_file, 
            model_file = model_file, 
            year = year, 
            month = month, 
            output_file = output_file)
    
if __name__ == '__main__':
    run()

