# This module works to make predictions from received data and return a response

import joblib
import pandas as pd

# Text data cleaned for prediction
def data_preprocess(data: dict):
    data_values = list(data.values())
    cleaned_data = clean_data(data_values)
    return model_run(cleaned_data)
    # return data_values

# Cleaning Function for received text data
def clean_data(data):
    data[3] = data[3].strip()
    data[3] = data[3].lower()
    data[5] = data[5].strip()
    data[5] = data[5].lower()

    if data[3] == 'no':
        data[3] = 1
    elif data[3] == 'yes':
        data[3] = 0
    else:
        print('Incorrect Data')

    if data[5] == 'yes':
        data[5] = 1
        data.insert(6,0)
    elif data[5] == 'no':
        data[5] = 0
        data.insert(6, 1)
    else:
        print('Incorrect Data')
    cleaned_data = [[float(entry) for entry in data]]
    return cleaned_data

# Prediction for outcome using saved model and outputs result
def model_run(frontend_data):
    features = ['cibil_score', 'loan_term', 'income_annum', 'education_ Not Graduate', 'loan_amount',
                'self_employed_ Yes',
                'self_employed_ No', 'luxury_assets_value']

    df = pd.DataFrame(frontend_data, columns=features)
    return model_outcome(df, model_prediction)
    # testing(data3)

# Loads scaler for received data and model for prediction
def model_prediction():
    loaded_scaler = joblib.load('scaler.joblib')
    loaded_model = joblib.load('model.joblib')
    return loaded_scaler, loaded_model

# Fuction makes a prediction and interprets result
def model_outcome(data, model_prediction):
    # print(data)
    scaler, model = model_prediction()
    # scaler1 = scaler.fit(data)
    transform_test = scaler.transform(data)
    pred = model.predict(transform_test)
    # print(transform_test)
    if pred == 0:
        print("Congratulations, Your loan has been approved.")
    elif pred == 1:
        print("Rejected, Sorry!!")
    else:
        print("I think we have an error here")
    return pred
