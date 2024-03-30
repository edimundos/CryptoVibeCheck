from datetime import datetime
import pytest
import pandas as pd
from helpers import get_pandas_df, create_csv, fetch_data, run_predict
import json
import os
from send_email import send_email
from model_classes import ArimaModel, SarimaxModel
from unittest.mock import patch
import warnings
import logging
import time

logging.basicConfig(level=logging.INFO)

current_dir = os.path.dirname(os.path.realpath(__file__))

VALID_CSV_FILE = os.path.join(current_dir, 'valid_data.csv')
INVALID_CSV_FILE = os.path.join(current_dir, 'invalid.csv')
NOTEXIST_CSV_FILE = os.path.join(current_dir, 'notexist.csv')

BTC_JSON_DATA = os.path.join(current_dir, 'btc_json_data.json')
GREED_JSON_DATA = os.path.join(current_dir, 'greed_json_data.json')
INVALID_JSON_DATA = os.path.join(current_dir, 'invalid_json_data.json')

# UNIT TESTS

def test_get_pandas_df_valid():
    actual_df = get_pandas_df(VALID_CSV_FILE)

    expected_data = {
        'price': [61924.02, 67857.3, 65493.22],
        'greedCoef': [74, 78, 75]
    }
    expected_df = pd.DataFrame(expected_data,
                               index=pd.to_datetime(["20-03-2024", "21-03-2024", "22-03-2024"], dayfirst=True))
    expected_df.index.name = 'date'

    pd.testing.assert_frame_equal(actual_df.sort_index(), expected_df.sort_index(), check_dtype=False)

def test_get_pandas_df_invalid():
    with pytest.raises(FileNotFoundError):
        get_pandas_df(NOTEXIST_CSV_FILE)

    with pytest.raises(ValueError):
        get_pandas_df(INVALID_CSV_FILE)

def test_create_csv_valid():
    with open(GREED_JSON_DATA, 'r') as file:
        greedJSON = json.load(file)
        
    with open(BTC_JSON_DATA, 'r') as file:
        btcJSON = json.load(file)
    
    output_csv_path = 'output.csv'
    create_csv(greedJSON, btcJSON, output_csv_path)
    
    assert os.path.exists(output_csv_path)

    os.remove(output_csv_path)
    
def test_create_csv_invalid():
    with open(INVALID_JSON_DATA, 'r') as file:
        invalidJSON = json.load(file)
        
    with pytest.raises(ValueError):
        create_csv(invalidJSON, invalidJSON, 'output.csv')
        
def test_email_sending_success(mocker):
    mock_smtp_class = mocker.patch('send_email.smtplib.SMTP_SSL')
    mocker.patch('builtins.open', mocker.mock_open(read_data="test@example.com\n"))
    
    prices_forecast_series = pd.Series(
        [69548.069586, 72197.699044, 74413.108224, 72036.672749, 75101.292521,
         74523.191588, 73627.264214, 72439.338986, 75696.905369, 76878.914300],
        index=pd.date_range(start="2024-03-31", periods=10),
        name="Predicted Prices"
    )

    today = pd.DataFrame({
        'price': [69903.72],
        'greedCoef': [75]
    }, index=pd.to_datetime(["2024-03-30"]))

    greed_forecast_series = pd.Series(
        [76.505326, 77.065571, 77.357898, 76.293440, 75.369683,
         74.500068, 74.204432, 74.166510, 73.792948, 73.222712],
        index=pd.date_range(start="2024-03-31", periods=10),
        name="Predicted GreedCoef"
    )
    
    send_email('cryptoemail377@gmail.com', 'durh yszf uljk xwet', prices_forecast_series, today, greed_forecast_series)
    
    # Assert SMTP_SSL was instantiated and used
    mock_smtp_class.assert_called_once_with('smtp.gmail.com', 465, context=mocker.ANY)

def test_email_sending_failure_due_to_smtp(mocker):
    mocker.patch('send_email.smtplib.SMTP_SSL', side_effect=Exception("SMTP error"))
    
    prices_forecast_series = pd.Series(
        [69548.069586, 72197.699044, 74413.108224, 72036.672749, 75101.292521,
         74523.191588, 73627.264214, 72439.338986, 75696.905369, 76878.914300],
        index=pd.date_range(start="2024-03-31", periods=10),
        name="Predicted Prices"
    )

    today = pd.DataFrame({
        'price': [69903.72],
        'greedCoef': [75]
    }, index=pd.to_datetime(["2024-03-30"]))

    greed_forecast_series = pd.Series(
        [76.505326, 77.065571, 77.357898, 76.293440, 75.369683,
         74.500068, 74.204432, 74.166510, 73.792948, 73.222712],
        index=pd.date_range(start="2024-03-31", periods=10),
        name="Predicted GreedCoef"
    )
    
    with pytest.raises(Exception) as exc_info:
        send_email('sender@example.com', 'password', prices_forecast_series, today, greed_forecast_series)
    
    assert "SMTP error" in str(exc_info.value)


# PERFORMANCE TESTS

def test_fetch_data_performance():
    start_time = time.time()

    data = fetch_data() 

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Data fetching execution time: {execution_time} seconds.")

    assert execution_time < 5, "Data fetching exceeded the acceptable performance threshold."
    
def test_run_predict_performance():
    start_time = time.time()

    run_predict() 

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"`run_predict` execution time: {execution_time} seconds.")

    assert execution_time < 30, "`run_predict` exceeded the acceptable performance threshold."
   
# INTEGRATION TESTS
 
def test_arima_model_integration():
    dates = pd.date_range(start=datetime.today(), periods=10)
    values = pd.Series(range(10), index=dates)

    arima_model = ArimaModel(data=values, order=(1, 1, 1), future_dates=dates[1:])
    arima_model.fit()
    arima_model.predict(steps=9)
    
    arima_predictions = arima_model.series()
    
    # Since the series method raises an exception if no forecast is available,
    # if we reach this point, it implies predictions are available.
    assert not arima_predictions.empty, "ArimaModel failed to make predictions."
    
def test_sarimax_model_integration():
    dates = pd.date_range(start=datetime.today(), periods=20)
    values = pd.Series(range(10), index=dates[:10])  
    exog = pd.DataFrame({'exog_var': range(10)}, index=dates[:10]) 

    sarimax_model = SarimaxModel(data=values, exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), future_dates=dates[5:10])
    sarimax_model.fit(exog=exog)
    sarimax_model.predict(steps=5, exog=exog.iloc[:5])
    
    arima_predictions = sarimax_model.series()
    
    # Since the series method raises an exception if no forecast is available,
    # if we reach this point, it implies predictions are available.
    assert not arima_predictions.empty, "SarimaxModel failed to make predictions."

def test_full_workflow_integration(mocker):
    dates = pd.date_range(start=datetime.today(), periods=20)
    values = pd.Series(range(20), index=dates)  # Main data series
    exog = pd.DataFrame({'exog_var': range(20)}, index=dates)  # Exogenous variables

    arima_model = ArimaModel(data=values[:10], order=(1, 1, 1), future_dates=dates[1:11])
    arima_model.fit()
    arima_model.predict(steps=10)
    arima_predictions = arima_model.series()
    assert not arima_predictions.empty, "ArimaModel failed to make predictions."

    sarimax_model = SarimaxModel(data=values[:10], exog=exog[:10], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), future_dates=dates[5:15])
    sarimax_model.fit(exog=exog[:10])
    sarimax_model.predict(steps=10, exog=exog[5:15])
    sarimax_predictions = sarimax_model.series()
    assert not sarimax_predictions.empty, "SarimaxModel failed to make predictions."

    with patch('send_email.smtplib.SMTP_SSL') as mock_smtp:
        sender = "cryptoemail377@gmail.com"
        password = "durh yszf uljk xwet"
        today_df = pd.DataFrame({'price': [values.iloc[0]], 'greedCoef': [exog['exog_var'].iloc[0]]}, index=[dates[0]])

        send_email(sender=sender, password=password, greed_forecast_series=arima_predictions, today=today_df, prices_forecast_series=sarimax_predictions)
        
        mock_smtp.assert_called_once_with('smtp.gmail.com', 465, context=mocker.ANY)

