import schedule
import time
import warnings
import pandas as pd
import csv
import time
import os
import requests
from datetime import datetime
from model_classes import ArimaModel, SarimaxModel
from send_email import send_email

PREDICT_DAY_COUNT = 10

amount_of_days = 10

def main():

    warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels.tsa.base.tsa_model")
    warnings.filterwarnings("ignore", message=".*An unsupported index was provided and will be ignored when e.g. forecasting.*", module="statsmodels.*")
    warnings.filterwarnings("ignore", message="No supported index is available. Prediction results will be given with an integer index beginning at `start`.", module="statsmodels.*")

    run_predict()

#     schedule.every().day.at("13:00").do(job)

#     while True:
#         schedule.run_pending()
#         time.sleep(3600)


# def job():
#     run_predict()


def fetch_data():
    """
    Fetches greed index and BTC price data from their respective APIs.

    Returns:
        tuple: Contains two lists, greed index data and BTC price data, respectively.

    Raises:
        SystemExit: If there's a network error.
        ValueError: If the fetched data is empty or unavailable.

    """

    greedIndexURL = "https://api.alternative.me/fng/?limit=100000&date_format=eu"
    key = "6a3f091047ad80bb804e418f7880da6fabebe0354ba1936578d7adc6d31e906d"
    BTCPriceURL = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=2000&api_key={key}"

    try:
        greed_response = requests.get(greedIndexURL)
        greed_response.raise_for_status()
        btc_response = requests.get(BTCPriceURL)
        btc_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"Network error occurred: {e}")

    greedJSON = greed_response.json().get("data", [])
    btcJSON = btc_response.json().get("Data", {}).get("Data", [])

    if not greedJSON or not btcJSON:
        raise ValueError("One or more datasets are empty or unavailable.")

    return greedJSON, btcJSON

def create_csv(greedJSON, btcJSON, file_name='data.csv'):
    """
    Creates a CSV file from greed index and BTC price data.

    Parameters:
        greedJSON (list): List of dictionaries containing greed index data.
        btcJSON (list): List of dictionaries containing BTC price data.
        file_name (str, optional): The name of the CSV file to be created. Defaults to 'data.csv'.

    Raises:
        ValueError: If no matching data is found between greed index and BTC prices.
        SystemExit: If an error occurs while writing to the CSV file.
    """
    rows = []
    for item in greedJSON:
        try:
            date_object = datetime.strptime(item["timestamp"], '%d-%m-%Y')
            unix_time = int(time.mktime(date_object.timetuple()))
        except ValueError as e:
            print(f"Skipping item due to date parsing error: {e}")
            continue

        for BTCItem in btcJSON:

            try:
                timeUnix = abs(BTCItem.get("time") - unix_time)
            except TypeError as e:
                raise TypeError(f"BTC time not in int format: {e}")

            if timeUnix <= 86400:
                BTCPrice = BTCItem.get("open")
                if BTCPrice is None:
                    print("Skipping item due to missing BTC price.")
                    continue
                line = [item["timestamp"], BTCPrice, item.get("value", "N/A")]
                rows.append(line)
                break

    if not rows:
        raise ValueError("No matching data found for greed index and BTC prices.")

    rows.reverse()

    try:
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["date", "price", "greedCoef"])
            writer.writerows(rows)
    except IOError as e:
        raise SystemExit(f"Failed to write data to {file_name}: {e}")

def get_csv():
    """
    Fetches data from APIs and creates a CSV file with greed index and BTC price data.

    This function serves as a wrapper that:
    1. Calls `fetch_data()` to retrieve greed index and BTC price data from their APIs.
    2. Calls `create_csv()` with the fetched data to generate a 'data.csv' file.

    Raises:
        Passes through any exceptions raised by `fetch_data()` or `create_csv()`.
    """
    greedJSON, btcJSON = fetch_data()
    create_csv(greedJSON, btcJSON)

def get_pandas_df(file):
    """
    Reads a CSV file into a pandas DataFrame with specific parsing rules.

    Parameters:
        file (str): Path to the CSV file to be read.

    Returns:
        pd.DataFrame: DataFrame containing the data from the CSV file.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        ValueError: If the CSV file is empty or an error occurs during processing.
    """

    if not os.path.exists(file):
        raise FileNotFoundError("The CSV file does not exist. Ensure 'get_csv' has been generated successfully.")

    try:
        df = pd.read_csv(file, index_col='date', parse_dates=True, dayfirst=True)
        df['greedCoef'] = pd.to_numeric(df['greedCoef'], errors='coerce')
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty. Ensure it contains data.")
    except Exception as e:
        raise ValueError(f"An error occurred while processing the CSV file: {e}") from e

    if df.empty:
        raise ValueError("DataFrame is empty after loading the CSV. Check the file content.")

    return df

def run_predict():
    """
    Orchestrates the fetching of data, prediction processes, and sending of email with forecast results.

    This function encapsulates the workflow of:
    1. Fetching recent greed index and BTC price data and saving it to 'data.csv'.
    2. Loading the data from 'data.csv' into a pandas DataFrame.
    3. Generating predictions using both ARIMA and SARIMAX models on the fetched data.
    4. Sending an email with the prediction results.

    Exceptions:
        Any exceptions raised during data preparation, model processing, or email sending are caught and logged,
        halting the execution of subsequent steps.
    """

    file = "data.csv"
    try:
        get_csv()
        df = get_pandas_df(file)
    except (SystemExit, FileNotFoundError, ValueError) as e:
        print(f"Error preparing data: {e}")
        return

    try:
        today = pd.Timestamp.now().normalize()
        future_dates = pd.date_range(start=today, periods=PREDICT_DAY_COUNT + 1)[1:]

        arima = ArimaModel(df['greedCoef'], order=(1, 0, 2), future_dates=future_dates)
        arima.fit()
        arima.predict(PREDICT_DAY_COUNT)

        future_greedCoef = arima.futureCoef()
        greed_forecast_series = arima.series()
        print(greed_forecast_series)

    except Exception as e:
        print(f"Error during ARIMA model processing: {e}")
        return

    try:
        sarimax = SarimaxModel(data=df['price'], exog=df[['greedCoef']], order=(5, 0, 0), seasonal_order=(2, 1, 0, 7), future_dates=future_dates)
        sarimax.fit(exog=df[['greedCoef']])
        sarimax.predict(PREDICT_DAY_COUNT, future_greedCoef[['greedCoef']])
        prices_forecast_series = sarimax.series()
        print(prices_forecast_series)

    except Exception as e:
        print(f"Error during SARIMAX model processing: {e}")
        return

    try:
        send_email(
            sender='cryptoemail377@gmail.com',
            password='durh yszf uljk xwet',
            prices_forecast_series=prices_forecast_series,
            today=df.tail(1),
            greed_forecast_series=greed_forecast_series
        )
    except Exception as e:
        print(f"Failed to send email: {e}")



if __name__ == "__main__":
    main()
