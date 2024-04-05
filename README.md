# Crypto Vibe Check

### Video Demo:  <[URL HERE](https://www.youtube.com/watch?v=D0Iw-dUW0OA)>

## Description:
Crypto Vibe Check is a predictive analytics tool designed to gauge market sentiment and forecast future trends in the cryptocurrency space, with a primary focus on Bitcoin. By analyzing the Fear and Greed Index alongside historical Bitcoin price data, this project aims to offer users actionable insights into potential market movements, enabling more informed investment decisions.

This tool is built for investors, analysts, and cryptocurrency enthusiasts looking to understand market dynamics and anticipate changes with a data-driven approach.

### Features:
- Real-time analysis of the cryptocurrency market sentiment using the Fear and Greed Index.
- Historical data analysis to identify trends and patterns in Bitcoin's price movement.
- Predictive modeling to forecast future price changes of Bitcoin.
- Notifications via email of predictions for a list of subscribers

### Project structure:

- *main.py*: Acts as the entry point of the application, orchestrating data fetching, processing, and prediction workflow.
- *fetch_data.py*: Contains logic to retrieve the latest Fear and Greed Index data and Bitcoin prices from their respective APIs. This file is crucial for ensuring our predictions are based on the most current market data.
- *model_classes.py*: Defines the Model, ArimaModel, and SarimaxModel classes. These classes are at the core of our predictive modeling, with ArimaModel focusing on time series analysis of the Fear and Greed Index and SarimaxModel leveraging both index data and external factors to forecast Bitcoin prices.
- *send_email.py*: Implements functionality to distribute predictions via email, enabling users to receive timely market insights directly in their inbox.
- *helpers.py*: Offers auxiliary functions that support data handling and preparation, ensuring smooth operation between different components of the project.

### Design choices
 ARIMA and SARIMAX were chosen for their strong performance in time series forecasting, crucial for analyzing cryptocurrency market trends:

  #### Why ARIMA?
  - **Versatility**: Handles various patterns in time series data, making it ideal for the Fear and Greed Index.
  - **Simplicity**: Straightforward implementation and interpretation.
  - **Effectiveness**: Provides accurate short-term forecasts.
  #### Why SARIMAX?
  - **Exogenous Variables**: Considers external factors, enhancing Bitcoin price predictions by including influences like the Fear and Greed Index.
  - **Seasonality**: Capable of identifying and adjusting for seasonal patterns, even if subtle, in cryptocurrency data.
  - **Accuracy**: Offers more precise forecasts by integrating additional data sources.


## Getting Started:
To get started with Crypto Vibe Check, follow these simple setup instructions.

### Prerequisites:
Ensure you have Python 3.12 installed on your system. You will also need `pip` for installing Python packages. This project relies on several third-party libraries, listed in `requirements.txt`.

```python3 -m pip install -r requirements.txt```


### Running the Code:
To run the main application, use the following command in your terminal:

```python project.py```

### Running the Tests:
To execute the automated test suite and ensure everything is functioning as expected, run:

```pytest```

#### Built With:
- Python: The core programming language used.
- Pandas & NumPy: For data manipulation and numerical calculations.
- Requests: To fetch data from APIs.
- Statsmodels: For implementing ARIMA and SARIMAX models.
- Matplotlib: For data visualization.

#### Author:
- **Eduards Žeiris** - *Initial Work*


#### Acknowledgments:
- ChatGPT
