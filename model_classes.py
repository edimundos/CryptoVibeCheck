from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd

class Model:
    """
    Base class for forecasting models. Should be subclassed for specific implementations.

    Attributes:
        data: Historical data for modeling.
        order: Model order parameters.
        future_dates: Dates for predictions.
        model: Model instance, defined in subclasses.
        results: Fitting results, defined in subclasses.

    Raises:
        NotImplementedError: If fit() or predict() methods are not implemented in subclasses.
    """
    def __init__(self, data, order, future_dates):
        self.data = data
        self.order = order
        self.future_dates = future_dates
        self.model = None
        self.results = None

    def fit(self):
        raise NotImplementedError

    def predict(self, steps):
        raise NotImplementedError


class ArimaModel(Model):
    """
    Implements an ARIMA model for time series forecasting extending the base Model class.

    Methods:
        fit(): Fits the ARIMA model to the provided data.
        predict(steps): Generates forecasts for the specified number of steps ahead.
        futureCoef(): Returns forecasted values in a DataFrame.
        series(): Returns forecasted values in a Series.

    Raises:
        ValueError: If fitting the model or forecasting fails due to an underlying ARIMA model issue.
        RuntimeError: If trying to access forecast values before they are generated.
    """
    def __init__(self, data, order, future_dates):
        super().__init__(data, order, future_dates)

    def fit(self):
        try:
            self.model = ARIMA(self.data, order=self.order)
            self.results = self.model.fit()
        except Exception as e:
            raise ValueError(f"Failed to fit ARIMA model: {e}")

    def predict(self, steps):
        try:
            self.greed_forecast = self.results.forecast(steps=steps)
        except Exception as e:
            raise ValueError(f"Failed to forecast with ARIMA model: {e}")
    
    def futureCoef(self):
        """
        Returns:
            DataFrame: Forecasted values with future dates as index.
        """
        if hasattr(self, 'greed_forecast'):
            return pd.DataFrame(self.greed_forecast.values, index=self.future_dates, columns=['greedCoef'])
        else:
            raise RuntimeError("No forecast available.")
    
    def series(self):
        """
        Returns:
            Series: Forecasted greed coefficients as a pandas Series.
        """
        if hasattr(self, 'greed_forecast'):
            return pd.Series(self.greed_forecast.values, index=self.future_dates, name='Predicted GreedCoef')
        else:
            raise RuntimeError("No forecast series available.")

class SarimaxModel(Model):
    """
    Implements a SARIMAX model for time series forecasting with exogenous variables.

    Inherits from Model class.

    Attributes:
        exog (pd.DataFrame or pd.Series): Exogenous variables for the model.
        seasonal_order (tuple): The seasonal order for the SARIMAX model.

    Methods:
        fit(exog, maxiter=200, method='bfgs'): Fits the SARIMAX model to the data.
        predict(steps, exog): Generates forecasts for the specified steps with exogenous variables.
        series(): Returns forecasted values as a pandas Series.

    Raises:
        ValueError: If fitting the model or forecasting fails.
    """
    def __init__(self, data, exog, order, seasonal_order, future_dates):
        super().__init__(data, order, future_dates)
        self.exog = exog
        self.seasonal_order = seasonal_order

    def fit(self, exog, maxiter=200, method='bfgs'):
        try:
            self.model = SARIMAX(self.data, exog=exog, order=self.order, seasonal_order=self.seasonal_order)
            self.results = self.model.fit(maxiter=maxiter, method=method)
        except Exception as e:
            raise ValueError(f"Failed to fit SARIMAX model: {e}")

    def predict(self, steps, exog):
        try:
            forecast = self.results.get_forecast(steps=steps, exog=exog)
            self.predicted = forecast.predicted_mean
        except Exception as e:
            raise ValueError(f"Failed to forecast with SARIMAX model: {e}")

    def series(self):
        """
        Returns:
            pd.Series: A series containing the forecasted values with the future dates as the index.

        Raises:
            RuntimeError: If the forecasted values are attempted to be accessed before forecasting.
        """
        if hasattr(self, 'predicted'):
            return pd.Series(self.predicted.values, index=self.future_dates, name='Predicted Prices')
        else:
            raise RuntimeError("No forecast series available.")
