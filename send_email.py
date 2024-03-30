from datetime import datetime
import smtplib
import ssl
from email.message import EmailMessage
import csv

def send_email(sender, password, prices_forecast_series, today, greed_forecast_series):
    """
    Sends an email with Bitcoin (BTC) price and greed index forecasts to a list of receivers.

    Parameters:
        sender (str): The email address of the sender.
        password (str): The password for the sender's email account.
        prices_forecast_series (pd.Series): A pandas Series containing forecasted BTC prices.
        today (pd.DataFrame): A pandas DataFrame containing today's BTC price and greed index.
        greed_forecast_series (pd.Series): A pandas Series containing forecasted greed indices.

    The email content includes:
    - Today's BTC price and greed index.
    - A warning if the market is due for a correction based on the greed index.
    - Forecasted BTC price increases for 5 and 10 days.
    - Lists of BTC prices and greed indices for the next 10 days.

    The function reads receiver email addresses from 'reciever_emails.csv' and sends them the forecast email.
    Each row in 'reciever_emails.csv' should contain one email address.

    Raises:
        Exception: If there's an error sending the email to any of the receivers.
    """
    
    receiver_csv = "reciever_emails.csv"
    
    correction_warning = "<span style='color: red;'>market due for a correction</span>" if any(int(value) > 100 for value in greed_forecast_series.iloc) else ""
    subject = "Today's BTC prediction"
    body_template = """
    <html>
    <body>
    <p>{datetime_now}</p>
    <p>Today open price: {today_price}</p>
    <p>Today greed coef: {today_greed_coef}</p>

    <h1>AI PREDICTIONS</h1>
    
    <p>{correction_warning}</p>
    <p>price increase <strong>5 days: {price_increase_5_days:.2f}%</strong></p>
    <p>price increase <strong>10 days: {price_increase_10_days:.2f}%</strong></p>

    <p>next 10 days BTC price:<br>
    {prices_forecast}
    </p>
    <p>next 10 days greed coef:<br>
    {greed_forecast}
    </p>
    </body>
    </html>
    """

    with open(receiver_csv, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            receiver_email = row[0]
            body = body_template.format(
                datetime_now=datetime.now(),
                today_price=today["price"].iloc[0],
                today_greed_coef=today["greedCoef"].iloc[0],
                correction_warning=correction_warning,
                price_increase_5_days=(prices_forecast_series.iloc[4] - today["price"].iloc[0])/today["price"].iloc[0] * 100,
                price_increase_10_days=(prices_forecast_series.iloc[9] - today["price"].iloc[0])/today["price"].iloc[0] * 100,
                prices_forecast="<br>".join([f"{idx.strftime('%Y-%m-%d')}: {value:.2f}" for idx, value in prices_forecast_series.items()]),
                greed_forecast="<br>".join([f"{idx.strftime('%Y-%m-%d')}: {value:.2f}" for idx, value in greed_forecast_series.items()])
            )

            em = EmailMessage()
            em['From'] = sender
            em['To'] = receiver_email
            em['Subject'] = subject
            em.add_alternative(body, subtype='html')

            context = ssl.create_default_context()
            try:
                with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
                    smtp.login(sender, password)
                    smtp.send_message(em)
                    print(f"Email sent successfully to {receiver_email}")
            except Exception as e:
                print(f"Error sending email to {receiver_email}: {e}")
                raise
