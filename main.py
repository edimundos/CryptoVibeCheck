import schedule
import time
import warnings
from helpers import run_predict

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

    
    
if __name__ == "__main__":
    main()
