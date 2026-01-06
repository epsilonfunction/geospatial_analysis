import pandas as pandas
import requests


class Pipeline_NEA_realtime_weather:

    BASE_URL = "https://api-open.data.gov.sg/v2"
    TIMEFRAME_MIDPOINT = "/real-time"
    
    def __init__(self):
        self.df = None
        self.endpoints = [
            "air-temperature",
            "rainfall",
            "relative-humidity",
            "wind-direction",
            "wind-speed"
        ]
    
    def fetch_data(self, endpoint):
        
        response = requests.get(
            f"{self.BASE_URL}{self.TIMEFRAME_MIDPOINT}/api/{endpoint}"
        )
        
        return response
        
        # if self.df is None:
        #     self.df = pandas.read_json(f"{self.BASE_URL}{self.TIMEFRAME_MIDPOINT}"

