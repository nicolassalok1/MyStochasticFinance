from yahoofinancials import YahooFinancials
from abc import ABC, abstractmethod, ABCMeta
import pandas as pd

import enum
import requests
from typing import List
import sys


class Frequency(enum.Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

class StockPriceDatasetAdapter(metaclass=ABCMeta):

    """
    Interface to access any data scource of stock price quotes. Multiple implementations can be made to support different
    data sources.

    """

    DEFAULT_TICKER = "PFE"

    @property
    @abstractmethod
    def training_set(self):
        ...

    """
      Function to get training dataset for a given stock symbol (ticker). This dataset can be used to train a stock price model. 
      Although there is no such restrictions on using it elsewhere.      
      
      Returns
      ----
      A dataframe. Each dataframe has two columns: stock price & time 
    
    """

    @property
    @abstractmethod
    def validation_set(self):
        ...

    """
      Function to get validation dataset for a given stock symbol (ticker). This dataset can be used to train a stock price model.
      Although there is no such restrictions on using it elsewhere.      
      
      Returns
      ----
      A dataframe. Each dataframe has two columns: stock price & time 

    """


class BaseStockPriceDatasetAdapter(StockPriceDatasetAdapter, ABC):
    def __init__(self, ticker: str = None):
        self._ticker = ticker
        self._training_set = None
        self._validation_set = None

    @abstractmethod
    def _connect_and_prepare(self, date_range: tuple):
        ...

    """
      This function should be overriden by the implementing data source adapter. It should connect to the stock 
      price data source and return records within the specified date range
    """

    @property
    def training_set(self):
        return self._training_set.copy()

    @property
    def validation_set(self, ticker=None):
        return self._validation_set.copy()


class YahooFinancialsAdapter(BaseStockPriceDatasetAdapter):

    """
    Dataset adapter for Yahoo Financials (https://finance.yahoo.com/).
    """

    def __init__(
        self,
        # Name of ticker for which stock values to be fetched. Default value is PFE i.e. for Pfizer
        ticker=StockPriceDatasetAdapter.DEFAULT_TICKER,
        # YahooFiancials support different time internvals. Frequency is an enum to hold
        # that value to be sent via request. Values can be: 'daily', 'weekly', 'monthly'
        frequency=Frequency.DAILY,
        # Stock values to be fetched for the given date range
        training_set_date_range=("2020-01-01", "2021-12-31"),
        validation_set_date_range=("2013-07-01", "2013-08-31"),
    ):
        super().__init__(ticker=ticker)
        self._frequency = frequency
        self._yf = YahooFinancials(self._ticker)
        self._training_set = self._connect_and_prepare(training_set_date_range)
        self._validation_set = self._connect_and_prepare(
            validation_set_date_range)

    def _connect_and_prepare(self, date_range: tuple):
        stock_price_records = None
        records = self._yf.get_historical_price_data(
            date_range[0], date_range[1], self._frequency.value
        )[self._ticker]

        stock_price_records = pd.DataFrame(data=records["prices"])[
            ["formatted_date", "close"]
        ]

        # Rename columns for convenience
        stock_price_records.rename(
            columns={"formatted_date": "time", "close": "stock price"}, inplace=True
        )

        return stock_price_records


class MarketStackAdapter(BaseStockPriceDatasetAdapter):

    """
    Dataset adapter for Market Stack (https://marketstack.com/).
    It can be used for symbols not supported by Yahoo Fiancials.
    """

    # Dictionary of request paramters
    _REQ_PARAMS = {
        "access_key": "ce72d47022d573ffb1c47820c7e98f15", "limit": 500}

    # REST API url to get end-of-day stock quotes
    # supported by marketstack.com
    _EOD_API_URL = "http://api.marketstack.com/v1/eod"

    # REST API url to get list of all stock symbol
    # supported by marketstack.com
    _TICKER_API_URL = "http://api.marketstack.com/v1/tickers"

    class _PaginatedRecords:

        """
        Market stack API sends paginated response with offset,
        limit & total records.Inner class _PaginatedRecords
        provides a stateful page navigation mechanism to
        iterated over records.

        """

        def __init__(self, api_url, req_params):
            self._req_params = req_params
            self._offset = 0
            self._total_records = sys.maxsize
            self._api_url = api_url

        def __getitem__(self, index):
            """
            Ducktyped function to get the current page records &
            increment the offset accordingly
            """

            if (self._offset + self._req_params["limit"]) >= self._total_records:
                raise StopIteration()

            self._req_params["offset"] = self._offset
            api_response = requests.get(self._api_url, self._req_params).json()
            self._total_records = api_response["pagination"]["total"]
            self._offset = self._offset + self._req_params["limit"] + 1
            return api_response["data"]

    def __init__(
        self,
        training_set_date_range=("2020-01-01", "2021-12-31"),
        validation_set_date_range=("2013-07-01", "2013-08-31"),
        ticker: str = None,
    ):
        super().__init__(ticker=ticker)

        self._training_set = self._connect_and_prepare(training_set_date_range)

    def _connect_and_prepare(self, date_range: tuple):
        def _extract_stock_price_details(stock_price_records, page):
            """
            Inner function to extract fields: 'close', 'date', 'symbol'
            of current element obtained from json response.
            """

            ticker_symbol = page["symbol"]
            stock_record_per_symbol = stock_price_records.get(ticker_symbol)
            if stock_record_per_symbol is None:
                stock_record_per_symbol = pd.DataFrame()

            entry = {
                "stock price": [page["close"]],
                "time": [page["date"].split("T")[0]],
            }

            stock_price_records[ticker_symbol] = pd.concat(
                [stock_record_per_symbol, pd.DataFrame(entry)], ignore_index=True
            )
            return stock_price_records

        if self._ticker is None:
            return None

        req_params = MarketStackAdapter._REQ_PARAMS.copy()
        req_params["symbols"] = self._ticker
        req_params["date_from"] = date_range[0]
        req_params["date_to"] = date_range[1]
        stock_price_records = {}

        # Iterated over response and fetch records to populate a custom dataframe
        for records in MarketStackAdapter._PaginatedRecords(
            api_url=MarketStackAdapter._EOD_API_URL, req_params=req_params
        ):
            for page in records:
                stock_price_records = _extract_stock_price_details(
                    stock_price_records, page
                )

        return stock_price_records

    @classmethod
    def get_samples_of_available_tickers(cls):
        """
        Function to get a collection of available symbols from MarketStack.
        Pagination support can be added as enhancement as well.

        """
        api_response = requests.get(
            MarketStackAdapter._TICKER_API_URL, MarketStackAdapter._REQ_PARAMS
        ).json()
        return [record["symbol"] for record in api_response["data"]]
