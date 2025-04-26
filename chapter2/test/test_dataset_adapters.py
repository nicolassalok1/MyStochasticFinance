
from chapter2.stock_price_dataset_adapters import YahooFinancialsAdapter
import chapter2.visualization as vs

def test_yahoo_financials_adapter():
    records = {'Apple Inc':
    YahooFinancialsAdapter(
                ticker="AAPL",
                training_set_date_range=("2021-02-01", "2021-04-30"),
            ).training_set,
            'Google':
    YahooFinancialsAdapter(
                ticker="GOOG",
                training_set_date_range=("2021-02-01", "2021-04-30"),
            ).training_set }
    
    vs.plot_security_prices(records, 'stock price')
