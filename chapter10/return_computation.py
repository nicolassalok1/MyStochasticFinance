from chapter2.stock_price_dataset_adapters import YahooFinancialsAdapter, Frequency
import chapter10.visualization as sv


def compute_returns():
    monthly = YahooFinancialsAdapter(ticker='WMT', frequency=Frequency.MONTHLY).\
        training_set
    monthly['Return'] = monthly['stock price'] / \
        monthly['stock price'].shift(1) - 1

    weekly = YahooFinancialsAdapter(ticker='WMT', frequency=Frequency.WEEKLY).\
        training_set
    weekly['Return'] = weekly['stock price']/weekly['stock price'].shift(1) - 1

    daily = YahooFinancialsAdapter(ticker='WMT', frequency=Frequency.DAILY).\
        training_set
    daily['Return'] = daily['stock price']/daily['stock price'].shift(1) - 1

    periodic_returns = [
        ('Daily', daily), ('Weekly', weekly), ('Monthy', monthly)]
    sv.plot_returns_for_different_periods('WMT', periodic_returns)


compute_returns()
