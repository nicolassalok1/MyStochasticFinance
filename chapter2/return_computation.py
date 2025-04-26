from stock_price_dataset_adapters import YahooFinancialsAdapter, Frequency
import visualization as vs

def compute_returns():
    monthly = YahooFinancialsAdapter(frequency=Frequency.MONTHLY).training_set

    #   R_t = S_t/S_(t-1) - 1
    monthly["Return"] = monthly["stock price"] / monthly["stock price"].shift(1) - 1

    weekly = YahooFinancialsAdapter(frequency=Frequency.WEEKLY).training_set
    weekly["Return"] = weekly["stock price"] / weekly["stock price"].shift(1) - 1

    daily = YahooFinancialsAdapter(frequency=Frequency.DAILY).training_set
    daily["Return"] = daily["stock price"] / daily["stock price"].shift(1) - 1

    periodic_returns = [("Daily", daily), ("Weekly", weekly), ("Monthy", monthly)]

    return periodic_returns

def test_plot_periodic_returns():
    periodic_returns = compute_returns()
    vs.plot_returns_for_different_periods("Pfizer", periodic_returns)
