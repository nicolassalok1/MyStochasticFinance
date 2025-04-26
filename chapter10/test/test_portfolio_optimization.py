import chapter10.visualization as sv
from chapter10.portfolio_assets import YahooFinancialsPortfolioAssets
from chapter2.stock_price_dataset_adapters import Frequency
from chapter10.markowitz_portfolio import MarkowitzMinVariancePortfolioOptimizer, ExtendedMarkowitzMinVariancePortfolioOptimizer
from chapter10.portfolio_simulation import PortfolioSimulation


def test_visualize_portfolio_returns():
    training_set_date_range = ('2021-01-01', '2021-12-31')
    yfp = YahooFinancialsPortfolioAssets(tickers=['GM', 'VZ', 'WMT'],
                                         date_range=training_set_date_range,
                                         frequency=Frequency.DAILY)
    sv.plot_returns_for_different_assets(
        yfp.periodic_returns_for_different_assets)


def test_visualize_portfolio_cov():
    training_set_date_range = ('2021-01-01', '2021-12-31')
    yfp = YahooFinancialsPortfolioAssets(tickers=['GM', 'VZ', 'WMT'],
                                         date_range=training_set_date_range,
                                         frequency=Frequency.DAILY)
    cov = yfp.covariance_of_returns
    cov['Mean Returns'] = yfp.unweighted_mean_returns

    return cov


def test_markowitz_portfolio_optimization():
    training_set_date_range = ('2017-01-01', '2020-12-31')

    # symbols of stocks of companies like Google, Meta, Tesla, AT&T, Walmart etc
    asset_tickers = ['GM', 'VZ', 'WMT', 'AMD', 'TSLA', 'T', 'GOOG', 'META']

    mp_1 = MarkowitzMinVariancePortfolioOptimizer(expected_mean_return=0.01)
    yf_1 = YahooFinancialsPortfolioAssets(tickers=asset_tickers,
                                          frequency=Frequency.DAILY,
                                          date_range=training_set_date_range)
    mp_1.fit(yf_1)
    print(mp_1.asset_allocation_distribution)

    mp_2 = MarkowitzMinVariancePortfolioOptimizer(expected_mean_return=0.01)
    yf_2 = YahooFinancialsPortfolioAssets(tickers=asset_tickers,
                                          frequency=Frequency.WEEKLY,
                                          date_range=training_set_date_range)
    mp_2.fit(yf_2)

    mp_3 = MarkowitzMinVariancePortfolioOptimizer(expected_mean_return=0.01)
    yf_3 = YahooFinancialsPortfolioAssets(tickers=asset_tickers,
                                          frequency=Frequency.MONTHLY,
                                          date_range=training_set_date_range)
    mp_3.fit(yf_3)

    sv.plot_portfolios([('Daily', mp_1), ('Weekly', mp_2), ('Monthly', mp_3)])


def test_extended_markowitz_portfolio_optimization():
    training_set_date_range = ('2017-01-01', '2020-12-31')
    asset_tickers = ['GM', 'VZ', 'WMT', 'AMD', 'TSLA', 'T', 'GOOG', 'META']

    expected_mean_return = 0.01
    mp_1 = ExtendedMarkowitzMinVariancePortfolioOptimizer(
        expected_mean_return=expected_mean_return)
    yahoofinancials_portfolio_assets_1 = YahooFinancialsPortfolioAssets(tickers=asset_tickers,
                                                                        frequency=Frequency.DAILY,
                                                                        date_range=training_set_date_range)
    mp_1.fit(yahoofinancials_portfolio_assets_1)

    mp_2 = ExtendedMarkowitzMinVariancePortfolioOptimizer(
        expected_mean_return=expected_mean_return)
    yahoofinancials_portfolio_assets_2 = YahooFinancialsPortfolioAssets(tickers=asset_tickers,
                                                                        frequency=Frequency.WEEKLY,
                                                                        date_range=training_set_date_range)
    mp_2.fit(yahoofinancials_portfolio_assets_2)

    mp_3 = ExtendedMarkowitzMinVariancePortfolioOptimizer(
        expected_mean_return=expected_mean_return)
    yahoofinancials_portfolio_assets_3 = YahooFinancialsPortfolioAssets(tickers=asset_tickers,
                                                                        frequency=Frequency.MONTHLY,
                                                                        date_range=training_set_date_range)
    mp_3.fit(yahoofinancials_portfolio_assets_3)

    portfolios = []
    if mp_1.optimal_variance is not None:
        portfolios.append(('Daily', mp_1))
    if mp_2.optimal_variance is not None:
        portfolios.append(('Weekly', mp_2))
    if mp_3.optimal_variance is not None:
        portfolios.append(('Monthly', mp_3))

    sv.plot_portfolios(portfolios)


def test_mean_var_distribution():
    training_set_date_range = ('2017-01-01', '2017-01-31')

    yfp = YahooFinancialsPortfolioAssets(tickers=['GM', 'VZ', 'WMT'],
                                         date_range=training_set_date_range,
                                         frequency=Frequency.DAILY)

    mv = PortfolioSimulation(portfolio_assets=yfp,
                             portfolio_optimizer_full_class_name='chapter10.markowitz_portfolio.MarkowitzMinVariancePortfolioOptimizer')
    mean_var = mv.mean_variance_distribution
    sv.plot_scatter(data=mean_var, y_name='Expected Return',
                    x_name='Volatility', title='Expected Return vs Volatility')


def test_efficient_frontier():
    training_set_date_range = ('2017-01-01', '2017-01-31')

    yfp = YahooFinancialsPortfolioAssets(tickers=['GM', 'VZ', 'WMT'],
                                         date_range=training_set_date_range,
                                         frequency=Frequency.DAILY)

    mv = PortfolioSimulation(portfolio_assets=yfp,
                             portfolio_optimizer_full_class_name='chapter10.markowitz_portfolio.MarkowitzMinVariancePortfolioOptimizer')

    eff_frn = mv.efficient_frontier
    sv.plot_scatter(data=eff_frn, y_name='Expected Return',
                    x_name='Volatility', title='Efficient Frontier')


# test_extended_markowitz_portfolio_optimization()
# test_markowitz_portfolio_optimization()
# test_mean_var_distribution()
# test_efficient_frontier()
test_visualize_portfolio_returns()
test_visualize_portfolio_cov()
