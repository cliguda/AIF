<div align="center">
<!-- Title: -->
  <h1><a href="https://github.com/AIF/">A.I.F. - Artificial Intelligence for Finance</a></h1>
<!-- Picture -->
  <a href="https://github.com/AIF/">
    <img src="https://cdn.website-editor.net/079e978268ba4c8a90e031ab3e62ff2a/dms3rep/multi/AIF_Intro_small.png" 
     height="300">
  </a>
</div>

# Intro

AIF is a framework to analyse cryptocurrencies, develop sophisticated strategies and generate trading signals. The main
focus during the developing process was to build a scalable and easy to extend framework to leverage artificial
intelligence on financial data. Therefore, ideas from Software Architecture and Data Science were brought together, to
provide the following main features:

- Ability to handle different assets (BTC, ETH, ...)
- Backtesting and cross-validate strategies with historical data to ensure, that only profitable strategies are used (
  Trading fees are considered in the evaluation process).
- Merge data from different timeframes together, to analyse short term movements by considering a long term trend.
- Augmenting price data with several indicators on different timeframes.
- Use rule based strategies in combination with stop-loss/take-profit and exit strategies for sophisticated long and
  short trading signals.
- Bring the power of Data Science to trading, by applying machine learning methods to identify optimal entry points for
  trades.
- Run experiments to develop new strategies and analyse the performance with backtesting, cross-validation and plotting
  to understand the pros and cons of your strategies.
- Run in Bot-mode to apply different strategies to real-time data of multiple assets on an hourly basis.
- AIF comes with an option to places trades automatically. WARNING: This feature is for education purpose only.
- Sending alerts for trading signals to your mobile device.

# Installation

**PLEASE READ THE DISCLAIMER FIRST!**

To setup the program, the following steps are mandatory (Tested on AWS t2.micro instance with Ubuntu 22.04).

**Update apt**:

- sudo apt update
- sudo apt upgrade

**Install Conda**:

- wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
- bash Miniconda3-latest-Linux-x86_64.sh
    - Agree terms and install in default location
    - Answer init question after installation: "yes"

**restart shell**: e.g. log-off and log-in again

**Checkout project from git**:

- git clone https://github.com/cliguda/AIF.git
- cd AIF

**Create virtual environment**:

- conda update -n base -c defaults conda
- conda config --set channel_priority strict
- conda env create -f environment.yml
- conda activate aif
- pip install -r requirements.txt

**Setup the project**:

- Copy settings-template.toml to settings.toml
- Copy secrets-template.toml to secrets.toml
- Edit "project_path" in settings.toml
- Create log directory: mkdir log
- Export Python-Path:
    - Linux: export PYTHONPATH=REPLACE_BY_PROJECT_PATH
    - Windows: set PYTHONPATH=REPLACE_BY_PROJECT_PATH

**Run**:

- python scripts/run_update_db.py
- python scripts/run_experiment.py
- python scripts/run_bot.py

*Note*: For running the bot on Linux it is the best to use tmux, so that the program is running even after logging out.

- Start:
    - tmux
    - conda activate aif
    - python scripts/run_bot.py
- Detach: Ctrl+b and then d (Session can be closed afterwards)
- Attach after login: tmux attach

**Activate Push-Notifications**: For push notifications the service https://www.pushsafer.com/ is used.

- Register under https://www.pushsafer.com/
- Download the PushSafer app in the App-Store.
- Register your smartphone.
- Enter the API key into secrets.toml. Warnings, Errors as well as trading signals are pushed to your smartphone.

## Main Concepts

To understand the code the following main concepts are important:

- **PriceData**: The PriceData class holds all data for one asset on a main timeframe (e.g. hourly data). Beside some
  meta-information PriceData can also aggregate the data to higher timeframes (e.g. 4-hourly, daily, ...) and can
  combine all data into a single DataFrame. Beside the main OHLCV data, additional indicators can be added to all
  timeframes. Furthermore, all values that can be put in relation to the closing price, can be converted as such in one
  step. This approach is very beneficial for machine learning, because the relative distance between the closing price
  and e.g. the EMA 20 is more suitable for most ML methods than the absolute value of the EMA 20.
- **Indicator**: All indicators are implemented by inheriting from the Indicator class. While doing so, some relevant
  meta-information are added to PriceData, when the indicator is applied.
- **DataProvider**: The DataProvider is the only class needed to load historical and real-time data for all assets.
  Currently, data from Gemini and Binance are supported, but for real-time data from Binance an API key is necessary.
- **Strategy**: The Strategy class provides a flexible approach to define all kind of strategies. For some examples,
  check the strategies in the library. A strategy is agnostic about assets and timeframes and before a strategy can be
  applied, it needs to be initialized for a concrete PriceData object.
- **StrategyManager**: The StrategyManager class keeps track of all strategies and the assets they should be applied to.
- **PortfolioManager**: The PortfolioManager is the connection for different exchanges. In the current version, API keys
  for ByBit can be provided to place orders and to receive additional trading information for the assets on the
  exchange (e.g. max. leverage, trading fees).

## Where to start

### Update data
After installation, it is a good idea to get up-to-date historical data, by running run_update_db.py in the scripts/
directory.

To get a first understanding of the program, the debug logging can be useful. So switch log_console_level in the
settings to debug.

## Running experiments
As a first test, run "run_experiment.py" in the scripts/ directory. It will load historical data for a given asset and
evaluate the performance of a strategy. Other strategies from strategies/library can be tested or modified, to get a
first understanding of defining and evaluating strategies. For a deeper understanding, use a debugger (e.g. from
PyCharm) to go through the code.<br/>

The script will generate an output like below. The backtest result is the result of the strategy for the whole dataset
while the cross-validation result relates to the current performance as provided in the settings. PPS refers to the
average performance per signal, while the total performance relates to the performance for all signals in the evaluation
period.

**Results:**<br/>
BTCUSD on timeframe HOURLY - Backtesting result: Win-rate: 43.0% / PPS: 3.9% / Total Performance: 931.93%<br/>
BTCUSD on timeframe HOURLY - Cross-Validation result: Win-rate: 54.0% / PPS: 29.1% / Total Performance: 378.26%<br/>
ETHUSD on timeframe HOURLY - Backtesting result: Win-rate: 46.0% / PPS: 11.5% / Total Performance: 2886.47%<br/>
ETHUSD on timeframe HOURLY - Cross-Validation result: Win-rate: 42.0% / PPS: 1.4% / Total Performance: 17.2%<br/>
BNBUSD on timeframe HOURLY - Backtesting result: Win-rate: 45.0% / PPS: 8.2% / Total Performance: 935.66%<br/>
BNBUSD on timeframe HOURLY - Cross-Validation result: Win-rate: 47.0% / PPS: 11.2% / Total Performance: 191.26%<br/>
XRPUSD on timeframe HOURLY - Backtesting result: Win-rate: 45.0% / PPS: 8.5% / Total Performance: 954.86%<br/>
XRPUSD on timeframe HOURLY - Cross-Validation result: Win-rate: 41.0% / PPS: -1.7% / Total Performance: -28.73%<br/>
ADAUSD on timeframe HOURLY - Backtesting result: Win-rate: 39.0% / PPS: -4.1% / Total Performance: -511.13%<br/>
ADAUSD on timeframe HOURLY - Cross-Validation result: Win-rate: 71.0% / PPS: 61.7% / Total Performance: 1048.71%<br/>
SOLUSD on timeframe HOURLY - Backtesting result: Win-rate: 53.0% / PPS: 26.7% / Total Performance: 2267.85%<br/>
SOLUSD on timeframe HOURLY - Cross-Validation result: Win-rate: 53.0% / PPS: 26.7% / Total Performance: 454.0%<br/>

## Plotting
The trades of the strategies can also be plotted by using the plot_data.py script. The script can also plot the
distribution of indicators for winning and losing trade. This can be used to evaluate, if losing trades correlate with
certain ranges of an indicator (e.g. more losing trades, when RSI < 30).

## Running the bot
By using the default configuration, the run_bot.py script uses publicly available data that do not require an API key.
The script starts the bot and will apply all profitable strategies to all assets for generating trading signals on an
hourly basis.

**Stay tuned for more content coming soon....**
