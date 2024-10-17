import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

#Top 50 tickers with most liquidity in the options market
OPCOES_TOP_50 = [
    "PETR4", "VALE3", "BOVA11", "BBAS3", "ITUB4", "BBDC4", "B3SA3", "MGLU3", "SUZB3",
    "SBSP3", "EQTL3", "ABEV3", "RRRP3", "WEGE3", "PRIO3", "ELET3", "LREN3", "BPAC11", "RENT3",
    "PETZ3", "ALOS3", "GGBR4", "PETR3", "JBSS3", "IRBR3", "CSAN3", "ELET6", "BHIA3", "ITSA4",
    "SMAL11", "EMBR3", "EZTC3", "ARZZ3", "JHSF3", "CSNA3", "USIM5", "BEEF3", "BOVV11", "BBDC3",
    "COGN3", "BRFS3", "KLBN11", "BRAP4", "BRKM5", "VBBR3", "CMIG4", "CEAB3", "AZUL4", "CYRE3"
] 
class YData:
    def __init__(self, ticker_symbol, interval='1d', period='max', world=False, start_date=None, end_date=None):
        self.ticker_symbol = ticker_symbol
        self.interval = interval
        self.period = period
        self.world = world
        self.start_date = start_date
        self.end_date = end_date

    def _add_sa_to_tickers(self, tickers):
        return f"{tickers}.SA" if not self.world else tickers

    def get_stock_data(self):
        ticker = self._add_sa_to_tickers(self.ticker_symbol)
        stock_data = yf.Ticker(ticker)

        # Fetch historical data based on date range or period
        if self.start_date and self.end_date:
            historical_data = stock_data.history(start=self.start_date, end=self.end_date, interval=self.interval)
        else:
            historical_data = stock_data.history(period=self.period, interval=self.interval)

        # Rename columns for clarity
        rename_cols = {
            'Open': 'Abertura', 
            'High': 'Máxima', 
            'Low': 'Mínima', 
            'Close': 'Fechamento', 
            'Volume': 'Volume', 
            'Dividends': 'Dividendos', 
            'Stock Splits': 'Desdobramentos'
        }
        historical_data.rename(columns=rename_cols, inplace=True)

        # Calculate returns
        historical_data['simple_return'] = historical_data['Fechamento'].pct_change()
        historical_data['log_return'] = np.log1p(historical_data['simple_return'])
        historical_data.dropna(subset=['simple_return', 'log_return'], inplace=True)

        return historical_data

def last_incident(historical_data, threshold=-0.05):
    below_threshold = historical_data[historical_data['log_return'] < threshold]
    timestamps_sorted = below_threshold.index.sort_values()
    interarrival_time = timestamps_sorted.to_series().diff().dropna()
    interarrival_days = interarrival_time.dt.days

    today = datetime.today().date()
    last_available_date = timestamps_sorted.max().date()
    running_days = np.busday_count(last_available_date, today)

    return running_days

def hist_interarrival(historical_data, threshold=-0.05):
    below_threshold = historical_data[historical_data['log_return'] < threshold]
    timestamps_sorted = below_threshold.index.sort_values()
    interarrival_time = timestamps_sorted.to_series().diff().dropna()
    return interarrival_time.dt.days

def mean_residual_life(interarrival_days, specific_day=None):
    """
    Calculate the mean residual life for each unique interarrival time or a specific day.
    :param interarrival_days: Series of interarrival times.
    :param specific_day: Specific day of survival to calculate MRL for, optional.
    :return: DataFrame with interarrival time and corresponding mean residual life,
             or a single MRL value if specific_day is provided.
    """
    if specific_day is not None:
        # Calculate MRL for the specific day
        residuals = interarrival_days[interarrival_days >= specific_day] - specific_day
        if residuals.empty:
            return 0.0  # Return 0 if residuals are empty
        mrl = residuals.mean()
        return mrl

    # Calculate MRL for all unique interarrival times
    unique_times = np.sort(interarrival_days.unique())
    mrl_values = []

    for t in unique_times:
        residuals = interarrival_days[interarrival_days >= t] - t
        if residuals.empty:
            mrl = 0.0  # Set MRL to 0 if residuals are empty
        else:
            mrl = residuals.mean()
        mrl_values.append(mrl)

    mrl_df = pd.DataFrame({'Interarrival Time': unique_times, 'Mean Residual Life': mrl_values})
    return mrl_df

def kaplan_meier_estimator_with_cumulative_hazard(interarrival_days, specific_day=None):
    interarrival_days = interarrival_days.sort_values()

    if interarrival_days.empty:
        return 1.0, 0.0, 0.0  # Return default values when no data points meet the criteria

    n = len(interarrival_days)
    event_counts = interarrival_days.value_counts().sort_index()
    
    survival_prob = 1.0
    cumulative_hazard = 0.0
    survival_probs = []
    hazard_rates = []
    cumulative_hazards = []

    for t, d in event_counts.items():
        at_risk = n
        survival_prob *= (1 - d / at_risk)
        survival_probs.append(survival_prob)
        hazard_rate = d / at_risk
        hazard_rates.append(hazard_rate)
        cumulative_hazard += hazard_rate
        cumulative_hazards.append(cumulative_hazard)
        n -= d

    km_df = pd.DataFrame({
        'Time': event_counts.index,
        'Survival Probability': survival_probs,
        'Hazard Rate': hazard_rates,
        'Cumulative Hazard': cumulative_hazards
    })
    
    if specific_day is not None:
        if km_df.empty:
            return 1.0, 0.0, 0.0  # Return default values when no data points meet the criteria
        closest_day_index = (km_df['Time'] - specific_day).abs().idxmin()
        closest_day = km_df.loc[closest_day_index]
        return closest_day['Survival Probability'], closest_day['Hazard Rate'], closest_day['Cumulative Hazard']

    return km_df

def plot_occurence(historical_data, threshold = -0.05, title: str = None):
    # Filter the data for log returns below the threshold
    below = historical_data[historical_data['log_return'] < threshold]
    
    # Create a scatter plot using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(below.index, below['log_return'], color='red', edgecolor='black', s=80, label=f'Log Returns < {threshold}')
    
    # Set the title and labels
    ax.set_title(title if title else f'Retornos Abaixo de {threshold}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Log Return')
    ax.grid(True)
    
    # Add a legend
    ax.legend(title=f'Log Returns < {threshold}')
    
    return fig

def plot_interarrival(historical_data, threshold=-0.05, title='Title'):
    below_threshold = historical_data[historical_data['log_return'] < threshold]
    timestamps_sorted = below_threshold.index.sort_values()
    interarrival_time = timestamps_sorted.to_series().diff().dropna()
    interarrival_days = interarrival_time.dt.days

    today = datetime.today().date()
    last_available_date = timestamps_sorted.max().date()
    running_days = np.busday_count(last_available_date, today)

    # Plot the histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(interarrival_days, bins=40, color='skyblue', edgecolor='black')
    ax.set_title(f'{title} Interarrival Time for Negative Returns Below {threshold}')
    ax.set_xlabel('Interarrival Time (days)')
    ax.set_ylabel('Frequency')
    ax.grid(True)

    # Plot a vertical line for running_days
    ax.axvline(x=running_days, color='red', linestyle='--', label=f'Running Business Days Since Last Incident: {running_days}')
    ax.legend()

    return fig

def process_tickers(tickers, threshold=-0.05):
    results = []
    for ticker in tickers:
        try:
            ydata = YData(ticker)
            stock_data = ydata.get_stock_data()
            interarrival_days = hist_interarrival(stock_data, threshold)

            running_days = last_incident(stock_data, threshold)

            if interarrival_days.empty:
                results.append({
                    'Ticker': ticker,
                    'Running Days Since Last Incident': running_days,
                    'Mean Residual Life': 0,
                    'Survival Probability': 1.0,
                    'Hazard Rate': 0.0,
                    'Cumulative Hazard': 0.0
                })
            else:
                mrl = mean_residual_life(interarrival_days, specific_day=running_days)
                survival_prob, hazard_rate, cumulative_hazard = kaplan_meier_estimator_with_cumulative_hazard(interarrival_days, specific_day=running_days)

                results.append({
                    'Ticker': ticker,
                    'Running Days Since Last Incident': running_days,
                    'Mean Residual Life': mrl,
                    'Survival Probability': survival_prob,
                    'Hazard Rate': hazard_rate,
                    'Cumulative Hazard': cumulative_hazard
                })
        except Exception as e:
            print(f"Error processing ticker {ticker}: {str(e)}")
            results.append({
                'Ticker': ticker,
                'Running Days Since Last Incident': 'N/A',
                'Mean Residual Life': 'N/A',
                'Survival Probability': 'N/A',
                'Hazard Rate': 'N/A',
                'Cumulative Hazard': 'N/A'
            })

    results_df = pd.DataFrame(results)
    return results_df

__all__ = ['YData', 'last_incident', 'hist_interarrival', 'mean_residual_life', 
           'kaplan_meier_estimator_with_cumulative_hazard', 'plot_occurence', 
           'plot_interarrival', 'process_tickers', 'OPCOES_TOP_50']