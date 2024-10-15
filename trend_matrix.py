import warnings
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import seaborn as sns

# Suppress specific warnings
warnings.filterwarnings("ignore", message='.*zipline.assets.*')

def print_description():
    print("""
    ====== Trend Matrix Calculator ======

    This program calculates a trend matrix for a list of Brazilian stocks based on various timeframes.

    The program will ask for the following information:
    - List of stock tickers (e.g., PETR4, VALE3, BBAS3, B3SA3)
    - Start date for historical analysis
    - End date for historical analysis
    - Timeframes to analyze (e.g., 1d, 1wk)

    Based on this data, the program will:
    1. Collect historical data for the stocks
    2. Calculate trend indicators
    3. Generate a trend matrix
    4. Display the trend matrix as a heatmap

    Note: This program uses historical data and does not guarantee future results.
    Always consult a financial professional before making investment decisions.

    =============================
    ========
    """)

IBRX_50_list = ['ALOS3', 'ABEV3', 'ASAI3', 'AZUL4', 'AZZA3', 'B3SA3', 'BBSE3', 'BBDC4', 'BBAS3', 'BRAV3', 'BRFS3',
                'BPAC11', 'CMIG4', 'CPLE6', 'CSAN3', 'CYRE3', 'ELET3', 'EMBR3', 'ENGI11', 'EQTL3', 'GGBR4', 'NTCO3', 'HAPV3', 'HYPE3', 'ITSA4',
                'ITUB4', 'JBSS3', 'KLBN11', 'RENT3', 'LREN3', 'MGLU3', 'MRVE3', 'MULT3', 'PETR3', 'PETR4', 'PRIO3', 'RADL3', 'RDOR3', 'RAIL3', 'SBSP3',
                'CSNA3', 'SUZB3', 'VIVT3', 'TIMS3', 'TOTS3', 'UGPA3', 'USIM5', 'VALE3', 'VBBR3', 'WEGE3']

IBOV_list =  [
    'VALE3', 'PETR4', 'ITUB4', 'PETR3', 'BBAS3', 'ELET3', 'BBDC4', 'B3SA3', 'WEGE3', 'ITSA4', 
    'ABEV3', 'RENT3', 'SUZB3', 'BPAC11', 'PRIO3', 'EQTL3', 'RADL3', 'UGPA3', 'RDOR3', 'SBSP3', 
    'VBBR3', 'BRFS3', 'RAIL3', 'GGBR4', 'JBSS3', 'EMBR3', 'BBSE3', 'VIVT3', 'ENEV3', 'BBDC3', 
    'ASAI3', 'CMIG4', 'CSAN3', 'KLBN11', 'HAPV3', 'CPLE6', 'LREN3', 'ENGI11', 'NTCO3', 'TIMS3', 
    'TOTS3', 'CCRO3', 'ALOS3', 'HYPE3', 'ELET6', 'EGIE3', 'TRPL4', 'SANB11', 'CSNA3', 'RRRP3', 
    'TAEE11', 'CRFB3', 'GOAU4', 'MULT3', 'CPFE3', 'BRKM5', 'CYRE3', 'CIEL3', 'CMIN3', 'RECV3', 
    'MGLU3', 'USIM5', 'BRAP4', 'IGTI11', 'YDUQ3', 'SMTO3', 'AZUL4', 'COGN3', 'RAIZ4', 'SLCE3', 
    'FLRY3', 'ARZZ3', 'MRFG3', 'VAMO3', 'IRBR3', 'MRVE3', 'DXCO3', 'LWSA3', 'BEEF3', 
    'ALPA4', 'EZTC3', 'CVCB3', 'PETZ3', 'PCAR3', 'BHIA3'
]

#Top 50 tickers with most liquidity in the options market
OPCOES_TOP_50 = [
    "PETR4", "VALE3", "BOVA11", "BBAS3", "ITUB4", "BBDC4", "B3SA3", "MGLU3", "SUZB3",
    "SBSP3", "EQTL3", "ABEV3", "RRRP3", "WEGE3", "PRIO3", "ELET3", "LREN3", "BPAC11", "RENT3",
    "PETZ3", "ALOS3", "GGBR4", "PETR3", "JBSS3", "IRBR3", "CSAN3", "ELET6", "BHIA3", "ITSA4",
    "SMAL11", "EMBR3", "EZTC3", "ARZZ3", "JHSF3", "CSNA3", "USIM5", "BEEF3", "BOVV11", "BBDC3",
    "COGN3", "BRFS3", "KLBN11", "BRAP4", "BRKM5", "VBBR3", "CMIG4", "CEAB3", "AZUL4", "CYRE3"
] 
class ydata:
    def __init__(self, ticker_symbol, interval='1d', period='max', world=False, start_date=None, end_date=None):
        self.ticker_symbol = ticker_symbol
        self.interval = interval
        self.period = period
        self.world = world
        self.start_date = start_date
        self.end_date = end_date

    def _add_sa_to_tickers(self, tickers):
        if not self.world:
            return tickers + '.SA'
        else:
            return tickers

    def get_stock_data(self):    
        ticker = self._add_sa_to_tickers(self.ticker_symbol)
        stock_data = yf.Ticker(ticker)
        
        if self.start_date and self.end_date:
            historical_data = stock_data.history(start=self.start_date, end=self.end_date, interval=self.interval)
        else:
            historical_data = stock_data.history(period=self.period, interval=self.interval)
        
        rename_cols = ['Abertura', 'Máxima', 'Mínima', 'Fechamento', 'Volume', 'Dividendos', 'Desdobramentos']
        historical_data = historical_data.rename(columns=dict(zip(historical_data.columns, rename_cols)))
        return historical_data

def window_size_calculator(time_frame):
    if time_frame == '1d':
        return 252
    elif time_frame == '1wk':
        return 52
    elif time_frame == '1mo':
        return 12
    else:
        return 252  # Default to daily

def trend_matrix_df(stock_df, interval):
    window_size = window_size_calculator(interval)
    stock_df['EMA_17'] = stock_df['Fechamento'].ewm(span=17, adjust=False).mean()
    stock_df['EMA_34'] = stock_df['Fechamento'].ewm(span=34, adjust=False).mean()
    stock_df['EMA_72'] = stock_df['Fechamento'].ewm(span=72, adjust=False).mean()
    stock_df['EMA_144'] = stock_df['Fechamento'].ewm(span=144, adjust=False).mean()
    stock_df['EMA_305'] = stock_df['Fechamento'].ewm(span=305, adjust=False).mean()
    
    # Calculate the rate of change for each EMA and create new columns
    for ema in ['17', '34', '72', '144', '305']:
        stock_df[f'EMA_{ema}_ROC'] = stock_df[f'EMA_{ema}'].diff()
        stock_df[f'EMA_ROC_{ema}_degrees'] = np.degrees(np.arctan(stock_df[f'EMA_{ema}_ROC']))

    stock_df['Weighted_ROC_degrees'] = (
        stock_df['EMA_ROC_17_degrees'] * 17 + 
        stock_df['EMA_ROC_34_degrees'] * 34 + 
        stock_df['EMA_ROC_72_degrees'] * 72 +
        stock_df['EMA_ROC_144_degrees'] * 144 +
        stock_df['EMA_ROC_305_degrees'] * 305
    ) / (17 + 34 + 72 + 144 + 305)
    
    return stock_df

def heat_map(matrix,title):
    current_datetime = datetime.datetime.now()
    current_datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')   
    # Convert all values in the DataFrame to numeric
    matrix_numeric = matrix.apply(pd.to_numeric, errors='coerce')
    
    # Create a heatmap of the correlation matrix with ticker symbols for both axes
    plt.figure(figsize=(70, 20))  # Adjust the figure size as needed
    sns.heatmap(matrix_numeric, annot=True, cmap='coolwarm_r', linewidths=0.9, fmt=".2f", annot_kws={"size": 9})
    plt.title(f"{title} Trend Matrix {current_datetime_str}", fontsize=14)
    plt.xlabel("Time Frame", fontsize=14)
    plt.ylabel("Ativos", fontsize=14)
    plt.xticks(fontsize=8)  # Adjust font size of x-axis tick labels
    plt.yticks(fontsize=8)  # Adjust font size of y-axis tick labels
    plt.show()

def calculate_trend_matrix(indice, time_frame, start_date, end_date):
    result_df = pd.DataFrame(index=indice, columns=time_frame)
    for interval in time_frame:
        for symbol in indice:
            try:
                data = ydata(ticker_symbol=symbol, interval=interval, world=False, start_date=start_date, end_date=end_date).get_stock_data()
                print(f'{symbol} data loaded successfully at {interval} time_frame')
                
                underlying_df = trend_matrix_df(stock_df=data, interval=interval)
                weighted_roc_degrees = underlying_df['Weighted_ROC_degrees'].iloc[-1]
                median = np.nanmedian(underlying_df['Weighted_ROC_degrees'])
                mad = np.nanmean(np.abs((underlying_df['Weighted_ROC_degrees'] - median)))
                relative_w_roc = (weighted_roc_degrees - median) / mad

                result_df.loc[symbol, interval] = relative_w_roc
            except Exception as e:
                print(f"Failed to fetch data for {symbol} at {interval} time_frame: {str(e)}")
                continue
    return result_df

def choose_stocks():
    print("\nChoose an option for stock selection:")
    print("1. Enter your own stock tickers")
    print("2. Use IBRX_50_list")
    print("3. Use IBOV_list")
    print("4. Use OPCOES_TOP_50 list")

    choice = input("Enter your choice (1-5): ")

    if choice == '1':
        tickers_input = input("Enter stock tickers separated by comma (e.g., PETR4,VALE3,BBAS3,B3SA3): ")
        return [ticker.strip() for ticker in tickers_input.split(',')]
    elif choice == '2':
        return IBRX_50_list
    elif choice == '3':
        return IBOV_list
    elif choice == '4':
        return OPCOES_TOP_50
    else:
        print("Invalid choice. Using IBRX_50_list as default.")
        return IBRX_50_list

def main():
    print_description()

    # Get user inputs for stock selection
    stocks = choose_stocks()

    start_date = input("Enter start date for analysis (YYYY-MM-DD): ")
    end_date = input("Enter end date for analysis (YYYY-MM-DD), or press Enter for today: ")
    if not end_date:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    timeframes = ['1d','1wk']

    # Calculate trend matrix
    trend_matrix = calculate_trend_matrix(stocks, timeframes, start_date, end_date)  

    # Display heatmap
    heat_map(trend_matrix, title='Zomma Quant')

if __name__ == "__main__":
    main()

    