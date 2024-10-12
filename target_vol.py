import numpy as np
import pandas as pd
import yfinance as yf
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from datetime import datetime, timedelta

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

def optimize_portfolio(tickers, start_date, end_date, target_volatility, max_allocation):
    # Verificar se o período é de pelo menos 1 ano
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    if (end_date - start_date).days < 365:
        print("O período deve ser de pelo menos 1 ano. Ajustando a data de início...")
        start_date = end_date - timedelta(days=365)

    print(f"Período de análise: de {start_date.date()} até {end_date.date()}")

    # Baixar dados históricos
    dados_historicos = {}
    for ticker in tickers:
        dados = ydata(ticker_symbol=ticker, interval='1d', start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d")).get_stock_data()
        dados_historicos[ticker] = dados['Fechamento']

    data = pd.DataFrame(dados_historicos)
    
    # Calcular retornos esperados e matriz de covariância
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)
    
    # Criar o objeto EfficientFrontier
    ef = EfficientFrontier(mu, S)
    
    # Adicionar restrição de alocação máxima
    ef.add_constraint(lambda w: w <= max_allocation)
    
    # Otimizar para a volatilidade alvo
    weights = ef.efficient_risk(target_volatility)
    
    # Obter o desempenho do portfólio
    performance = ef.portfolio_performance()
    
    return weights, performance

def main():
    # Lista de ações
    tickers_input = input("Digite os tickers das ações separados por vírgula (ex: PETR4,VALE3,BBAS3,B3SA3): ")
    tickers = [ticker.strip() for ticker in tickers_input.split(',')]

    # Solicitar a data de início ao usuário
    start_date_str = input("Digite a data de início para a análise (formato YYYY-MM-DD): ")
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Solicitando a volatilidade alvo ao usuário
    target_volatility = float(input("Digite a volatilidade anual desejada (em decimal, ex: 0.25 para 25%): "))

    # Máximo de alocação em 1 ativo
    max_allocation =  float(input("Digite a alocação máxima por ativo (em decimal, ex: 0.30 para 30%): "))

    weights, performance = optimize_portfolio(tickers, start_date_str, end_date, target_volatility, max_allocation)

    print("\nPesos otimizados do portfólio:")
    for ticker, weight in weights.items():
        print(f"{ticker}: {weight:.4f}")

    print(f"\nRetorno esperado anual: {performance[0]:.4f}")
    print(f"Volatilidade anual: {performance[1]:.4f}")
    print(f"Sharpe Ratio: {performance[2]:.4f}")

if __name__ == "__main__":
    main()