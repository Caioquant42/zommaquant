import numpy as np
import pandas as pd
from scipy.optimize import linprog
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from scipy.optimize import OptimizeWarning

warnings.filterwarnings("ignore", category=OptimizeWarning)

def print_description():
    print("""
    ====== Otimizador de Portfólio (Versão Simplex) ======

    Este programa otimiza um portfólio de ações brasileiras usando o método Simplex.

    O programa irá solicitar as seguintes informações:
    - Lista de tickers das ações (ex: PETR4, VALE3, BBAS3, B3SA3)
    - Data de início para análise histórica
    - Retorno anual desejado
    - Alocação máxima por ativo

    Com base nesses dados, o programa irá:
    1. Coletar dados históricos das ações
    2. Calcular retornos esperados e matriz de covariância
    3. Otimizar o portfólio usando o método Simplex
    4. Apresentar a alocação ótima, retorno esperado e volatilidade do portfólio

    Observação: Este programa utiliza dados históricos e não garante resultados futuros.
    Sempre consulte um profissional financeiro antes de tomar decisões de investimento.

    =====================================
    """)

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

    def check_interval(self):
        valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        if self.interval not in valid_intervals:
            raise ValueError("Intervalo não disponível, opções válidas: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo")
        if self.period == 'max':
            if self.interval == '1h':
                self.period = '730d'
            elif self.interval == '1m':
                self.period = '7d'
            elif self.interval in ['1d', '1wk', '1mo', '3mo']:
                self.period = 'max'
            elif self.interval in ['90m', '30m', '15m', '5m']:
                self.period = '60d'
            else:
                raise ValueError("Erro: Período Inválido.")
            return self.period
        else:
            return self.period

    def get_stock_data(self):    
        ticker = self._add_sa_to_tickers(self.ticker_symbol)
        stock_data = yf.Ticker(ticker)
        
        if self.start_date and self.end_date:
            historical_data = stock_data.history(start=self.start_date, end=self.end_date, interval=self.interval)
        else:
            period = self.check_interval()
            historical_data = stock_data.history(period=period, interval=self.interval)
        
        rename_cols = ['Abertura', 'Máxima', 'Mínima', 'Fechamento', 'Volume', 'Dividendos', 'Desdobramentos']
        historical_data = historical_data.rename(columns=dict(zip(historical_data.columns, rename_cols)))
        return historical_data


def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def portfolio_return(weights, returns):
    return np.sum(returns * weights)

def optimize_portfolio_simplex(returns, cov_matrix, target_return, max_allocation):
    num_assets = len(returns)
    
    # Objetivo: minimizar a soma dos pesos (uma aproximação para minimizar a variância)
    c = np.ones(num_assets)
    
    # Restrições de igualdade
    A_eq = np.array([
        returns,  # Restrição de retorno alvo
        np.ones(num_assets)  # Soma dos pesos = 1
    ])
    b_eq = np.array([target_return, 1])
    
    # Restrições de desigualdade (alocação máxima por ativo)
    A_ub = np.eye(num_assets)
    b_ub = np.full(num_assets, max_allocation)
    
    # Limites das variáveis
    bounds = [(0, max_allocation) for _ in range(num_assets)]
    
    # Otimização usando o método Simplex
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='simplex')
        
        if result.success:
            return result.x, None
        else:
            return None, "Otimização não convergiu: " + result.message
    except Exception as e:
        return None, f"Erro na otimização: {str(e)}"

def optimize_portfolio(tickers, start_date, end_date, target_return, max_allocation):
    acoes = tickers
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Buscar dados históricos
    dados_historicos = {}
    for acao in acoes:
        dados = ydata(ticker_symbol=acao, interval='1d', start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d")).get_stock_data()
        dados_historicos[acao] = dados['Fechamento']

    # Criar um DataFrame com os preços de fechamento
    df_precos = pd.DataFrame(dados_historicos)

    # Calcular retornos diários
    retornos_diarios = df_precos.pct_change().dropna()

    # Calcular retornos anuais esperados
    retornos_esperados = retornos_diarios.mean() * 252

    # Calcular matriz de covariância
    cov_matrix = retornos_diarios.cov() * 252
    
        # Otimizando o portfólio usando o método Simplex
    weights, error_message = optimize_portfolio_simplex(retornos_esperados.values, cov_matrix.values, target_return, max_allocation)

    if weights is not None:
        retorno_otimo = portfolio_return(weights, retornos_esperados.values)
        volatilidade_otima = portfolio_volatility(weights, cov_matrix.values)
        sharpe_ratio = (retorno_otimo - 0.05) / volatilidade_otima  # Assumindo taxa livre de risco de 5%

        pesos = {acao: peso for acao, peso in zip(acoes, weights)}
        desempenho = (retorno_otimo, volatilidade_otima, sharpe_ratio)
        return pesos, desempenho
    else:
        return None, error_message

def main():
    print_description()

    # Lista de ações
    tickers_input = input("Digite os tickers das ações separados por vírgula (ex: PETR4,VALE3,BBAS3,B3SA3): ")
    acoes = [ticker.strip() for ticker in tickers_input.split(',')]

    # Solicitar a data de início ao usuário
    start_date_str = input("Digite a data de início para a análise (formato YYYY-MM-DD): ")
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Solicitando o retorno alvo ao usuário
    target_return = float(input("Digite o retorno anual desejado (em decimal, ex: 0.25 para 25%): "))

    # Máximo de alocação em 1 ativo
    max_allocation = float(input("Digite a alocação máxima por ativo (em decimal, ex: 0.30 para 30%): "))

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.now()

    # Verificar se o período é de pelo menos 1 ano
    if (end_date - start_date).days < 365:
        print("O período deve ser de pelo menos 1 ano. Ajustando a data de início...")
        start_date = end_date - timedelta(days=365)

    print(f"Período de análise: de {start_date.date()} até {end_date.date()}")
  # Buscar dados históricos
    dados_historicos = {}
    for acao in acoes:
        dados = ydata(ticker_symbol=acao, interval='1d', start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d")).get_stock_data()
        dados_historicos[acao] = dados['Fechamento']

    # Criar um DataFrame com os preços de fechamento
    df_precos = pd.DataFrame(dados_historicos)

    # Calcular retornos diários
    retornos_diarios = df_precos.pct_change().dropna()

    # Calcular retornos anuais esperados
    retornos_esperados = retornos_diarios.mean() * 252

    # Calcular matriz de covariância
    cov_matrix = retornos_diarios.cov() * 252

    # Otimizando o portfólio usando o método Simplex
    weights, error_message = optimize_portfolio_simplex(retornos_esperados.values, cov_matrix.values, target_return, max_allocation)

    if weights is not None:
        print("\nAlocação ótima do portfólio:")
        for acao, peso in zip(acoes, weights):
            print(f"{acao}: {peso*100:.2f}%")

        retorno_otimo = portfolio_return(weights, retornos_esperados.values)
        volatilidade_otima = portfolio_volatility(weights, cov_matrix.values)

        print(f"\nRetorno esperado do portfólio: {retorno_otimo*100:.2f}%")
        print(f"Volatilidade do portfólio: {volatilidade_otima*100:.2f}%")
    else:
        print(f"\nNão foi possível encontrar uma solução ótima. {error_message}")
        print("\nRetornos esperados anuais das ações:")
        for acao, retorno in zip(acoes, retornos_esperados):
            print(f"{acao}: {retorno*100:.2f}%")

if __name__ == "__main__":
    main()