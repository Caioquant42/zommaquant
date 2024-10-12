import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta

def print_description():
    print("""
    ====== Otimizador de Portfólio ======

    Este programa otimiza um portfólio de ações brasileiras com base nos seguintes critérios:

    1. Maximiza o retorno esperado para um nível de risco específico.
    2. Minimiza o risco (volatilidade) para um retorno alvo.
    3. Permite definir uma alocação máxima por ativo para garantir diversificação.

    O programa irá solicitar as seguintes informações:
    - Lista de tickers das ações (ex: PETR4, VALE3, BBAS3, B3SA3)
    - Data de início para análise histórica
    - Retorno anual desejado
    - Alocação máxima por ativo

    Com base nesses dados, o programa irá:
    1. Coletar dados históricos das ações
    2. Calcular retornos esperados e matriz de covariância
    3. Otimizar o portfólio
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

def optimize_portfolio(returns, cov_matrix, target_return, constraints):
    num_assets = len(returns)
    
    def objective(weights):
        return portfolio_volatility(weights, cov_matrix)
    
    def constraint_return(weights):
        return portfolio_return(weights, returns) - target_return
    
    def constraint_sum(weights):
        return np.sum(weights) - 1.0
    
    constraints.append({'type': 'eq', 'fun': constraint_return})
    constraints.append({'type': 'eq', 'fun': constraint_sum})
    
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    result = minimize(objective, num_assets*[1./num_assets,], method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def main():
    
    print_description()  # Adicionamos esta linha para imprimir a descrição
    # Lista de ações
    tickers_input = input("Digite os tickers das ações separados por vírgula (ex: PETR4,VALE3,BBAS3,B3SA3): ")
    acoes = [ticker.strip() for ticker in tickers_input.split(',')]

    # Solicitar a data de início ao usuário
    start_date_str = input("Digite a data de início para a análise (formato YYYY-MM-DD): ")
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Solicitando o retorno alvo ao usuário
    target_return = float(input("Digite o retorno anual desejado (em decimal, ex: 0.25 para 25%): "))

    #Máximo de alocação em 1 ativo
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

    # Calcular volatilidades anuais
    volatilidades = retornos_diarios.std() * np.sqrt(252)

    # Calcular matriz de correlação
    correlacao = retornos_diarios.corr()

    # Calculando a matriz de covariância
    cov_matrix = np.outer(volatilidades, volatilidades) * correlacao

    # Definindo restrições adicionais (exemplo: limite máximo por ação)
    constraints = [{'type': 'ineq', 'fun': lambda w: max_allocation - w[i]} for i in range(len(acoes))]

    # Otimizando o portfólio
    result = optimize_portfolio(retornos_esperados.values, cov_matrix.values, target_return, constraints)

    if result.success:
        print("\nAlocação ótima do portfólio:")
        for acao, peso in zip(acoes, result.x):
            print(f"{acao}: {peso*100:.2f}%")
        
        print(f"\nRetorno esperado do portfólio: {portfolio_return(result.x, retornos_esperados.values)*100:.2f}%")
        print(f"Volatilidade do portfólio: {portfolio_volatility(result.x, cov_matrix.values)*100:.2f}%")
    else:
        print("A otimização não foi bem-sucedida. O retorno alvo pode ser inatingível com as restrições atuais.")

if __name__ == "__main__":
    main()
