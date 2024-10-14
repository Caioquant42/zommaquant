import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from io import BytesIO
# Import functions from return_target.py
from return_target import ydata, portfolio_volatility, portfolio_return, optimize_portfolio, minimum_variance_portfolio

def portfolio_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
    port_return = portfolio_return(weights, returns)
    port_volatility = portfolio_volatility(weights, cov_matrix)
    return (port_return - risk_free_rate) / port_volatility

def max_sharpe_ratio_portfolio(returns, cov_matrix, risk_free_rate, additional_constraints=None):
    num_assets = len(returns)
    
    def negative_sharpe_ratio(weights):
        return -portfolio_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate)
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    ]
    
    if additional_constraints:
        constraints.extend(additional_constraints)
    
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    initial_weights = np.array([1.0/num_assets] * num_assets)
    
    result = minimize(negative_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x
def calculate_cumulative_returns(df_precos, weights):
    portfolio_value = (df_precos * weights).sum(axis=1)
    cumulative_returns = (1 + portfolio_value.pct_change()).cumprod()
    return cumulative_returns
# Theme configuration
st.set_page_config(
    page_title="Otimizador de Portfólio Zomma Quant - SLSQP",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom theme
st.markdown("""
    <style>
    .reportview-container {
        background-color: #9fcaf7;
    }
    .sidebar .sidebar-content {
        background-color: #F0F2F6;
    }
    .Widget>label {
        color: #262730;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #FF4B4B;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title('Otimizador de Portfólio Zomma Quant - SLSQP')

# Entrada para os tickers
tickers_input = st.text_input("Digite os tickers das ações separados por vírgula (ex: PETR4,VALE3,BBAS3,B3SA3)")

# Calcula os intervalos de datas
hoje = datetime.now().date()
data_maxima = hoje - timedelta(days=366)  # Um ano atrás
data_minima = hoje - timedelta(days=3650)  # Aproximadamente 10 anos atrás

# Usa data_maxima como valor padrão
data_inicio = st.date_input("Selecione a data de início para análise", 
                            value=data_maxima,
                            min_value=data_minima,
                            max_value=data_maxima)

data_fim = hoje

# Slider para retorno alvo
retorno_alvo = st.slider("Selecione o retorno anual alvo", 0.0, 1.0, 0.25, 0.01)

# Slider para alocação máxima
alocacao_maxima = st.slider("Selecione a alocação máxima por ativo", 0.0, 1.0, 1.0, 0.01)


if st.button('Otimizar Portfólio'):
    if tickers_input:
        tickers = [ticker.strip() for ticker in tickers_input.split(',')]
        
        # Buscar dados históricos
        dados_historicos = {}
        for acao in tickers:
            dados = ydata(ticker_symbol=acao, interval='1d', start_date=data_inicio.strftime("%Y-%m-%d"), end_date=data_fim.strftime("%Y-%m-%d")).get_stock_data()
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

        additional_constraints = [{'type': 'ineq', 'fun': lambda w: alocacao_maxima - w[i]} for i in range(len(tickers))]
        result = optimize_portfolio(retornos_esperados.values, cov_matrix.values, retorno_alvo, additional_constraints)

        if result.success:

            # Calculate maximum Sharpe ratio portfolio
            risk_free_rate = 0  # You can make this a user input if desired
            additional_constraints = [{'type': 'ineq', 'fun': lambda w: alocacao_maxima - w[i]} for i in range(len(tickers))]
            min_var_weights = minimum_variance_portfolio(cov_matrix.values, additional_constraints)
            max_sharpe_weights = max_sharpe_ratio_portfolio(retornos_esperados.values, cov_matrix.values, risk_free_rate, additional_constraints)           
            max_sharpe_return = portfolio_return(max_sharpe_weights, retornos_esperados.values)
            max_sharpe_volatility = portfolio_volatility(max_sharpe_weights, cov_matrix.values)
            max_sharpe_ratio = portfolio_sharpe_ratio(max_sharpe_weights, retornos_esperados.values, cov_matrix.values, risk_free_rate)

            st.subheader('Pesos Otimizados do Portfólio:')
            df_pesos = pd.DataFrame(list(zip(tickers, result.x * 100)), columns=['Ativo', 'Peso'])
            df_pesos['Peso (%)'] = df_pesos['Peso'].apply(lambda x: f'{x:.2f}%')
            st.table(df_pesos[['Ativo', 'Peso (%)']])
            
            retorno_esperado = portfolio_return(result.x, retornos_esperados.values)
            volatilidade = portfolio_volatility(result.x, cov_matrix.values)
            
            st.subheader('Desempenho do Portfólio:')
            st.write(f"Retorno Anual Esperado: {retorno_esperado*100:.2f}%")
            st.write(f"Volatilidade Anual: {volatilidade*100:.2f}%")
            
            # Cria um gráfico de pizza dos pesos do portfólio
            fig = px.pie(df_pesos, values='Peso', names='Ativo', title='Alocação do Portfólio')
            st.plotly_chart(fig)

                # Calculate minimum variance portfolio
           
            min_var_return = portfolio_return(min_var_weights, retornos_esperados.values)
            min_var_volatility = portfolio_volatility(min_var_weights, cov_matrix.values)

            st.subheader('Portfólio de Variância Mínima:')
            df_min_var = pd.DataFrame(list(zip(tickers, min_var_weights * 100)), columns=['Ativo', 'Peso'])
            df_min_var['Peso (%)'] = df_min_var['Peso'].apply(lambda x: f'{x:.2f}%')
            st.table(df_min_var[['Ativo', 'Peso (%)']])

            st.write(f"Retorno Anual Esperado: {min_var_return*100:.2f}%")
            st.write(f"Volatilidade Anual: {min_var_volatility*100:.2f}%")

            # Create a pie chart for the minimum variance portfolio
            fig_min_var = px.pie(df_min_var, values='Peso', names='Ativo', title='Alocação do Portfólio de Variância Mínima')
            st.plotly_chart(fig_min_var)

            # Compare the two portfolios
            st.subheader('Comparação dos Portfólios:')
            comparison_data = {
                'Portfólio': ['Retorno Alvo', 'Variância Mínima'],
                'Retorno Esperado': [f"{retorno_esperado*100:.2f}%", f"{min_var_return*100:.2f}%"],
                'Volatilidade': [f"{volatilidade*100:.2f}%", f"{min_var_volatility*100:.2f}%"]
            }
            df_comparison = pd.DataFrame(comparison_data)
            st.table(df_comparison)

            # Create a scatter plot to compare the two portfolios
            fig_comparison = px.scatter(
                x=[volatilidade*100, min_var_volatility*100],
                y=[retorno_esperado*100, min_var_return*100],
                text=['Retorno Alvo', 'Variância Mínima'],
                labels={'x': 'Volatilidade (%)', 'y': 'Retorno Esperado (%)'},
                title='Comparação dos Portfólios'
            )
            fig_comparison.update_traces(textposition='top center')
            st.plotly_chart(fig_comparison)


            st.subheader('Portfólio de Máximo Índice de Sharpe:')
            df_max_sharpe = pd.DataFrame(list(zip(tickers, max_sharpe_weights * 100)), columns=['Ativo', 'Peso'])
            df_max_sharpe['Peso (%)'] = df_max_sharpe['Peso'].apply(lambda x: f'{x:.2f}%')
            st.table(df_max_sharpe[['Ativo', 'Peso (%)']])

            st.write(f"Retorno Anual Esperado: {max_sharpe_return*100:.2f}%")
            st.write(f"Volatilidade Anual: {max_sharpe_volatility*100:.2f}%")
            st.write(f"Índice de Sharpe: {max_sharpe_ratio:.2f}")

            # Create a pie chart for the maximum Sharpe ratio portfolio
            fig_max_sharpe = px.pie(df_max_sharpe, values='Peso', names='Ativo', title='Alocação do Portfólio de Máximo Índice de Sharpe')
            st.plotly_chart(fig_max_sharpe)

            # Update the comparison table and scatter plot
            comparison_data = {
                'Portfólio': ['Retorno Alvo', 'Variância Mínima', 'Máximo Índice de Sharpe'],
                'Retorno Esperado': [f"{retorno_esperado*100:.2f}%", f"{min_var_return*100:.2f}%", f"{max_sharpe_return*100:.2f}%"],
                'Volatilidade': [f"{volatilidade*100:.2f}%", f"{min_var_volatility*100:.2f}%", f"{max_sharpe_volatility*100:.2f}%"],
                'Índice de Sharpe': [f"{(retorno_esperado - risk_free_rate) / volatilidade:.2f}", 
                                    f"{(min_var_return - risk_free_rate) / min_var_volatility:.2f}", 
                                    f"{max_sharpe_ratio:.2f}"]
            }
            df_comparison = pd.DataFrame(comparison_data)
            st.table(df_comparison)

            # Update the scatter plot to include the maximum Sharpe ratio portfolio
            fig_comparison = px.scatter(
                x=[volatilidade*100, min_var_volatility*100, max_sharpe_volatility*100],
                y=[retorno_esperado*100, min_var_return*100, max_sharpe_return*100],
                text=['Retorno Alvo', 'Variância Mínima', 'Máximo Índice de Sharpe'],
                labels={'x': 'Volatilidade (%)', 'y': 'Retorno Esperado (%)'},
                title='Comparação dos Portfólios'
            )
            fig_comparison.update_traces(textposition='top center')
            st.plotly_chart(fig_comparison)
            # Calculate cumulative returns for each portfolio
            cumulative_returns_target = calculate_cumulative_returns(df_precos, result.x)
            cumulative_returns_min_var = calculate_cumulative_returns(df_precos, min_var_weights)
            cumulative_returns_max_sharpe = calculate_cumulative_returns(df_precos, max_sharpe_weights)

            # Create cumulative return plot
            fig_cumulative_returns = plt.figure(figsize=(10, 6))
            plt.plot(cumulative_returns_target, label='Retorno Alvo')
            plt.plot(cumulative_returns_min_var, label='Variância Mínima')
            plt.plot(cumulative_returns_max_sharpe, label='Máximo Índice de Sharpe')
            plt.title('Retorno Cumulativo dos Portfólios')
            plt.xlabel('Data')
            plt.ylabel('Retorno Cumulativo')
            plt.legend()

            # Convert plot to image
            buf = BytesIO()
            fig_cumulative_returns.savefig(buf, format="png")
            buf.seek(0)
            st.image(buf)

            # Calculate daily returns for pyfolio
            daily_returns_target = (df_precos * result.x).sum(axis=1).pct_change().dropna()
            daily_returns_min_var = (df_precos * min_var_weights).sum(axis=1).pct_change().dropna()
            daily_returns_max_sharpe = (df_precos * max_sharpe_weights).sum(axis=1).pct_change().dropna()

        else:
            st.error("A otimização não foi bem-sucedida. O retorno alvo pode ser inatingível com as restrições atuais.")
            
    else:
        st.warning('Por favor, insira os tickers das ações.')

# Adiciona uma barra lateral com informações adicionais
st.sidebar.header('Sobre a Zomma Quant')
st.sidebar.write('A Zomma Quant fornece ferramentas de otimização de portfólio usando a teoria moderna de portfólio.')
st.sidebar.write('Este aplicativo usa o algoritmo SLSQP (Sequential Least Squares Programming) para otimização.')

# Adiciona informações sobre a nova atualização
st.sidebar.header('Atualização (14/10/2024)')
st.sidebar.write('Nesta versão, foram incluídas as seguintes melhorias:')
st.sidebar.markdown("""
- Adição do portfólio de Máximo Índice de Sharpe
- Implementação de restrição de alocação máxima por ativo
- Gráfico de retorno cumulativo para comparação dos portfólios
- Estatísticas detalhadas do portfólio usando Pyfolio
- Melhorias na visualização e comparação dos resultados
""")

st.sidebar.write('Estas atualizações fornecem uma análise mais abrangente e flexível para a otimização de portfólios.')