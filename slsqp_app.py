import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px

# Import functions from return_target.py
from return_target import ydata, portfolio_volatility, portfolio_return, optimize_portfolio

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
alocacao_maxima = st.slider("Selecione a alocação máxima por ativo", 0.0, 1.0, 0.30, 0.01)

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

        # Definindo restrições adicionais
        constraints = [{'type': 'ineq', 'fun': lambda w: alocacao_maxima - w[i]} for i in range(len(tickers))]

        # Otimizando o portfólio
        result = optimize_portfolio(retornos_esperados.values, cov_matrix.values, retorno_alvo, constraints)

        if result.success:
            st.subheader('Pesos Otimizados do Portfólio:')
            df_pesos = pd.DataFrame(list(zip(tickers, result.x * 100)), columns=['Ativo', 'Peso (%)'])
            df_pesos['Peso (%)'] = df_pesos['Peso (%)'].apply(lambda x: f'{x:.2f}%')
            st.table(df_pesos)
            
            retorno_esperado = portfolio_return(result.x, retornos_esperados.values)
            volatilidade = portfolio_volatility(result.x, cov_matrix.values)
            
            st.subheader('Desempenho do Portfólio:')
            st.write(f"Retorno Anual Esperado: {retorno_esperado*100:.2f}%")
            st.write(f"Volatilidade Anual: {volatilidade*100:.2f}%")
            
            # Cria um gráfico de pizza dos pesos do portfólio
            fig = px.pie(df_pesos, values='Peso (%)', names='Ativo', title='Alocação do Portfólio')
            st.plotly_chart(fig)
        else:
            st.error("A otimização não foi bem-sucedida. O retorno alvo pode ser inatingível com as restrições atuais.")
    else:
        st.warning('Por favor, insira os tickers das ações.')

# Adiciona uma barra lateral com informações adicionais
st.sidebar.header('Sobre a Zomma Quant')
st.sidebar.write('A Zomma Quant fornece ferramentas de otimização de portfólio usando a teoria moderna de portfólio.')
st.sidebar.write('Este aplicativo usa o algoritmo SLSQP (Sequential Least Squares Programming) para otimização.')