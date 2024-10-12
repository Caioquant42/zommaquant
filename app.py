import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from datetime import datetime, timedelta
import plotly.express as px


st.set_page_config(page_title="Otimizador de Portfólio Zomma Quant", page_icon="", layout="wide")

# Importe a classe ydata e a função optimize_portfolio do seu arquivo target_vol.py
from target_vol import ydata, optimize_portfolio

st.title('Otimizador de Portfólio Zomma Quant')

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

# Slider para volatilidade alvo
volatilidade_alvo = st.slider("Selecione a volatilidade anual alvo", 0.0, 1.0, 0.25, 0.01)

# Slider para alocação máxima
alocacao_maxima = st.slider("Selecione a alocação máxima por ativo", 0.0, 1.0, 0.30, 0.01)

if st.button('Otimizar Portfólio'):
    if tickers_input:
        tickers = [ticker.strip() for ticker in tickers_input.split(',')]
        
        pesos, desempenho = optimize_portfolio(tickers, data_inicio.strftime("%Y-%m-%d"), 
                                               data_fim.strftime("%Y-%m-%d"), 
                                               volatilidade_alvo, alocacao_maxima)
        
        st.subheader('Pesos Otimizados do Portfólio:')
        df_pesos = pd.DataFrame(list(pesos.items()), columns=['Ativo', 'Peso'])
        
        # Cria uma cópia de df_pesos para exibição
        df_exibicao = df_pesos.copy()
        df_exibicao['Peso'] = df_exibicao['Peso'].apply(lambda x: f'{x:.2%}')
        st.table(df_exibicao)
        
        st.subheader('Desempenho do Portfólio:')
        st.write(f"Retorno Anual Esperado: {desempenho[0]:.2%}")
        st.write(f"Volatilidade Anual: {desempenho[1]:.2%}")
        st.write(f"Índice de Sharpe: {desempenho[2]:.2f}")
        
        # Cria um gráfico de pizza dos pesos do portfólio
        fig = px.pie(df_pesos, values='Peso', names='Ativo', title='Alocação do Portfólio')
        st.plotly_chart(fig)
    else:
        st.warning('Por favor, insira os tickers das ações.')

# Adiciona uma barra lateral com informações adicionais
st.sidebar.header('Sobre a Zomma Quant')
st.sidebar.write('A Zomma Quant fornece ferramentas de otimização de portfólio usando a teoria moderna de portfólio.')
st.sidebar.write('Este aplicativo usa a fronteira eficiente para otimizar portfólios com base em uma volatilidade alvo.')