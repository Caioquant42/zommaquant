import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Import functions from your original script
from trend_matrix import (
    IBRX_50_list, IBOV_list, OPCOES_TOP_50,
    ydata, window_size_calculator, trend_matrix_df,
    calculate_trend_matrix
)

st.set_page_config(
    page_title="Trend Matrix - Zomma Quant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title('Trend Matrix - Zomma Quant')

def choose_stocks():
    option = st.selectbox(
        "Choose an option for stock selection:",
        ("Coloque seus próprios tickers", "Use IBRX 50", "Use IBOVESPA", "Use OPCOES TOP 50")
    )

    if option == "Coloque seus próprios tickers":
        tickers_input = st.text_input("Enter stock tickers separated by comma (e.g., PETR4,VALE3,BBAS3,B3SA3)")
        return [ticker.strip() for ticker in tickers_input.split(',')] if tickers_input else []
    elif option == "Use IBRX 50":
        return IBRX_50_list
    elif option == "Use IBOVESPA":
        return IBOV_list
    elif option == "Use OPCOES TOP 50":
        return OPCOES_TOP_50

def heat_map(matrix, title):
    current_datetime = datetime.datetime.now()
    current_datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')   
    matrix_numeric = matrix.apply(pd.to_numeric, errors='coerce')
    
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(matrix_numeric, annot=True, cmap='coolwarm_r', linewidths=0.9, fmt=".2f", annot_kws={"size": 9}, ax=ax)
    plt.title(f"{title} Trend Matrix {current_datetime_str}", fontsize=14)
    plt.xlabel("Time Frame", fontsize=14)
    plt.ylabel("Ativos", fontsize=14)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    
    return fig

def main():
    st.sidebar.header('Calculadora Trend Matrix')
    
    stocks = choose_stocks()
    
    start_date = st.sidebar.date_input(
        "Data Inicial",
        value=datetime.date.today() - datetime.timedelta(days=365)
    )
    
    end_date = st.sidebar.date_input(
        "Data Final",
        value=datetime.date.today()
    )
    
    timeframes = st.sidebar.multiselect(
        "Selecione timeframes para analisar",
        options=['1d', '1wk', '1mo'],
        default=['1d', '1wk']
    )
    
    if st.sidebar.button('Calcular Trend Matrix'):
        if stocks:
            with st.spinner('Calculando Trend Matrix...'):
                trend_matrix = calculate_trend_matrix(stocks, timeframes, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                
                st.subheader('Trend Matrix')
                st.dataframe(trend_matrix)
                
                st.subheader('Visualização Heatmap')
                fig = heat_map(trend_matrix, title='Zomma Quant')
                st.pyplot(fig)
                
                # Option to download the trend matrix as CSV
                csv = trend_matrix.to_csv(index=True)
                st.download_button(
                    label="Baixar Trend Matrix em CSV",
                    data=csv,
                    file_name="trend_matrix.csv",
                    mime="text/csv",
                )
        else:
            st.warning('Porfavor Selecione Ativos.')

    st.sidebar.markdown('---')
    st.sidebar.header('Sobre a Zomma Quant')
    st.sidebar.write('A Zomma Quant fornece ferramentas para análise de tendências e otimização de portfólio utilizando técnicas quantitativas modernas.')
    
    st.sidebar.header('Atualização (15/10/2024)')
    st.sidebar.write('Nesta versão, incluímos as seguintes melhorias:')
    st.sidebar.markdown("""
    - Seleção interativa de ações
    - Análise de múltiplos períodos
    - Visualização da matriz de tendência em mapa de calor
    - Opção para baixar os resultados em CSV
    """)

if __name__ == "__main__":
    main()