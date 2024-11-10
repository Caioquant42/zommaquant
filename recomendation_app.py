import streamlit as st
import pandas as pd
from recomendation import YData, get_analyst_information, TICKERS_DICT

# Configuração do tema do Streamlit
st.set_page_config(
    page_title="Aplicativo de Informações de Analistas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title('Visualizador de Informações de Analistas')

# Método de seleção de tickers
st.header('Seleção de Tickers')
selection_method = st.radio(
    "Escolha o método de seleção de tickers:",
    ('Entrada Manual', 'Usar Índice Predefinido')
)

if selection_method == 'Entrada Manual':
    tickers_input = st.text_input("Digite os tickers das ações separados por vírgula (ex.: PETR4,VALE3,BBAS3,B3SA3)")
    tickers = [ticker.strip() for ticker in tickers_input.split(',')] if tickers_input else []
else:
    index = st.selectbox("Selecione um índice:", list(TICKERS_DICT.keys()))
    tickers = TICKERS_DICT.get(index, [])

# Exibir tickers selecionados
st.write("Tickers selecionados:", ", ".join(tickers) if tickers else "Nenhum ticker selecionado")

# Botão para calcular informações de analistas
if st.button('Obter Informações de Analistas'):
    if tickers:
        with st.spinner('Recuperando informações de analistas...'):
            analyst_df = get_analyst_information(tickers)
            if analyst_df is not None:
                st.subheader('Informações de Analistas')
                st.dataframe(analyst_df)
                
                # Opção para baixar os resultados como CSV
                csv = analyst_df.to_csv(index=False)
                st.download_button(
                    label="Baixar Informações de Analistas como CSV",
                    data=csv,
                    file_name="informacoes_de_analistas.csv",
                    mime="text/csv",
                )
            else:
                st.warning('Nenhuma informação de analistas disponível para os tickers selecionados.')
    else:
        st.warning('Por favor, selecione ou insira os tickers das ações.')

# Seção de informações adicionais
st.sidebar.markdown('---')
st.sidebar.header('Sobre Este Aplicativo')
st.sidebar.write('Este aplicativo fornece informações de analistas para os tickers de ações selecionados usando dados do Yahoo Finance.')

# Atualizações ou informações adicionais
st.sidebar.header('Atualização (15/10/2024)')
st.sidebar.write('Nesta versão, incluímos as seguintes melhorias:')
st.sidebar.markdown("""
- Seleção interativa de tickers
- Opção de usar lista de índices predefinidos
- Baixar resultados em formato CSV
""")
