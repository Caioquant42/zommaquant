import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from io import BytesIO

# Import functions from your MRL.py file
from MRL import (YData, last_incident, hist_interarrival, mean_residual_life, 
                 kaplan_meier_estimator_with_cumulative_hazard, plot_occurence, 
                 plot_interarrival, process_tickers, OPCOES_TOP_50)

# Theme configuration
st.set_page_config(
    page_title="Stock Analysis App - Zomma Quant",
    page_icon="📈",
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

st.title('Sobrevivência - Zomma Quant')

# Stock selection method
st.header('Stock Selection')
selection_method = st.radio(
    "Choose stock selection method:",
    ('Manual Input', 'Use OPCOES TOP 50')
)

if selection_method == 'Manual Input':
    tickers_input = st.text_input("Enter stock tickers separated by comma (e.g., PETR4,VALE3,BBAS3,B3SA3)")
    tickers = [ticker.strip() for ticker in tickers_input.split(',')] if tickers_input else []
else:
    st.write("Using OPCOES TOP 50 list")
    tickers = OPCOES_TOP_50

# Display selected tickers
st.write("Selected tickers:", ", ".join(tickers) if tickers else "No tickers selected")

# Threshold input (kept in sidebar)
st.sidebar.header('Analysis Settings')
threshold = st.sidebar.slider("Select threshold for negative returns", -0.15, 0.0, -0.05, 0.01)

if st.button('Analyze Stocks'):
    if tickers:
        with st.spinner('Analyzing stocks...'):
            results_df = process_tickers(tickers, threshold)
            
            st.subheader('Analysis Results')
            st.dataframe(results_df)

            # Option to download the results as CSV
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="stock_analysis_results.csv",
                mime="text/csv",
            )               
    else:
        st.warning('Please select or enter stock tickers.')

# New section for detailed analysis
st.header('Detailed Stock Analysis')
detailed_ticker = st.text_input("Enter a single ticker for detailed analysis")

col1, col2 = st.columns(2)

with col1:
    if st.button('Plot Occurrence'):
        if detailed_ticker:
            ydata = YData(detailed_ticker)
            stock_data = ydata.get_stock_data()
            fig_occurrence = plot_occurence(stock_data, threshold, f'Returns Below {threshold} for {detailed_ticker}')
            st.pyplot(fig_occurrence)
        else:
            st.warning('Please enter a ticker for detailed analysis.')

with col2:
    if st.button('Plot Interarrival'):
        if detailed_ticker:
            ydata = YData(detailed_ticker)
            stock_data = ydata.get_stock_data()
            fig_interarrival = plot_interarrival(stock_data, threshold, f'{detailed_ticker} Interarrival Time')
            st.pyplot(fig_interarrival)
        else:
            st.warning('Please enter a ticker for detailed analysis.')

st.sidebar.markdown('---')
st.sidebar.header('About Zomma Quant')
st.sidebar.write('Zomma Quant provides tools for stock trend analysis and portfolio optimization using modern quantitative techniques.')

# Add information about the new update
st.sidebar.header('Update (15/10/2024)')
st.sidebar.write('In this version, we included the following improvements:')
st.sidebar.markdown("""
- Interactive stock selection
- Option to use predefined OPCOES TOP 50 list
- Multiple period analysis
- Visualization of trend matrix in heatmap
- Option to download results in CSV
- Kaplan-Meier estimator visualization
- Cumulative returns plot for all analyzed stocks
- Detailed analysis for individual stocks
""")