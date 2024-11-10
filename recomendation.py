import pandas as pd
import numpy as np
import yfinance as yf

TICKERS_DICT = {
    'ITAG': ['TTEN3', 'ABCB4', 'AERI3', 'AALR3', 'ALLD3', 'ALOS3', 'AVLL3', 'ALUP11', 
             'AMBP3', 'ANIM3', 'ARML3', 'ASAI3', 'AURE3', 'AZUL4', 'AZZA3', 'B3SA3', 'BAHI3', 
             'BMGB4', 'BPAN4', 'BGIP3', 'BGIP4', 'BEES3', 'BEES4', 'BRSR3', 'BRSR6', 'BBSE3', 'BMOB3', 
             'BIOM3', 'BLAU3', 'SOJA3', 'BBDC3', 'BBDC4', 'BRAP3', 'BRAP4', 'BBAS3', 'AGRO3', 'BRKM3', 'BRKM5', 
             'BRAV3', 'BRFS3', 'BRIT3', 'BPAC11', 'CXSE3', 'CAML3', 'CRFB3', 'BHIA3', 'CBAV3', 'CCRO3',
               'CEAB3', 'CEDO4', 'CLSC3', 'CLSC4', 'CLSA3', 'COGN3', 'CSMG3', 'CPLE3', 'CPLE5', 'CPLE6', 
               'CSAN3', 'CPFE3', 'CSED3', 'CMIN3', 'CSUD3', 'CURY3', 'CVCB3', 'CYRE3', 'DMVF3', 'DASA3',
                 'DESK3', 'DXCO3', 'PNVL3', 'DIRR3', 'DOTZ3', 'ECOR3', 'ELMD3', 'EMBR3', 'ENGI11', 'ENEV3',
                   'EGIE3', 'ENJU3', 'EQTL3', 'ESPA3', 'ALPK3', 'ETER3', 'EVEN3', 'EZTC3', 'FHER3', 'FLRY3',
                     'GFSA3', 'GEPA4', 'GGBR3', 'GGBR4', 'GOAU3', 'GOAU4', 'NINJ3', 'GGPS3', 'CGRA3', 'CGRA4',
                       'GRND3', 'GMAT3', 'NTCO3', 'SBFG3', 'GUAR3', 'HAPV3', 'HBRE3', 'HBOR3', 'HBSA3', 'HYPE3', 
                       'IGTI11', 'MEAL3', 'INEP3', 'INEP4', 'INTB3', 'MYPK3', 'RANI3', 'IRBR3', 'ITSA4', 'ITUB4', 
                       'JALL3', 'JBSS3', 'JHSF3', 'JSLG3', 'KEPL3', 'KLBN11', 'LAVV3', 'RENT3', 'LOGG3', 'LOGN3', 
                       'AMAR3', 'LREN3', 'LPSB3', 'LUPA3', 'LWSA3', 'MDIA3', 'MGLU3', 'POMO3', 'POMO4', 'MRFG3', 'MATD3', 
                       'CASH3', 'MELK3', 'LEVE3', 'FRIO3', 'MILS3', 'BEEF3', 'MTRE3', 'MBLY3', 'MDNE3', 'MOVI3', 'MRVE3', 
                       'MLAS3', 'MULT3', 'NEOE3', 'NGRD3', 'OPCT3', 'ODPV3', 'ONCO3', 'ORVR3', 'OFSA3', 'PCAR3', 'PDTC3',
                         'PGMN3', 'PETR3', 'PETR4', 'RECV3', 'PRIO3', 'PTNT4', 'PETZ3', 'PINE3', 'PINE4', 'PLPL3', 'PSSA3', 
                         'PTBL3', 'POSI3', 'PRNR3', 'PFRM3', 'QUAL3', 'LJQQ3', 'RADL3', 'RAIZ4', 'RAPT4', 'RDOR3', 'RDNI3', 
                         'ROMI3', 'RAIL3', 'SBSP3', 'SAPR11', 'SANB11', 'STBP3', 'SCAR3', 'SMTO3', 'SEER3', 'SRNA3', 'SIMH3', 
                         'SLCE3', 'SMFT3', 'SUZB3', 'SYNE3', 'TAEE11', 'TASA3', 'TASA4', 'TRAD3', 'TECN3', 'TCSA3', 'TGMA3', 
                         'TEND3', 'LAND3', 'TIMS3', 'SHOW3', 'TOTS3', 'TFCO4', 'TRIS3', 'TPIS3', 'TUPY3', 'UGPA3', 
    'UCAS3', 'FIQE3', 'VALE3', 'VLID3', 'VAMO3', 'VSTE3', 'VBBR3', 'VITT3', 'VIVA3', 'VVEO3', 'VIVR3', 'VULC3', 
    'LVTC3', 'WEGE3', 'PORT3', 'WIZC3', 'YDUQ3', 'ZAMP3'],
    'IBEE': ['BBSE3', 'BBAS3', 'CXSE3', 'CMIG4', 'PETR3', 'PETR4'],
    'IBEP': ['ALOS3', 'ALPA4', 'ABEV3', 'ASAI3', 'AURE3', 'AZUL4', 'AZZA3', 'B3SA3',
             'BBDC3', 'BBDC4', 'BRAP4', 'BRKM5', 'BRAV3', 'BRFS3', 'BPAC11', 'CRFB3',
             'CCRO3', 'COGN3', 'CPLE6', 'CSAN3', 'CPFE3', 'CMIN3', 'CVCB3', 'CYRE3',
             'ELET3', 'ELET6', 'EMBR3', 'ENGI11', 'ENEV3', 'EGIE3', 'EQTL3', 'EZTC3',
             'FLRY3', 'GGBR4', 'GOAU4', 'NTCO3', 'HAPV3', 'HYPE3', 'IGTI11', 'IRBR3',
             'ITSA4', 'ITUB4', 'JBSS3', 'KLBN11', 'RENT3', 'LREN3', 'LWSA3', 'MGLU3',
             'MRFG3', 'BEEF3', 'MRVE3', 'MULT3', 'PCAR3', 'RECV3', 'PRIO3', 'PETZ3',
             'RADL3', 'RAIZ4', 'RDOR3', 'RAIL3', 'SBSP3', 'SANB11', 'STBP3', 'SMTO3',
             'CSNA3', 'SLCE3', 'SUZB3', 'TAEE11', 'VIVT3', 'TIMS3', 'TOTS3', 'TRPL4',
             'UGPA3', 'USIM5', 'VALE3', 'VAMO3', 'VBBR3', 'VIVA3', 'WEGE3', 'YDUQ3'],

    'ICON': [
        'TTEN3', 'ALPA4', 'ABEV3', 'ANIM3', 'ASAI3', 'AZZA3', 'BLAU3', 'SOJA3',
        'AGRO3', 'BRFS3', 'CAML3', 'CRFB3', 'BHIA3', 'CEAB3', 'COGN3', 'CURY3',
        'CVCB3', 'CYRE3', 'DASA3', 'PNVL3', 'DIRR3', 'EVEN3', 'EZTC3', 'FLRY3',
        'GFSA3', 'GRND3', 'GMAT3', 'NTCO3', 'SBFG3', 'GUAR3', 'HAPV3', 'HYPE3',
        'MYPK3', 'JALL3', 'JBSS3', 'JHSF3', 'LAVV3', 'RENT3', 'LREN3', 'MDIA3',
        'MGLU3', 'MRFG3', 'MATD3', 'LEVE3', 'BEEF3', 'MTRE3', 'MDNE3', 'MOVI3',
        'MRVE3', 'ODPV3', 'ONCO3', 'PCAR3', 'PGMN3', 'PETZ3', 'PLPL3', 'QUAL3',
        'LJQQ3', 'RADL3', 'RDOR3', 'SMTO3', 'SEER3', 'SLCE3', 'SMFT3', 'TEND3',
        'TRIS3', 'VAMO3', 'VIVA3', 'VVEO3', 'VULC3', 'YDUQ3', 'ZAMP3'
    ],
    'IBLV': [
        'ALOS3', 'ABEV3', 'AURE3', 'BBSE3', 'BRAP4', 'BBAS3', 'CXSE3', 'CCRO3',
        'CPLE6', 'CPFE3', 'ENGI11', 'ENEV3', 'EGIE3', 'EQTL3', 'FLRY3', 'GOAU4',
        'IGTI11', 'ITSA4', 'ITUB4', 'KLBN11', 'RADL3', 'SANB11', 'SLCE3', 'TAEE11',
        'VIVT3', 'TIMS3', 'TRPL4', 'VALE3'
    ],
    'IBHB': [
        'ALOS3', 'ALPA4', 'AZUL4', 'B3SA3', 'BRFS3', 'BPAC11', 'COGN3', 'CSAN3',
        'CVCB3', 'CYRE3', 'EZTC3', 'NTCO3', 'HAPV3', 'IGTI11', 'IRBR3', 'RENT3',
        'LREN3', 'LWSA3', 'MGLU3', 'MRVE3', 'PCAR3', 'PETR4', 'CSNA3', 'UGPA3',
        'USIM5', 'VBBR3', 'VIVA3', 'YDUQ3'
    ],

    'BDRX': [
        'ABBV34', 'ADBE34', 'A1AP34', 'A1MD34', 'AIRB34', 'A1LB34', 'BABA34',
        'GOGL34', 'GOGL35', 'AMZO34', 'AALL34', 'AXPB34', 'T1OW34', 'AAPL34',
        'A1MT34', 'ARMT34', 'A1NE34', 'ASML34', 'A1ZN34', 'ATTB34', 'BIDU34',
        'B1SA34', 'C2OL34', 'BOAC34', 'BERK34', 'B2YN34', 'B1IL34', 'B2HI34',
        'B1NT34', 'BLAK34', 'S2QU34', 'BKNG34', 'B1PP34', 'B1TI34', 'AVGO34',
        'CATP34', 'CHCM34', 'CHVX34', 'CSCO34', 'CTGP34', 'COCA34', 'C2OI34',
        'COLG34', 'CMCS34', 'COPH34', 'COWC34', 'C2RW34', 'CSXC34', 'DHER34',
        'DEEC34', 'D1EL34', 'DEOP34', 'DGCO34', 'E1CO34', 'EQIX34', 'E1QN34',
        'EXXO34', 'FSLR34', 'FDMO34', 'F2NV34', 'GEOO34', 'GMCO34', 'GSGI34',
        'HOME34', 'HPQB34', 'H1SB34', 'ITLC34', 'JDCO34', 'JNJB34', 'JPMC34',
        'K2CG34', 'KHCB34', 'LILY34', 'L1YG34', 'L1MN34', 'MSCD34', 'MCDC34',
        'M2PW34', 'MELI34', 'MRCK34', 'M1TA34', 'MUTC34', 'MSFT34', 'M2ST34',
        'M1RN34', 'M1NS34', 'MSBR34', 'N1DA34', 'NFLX34', 'E1DU34', 'N1EM34',
        'NEXT34', 'NIKE34', 'N1VO34', 'ROXO34', 'NVDC34', 'OXYP34', 'ORCL34',
        'PAGS34', 'P2LT34', 'P2AN34', 'PYPL34', 'P1DD34', 'PEPB34', 'PFIZ34',
        'PGCO34', 'P1LD34', 'QCOM34', 'R1IN34', 'RIOT34', 'R2BL34', 'SSFO34',
        'BCSA34', 'SCHW34', 'S2EA34', 'N1OW34', 'S2HO34', 'S1BS34', 'S2GM34',
        'SIMN34', 'S1LG34', 'S2NW34', 'SNEC34', 'S1PO34', 'S2TA34', 'SBUB34',
        'S2UI34', 'TSMC34', 'T1TW34', 'T1AL34', 'T2DH34', 'TLNC34', 'TSLA34',
        'TEXA34', 'TMOS34', 'TMCO34', 'T2TD34', 'RIGG34', 'U1BE34', 'ULEV34',
        'U1RI34', 'UNHH34', 'U2ST34', 'U2PS34', 'VERZ34', 'VISA34', 'V1OD34',
        'WALM34', 'WGBA34', 'DISB34', 'W1BD34', 'WFCO34', 'Z1TS34', 'Z1OM34'
    ],
    'IVBX': [
        'ALOS3', 'ASAI3', 'AZUL4', 'AZZA3', 'BBSE3', 'BRKM5', 'BRAV3', 'BRFS3',
        'CRFB3', 'CCRO3', 'CMIG4', 'COGN3', 'CPLE6', 'CSAN3', 'CYRE3', 'ELET3',
        'EMBR3', 'ENGI11', 'ENEV3', 'EQTL3', 'GGBR4', 'GOAU4', 'NTCO3', 'HAPV3',
        'HYPE3', 'IGTI11', 'JBSS3', 'KLBN11', 'LREN3', 'MGLU3', 'MRFG3', 'BEEF3',
        'MRVE3', 'MULT3', 'RADL3', 'RDOR3', 'RAIL3', 'SBSP3', 'CSNA3', 'SUZB3',
        'VIVT3', 'TIMS3', 'TOTS3', 'TRPL4', 'UGPA3', 'USIM5', 'VAMO3', 'VBBR3',
        'VIVA3', 'YDUQ3'
    ],

    'SMLL': [
        'TTEN3', 'ABCB4', 'AERI3', 'ALOS3', 'ALPA4', 'ALUP11', 'AMBP3', 'ANIM3',
        'ARML3', 'ASAI3', 'AURE3', 'AZEV4', 'AZUL4', 'AZZA3', 'BPAN4', 'BRSR6',
        'BMOB3', 'BLAU3', 'SOJA3', 'BRAP4', 'AGRO3', 'BRKM5', 'BRAV3', 'CAML3',
        'BHIA3', 'CBAV3', 'CEAB3', 'CLSA3', 'COGN3', 'CSMG3', 'CURY3', 'CVCB3',
        'CYRE3', 'DASA3', 'DXCO3', 'PNVL3', 'DIRR3', 'ECOR3', 'EVEN3', 'EZTC3',
        'FESA4', 'FLRY3', 'FRAS3', 'GFSA3', 'GOAU4', 'GGPS3', 'GRND3', 'SBFG3',
        'GUAR3', 'HBSA3', 'IGTI11', 'INTB3', 'MYPK3', 'RANI3', 'IRBR3', 'JALL3',
        'JHSF3', 'JSLG3', 'KEPL3', 'LAVV3', 'LOGG3', 'LWSA3', 'MDIA3', 'MGLU3',
        'POMO4', 'MRFG3', 'CASH3', 'LEVE3', 'MILS3', 'BEEF3', 'MTRE3', 'MDNE3',
        'MOVI3', 'MRVE3', 'MLAS3', 'OPCT3', 'ODPV3', 'ONCO3', 'ORVR3', 'PCAR3',
        'PGMN3', 'RECV3', 'PETZ3', 'PLPL3', 'PTBL3', 'POSI3', 'QUAL3', 'LJQQ3',
        'RAPT4', 'RCSL3', 'ROMI3', 'SAPR11', 'STBP3', 'SMTO3', 'SEER3', 'SRNA3',
        'SIMH3', 'SLCE3', 'SMFT3', 'TAEE11', 'TASA4', 'TGMA3', 'TEND3', 'TRIS3',
        'TUPY3', 'UNIP6', 'USIM3', 'USIM5', 'VLID3', 'VAMO3', 'VIVA3', 'VVEO3',
        'VULC3', 'PORT3', 'WIZC3', 'YDUQ3', 'ZAMP3'
    ],
    'IDIV': [
        'ABCB4', 'AURE3', 'BRSR6', 'BBSE3', 'BBDC3', 'BBDC4', 'BRAP4', 'BBAS3',
        'AGRO3', 'CXSE3', 'CMIG3', 'CMIG4', 'CSMG3', 'CPFE3', 'CMIN3', 'CURY3',
        'DIRR3', 'EGIE3', 'FESA4', 'FLRY3', 'GGBR4', 'GOAU4', 'RANI3', 'ITSA4',
        'JBSS3', 'JHSF3', 'KEPL3', 'KLBN11', 'LAVV3', 'POMO4', 'LEVE3', 'MTRE3',
        'PETR3', 'PETR4', 'SAPR4', 'SANB11', 'STBP3', 'CSNA3', 'TAEE11', 'TASA4',
        'TGMA3', 'VIVT3', 'TIMS3', 'TRPL4', 'UNIP6', 'USIM5', 'VALE3', 'WIZC3'
    ],
    'IBSD': [
        'BBSE3', 'BBDC4', 'BRAP4', 'BBAS3', 'CXSE3', 'CMIG4', 'CPLE6', 'CPFE3',
        'CMIN3', 'EGIE3', 'GGBR4', 'GOAU4', 'ITSA4', 'PETR4', 'SANB11', 'CSNA3',
        'TAEE11', 'VIVT3', 'TRPL4', 'VALE3', 'VBBR3'
    ],
    'AGFS': [
        'TTEN3', 'ABEV3', 'ARML3', 'ASAI3', 'SOJA3', 'AGRO3', 'BRFS3', 'CAML3',
        'CRFB3', 'CSAN3', 'DXCO3', 'GMAT3', 'HBSA3', 'RANI3', 'JALL3', 'JBSS3',
        'JSLG3', 'KEPL3', 'KLBN11', 'MDIA3', 'MRFG3', 'BEEF3', 'PCAR3', 'RAIZ4',
        'RAPT4', 'RCSL3', 'RAIL3', 'SMTO3', 'SLCE3', 'SUZB3', 'TUPY3', 'VAMO3'
    ],
    'IBOV': [
        'ALOS3', 'ALPA4', 'ABEV3', 'ASAI3', 'AURE3', 'AZUL4', 'AZZA3', 'B3SA3',
        'BBSE3', 'BBDC3', 'BBDC4', 'BRAP4', 'BBAS3', 'BRKM5', 'BRAV3', 'BRFS3',
        'BPAC11', 'CXSE3', 'CRFB3', 'CCRO3', 'CMIG4', 'COGN3', 'CPLE6', 'CSAN3',
        'CPFE3', 'CMIN3', 'CVCB3', 'CYRE3', 'ELET3', 'ELET6', 'EMBR3', 'ENGI11',
        'ENEV3', 'EGIE3', 'EQTL3', 'EZTC3', 'FLRY3', 'GGBR4', 'GOAU4', 'NTCO3',
        'HAPV3', 'HYPE3', 'IGTI11', 'IRBR3', 'ITSA4', 'ITUB4', 'JBSS3', 'KLBN11',
        'RENT3', 'LREN3', 'LWSA3', 'MGLU3', 'MRFG3', 'BEEF3', 'MRVE3', 'MULT3',
        'PCAR3', 'PETR3', 'PETR4', 'RECV3', 'PRIO3', 'PETZ3', 'RADL3', 'RAIZ4',
        'RDOR3', 'RAIL3', 'SBSP3', 'SANB11', 'STBP3', 'SMTO3', 'CSNA3', 'SLCE3',
        'SUZB3', 'TAEE11', 'VIVT3', 'TIMS3', 'TOTS3', 'TRPL4', 'UGPA3', 'USIM5',
        'VALE3', 'VAMO3', 'VBBR3', 'VIVA3', 'WEGE3', 'YDUQ3'
    ]
}
class YData:
    def __init__(self, ticker_symbol, interval='1d', period='max', world=False, start_date=None, end_date=None):
        self.ticker_symbol = ticker_symbol
        self.interval = interval
        self.period = period
        self.world = world
        self.start_date = start_date
        self.end_date = end_date
        self.ticker = self._add_sa_to_tickers(self.ticker_symbol) # Initialize ticker here
        self.stock_data = yf.Ticker(self.ticker) # Initialize yf.Ticker once


    def _add_sa_to_tickers(self, tickers):
        return f"{tickers}.SA" if not self.world else tickers

    def get_stock_data(self):

        # Fetch historical data based on date range or period
        if self.start_date and self.end_date:
            historical_data = self.stock_data.history(start=self.start_date, end=self.end_date, interval=self.interval)
        else:
            historical_data = self.stock_data.history(period=self.period, interval=self.interval)

        if historical_data.empty:
            print(f"No historical data found for {self.ticker}. Check the ticker and date range.")
            return pd.DataFrame() # Return empty dataframe to avoid errors

        # Rename columns for clarity
        rename_cols = {
            'Open': 'Abertura', 
            'High': 'Máxima', 
            'Low': 'Mínima', 
            'Close': 'Fechamento', 
            'Volume': 'Volume', 
            'Dividends': 'Dividendos', 
            'Stock Splits': 'Desdobramentos'
        }
        historical_data.rename(columns=rename_cols, inplace=True)

        # Calculate returns
        historical_data['simple_return'] = historical_data['Fechamento'].pct_change()
        historical_data['log_return'] = np.log1p(historical_data['simple_return'])
        historical_data.dropna(subset=['simple_return', 'log_return'], inplace=True)

        return historical_data

    def get_fundamental_data(self, info_keys=None):
       """Retrieves fundamental data.

       Args:
           info_keys: A list of keys to retrieve from the info dictionary.
                      If None, retrieves all available keys.

       Returns:
           A dictionary containing the requested fundamental data or
           None if no data is found or if there's an error.
       """
       try:
           info = self.stock_data.info
           if info_keys is None:
               return info
           else:
               return {key: info.get(key) for key in info_keys if key in info}
       except Exception as e:
           print(f"Error retrieving fundamental data for {self.ticker}: {e}")
           return None
       
    def get_fundamental_data_summary(self):
        """
        Retrieves and summarizes fundamental data in a DataFrame.

        Returns:
            A pandas DataFrame summarizing the fundamental data, or None if an error occurs.
        """
        try:
            info = self.stock_data.info
            
            # Convert the info dictionary to a DataFrame
            df_info = pd.DataFrame.from_dict(info, orient='index', columns=['Value'])
            df_info.index.name = 'Metric'
            return df_info

        except Exception as e:
            print(f"Error retrieving fundamental data summary for {self.ticker}: {e}")
            return None
       
    def save_fundamental_data_summary_to_txt(self, filename="fundamental_data_summary.txt"):
        """
        Retrieves fundamental data summary and saves it to a text file.

        Args:
            filename: The name of the file to save the data to.
        """

        try:
            df_summary = self.get_fundamental_data_summary()

            if df_summary is not None:
                with open(filename, 'w', encoding='utf-8') as f:  # Abre o arquivo em modo de escrita ('w') com codificação UTF-8
                    f.write(df_summary.to_string())  # Escreve o DataFrame no arquivo
                print(f"Fundamental data summary saved to {filename}")
            
        except Exception as e:
            print(f"Error saving fundamental data summary to {filename}: {e}")

    ''' Example of usage
        ydata = YData("PETR4")
        ydata.save_fundamental_data_summary_to_txt() # Salva no arquivo padrão "fundamental_data_summary.txt"

        # Para salvar em um arquivo diferente:
        ydata.save_fundamental_data_summary_to_txt(filename="petr4_summary.txt")"'''
def get_dividend_information(tickers):
    """Retrieves dividend information for a list of tickers.

    Args:
        tickers: A list of ticker symbols.

    Returns:
        A pandas DataFrame containing dividend information with 'Ticker' as the first column,
         or None if an error occurs.
    """
    try:
        data = []
        for ticker in tickers:
            ydata = YData(ticker)
            info = ydata.get_fundamental_data(info_keys=[
                'currentPrice', 'dividendRate', 'dividendYield',
                'trailingAnnualDividendRate', 'trailingAnnualDividendYield',
                'fiveYearAvgDividendYield', 'lastDividendValue', 'lastDividendDate'
            ])
            if info:  # Check if info is not None or empty
                info['Ticker'] = ticker  # Add the ticker to the data
                
                if 'lastDividendDate' in info and info['lastDividendDate']:
                    info['lastDividendDate'] = pd.to_datetime(info['lastDividendDate'], unit='s').strftime('%Y-%m-%d')
                
                data.append(info)

        if data:  # Check if any data was retrieved
            df = pd.DataFrame(data)
            
            # Reorder columns to put 'Ticker' first
            if 'Ticker' in df.columns:
                cols = ['Ticker'] + [col for col in df.columns if col != 'Ticker']
                df = df[cols]
            return df
        else:
            return None

    except Exception as e:
        print(f"Error retrieving dividend information: {e}")
        return None
    

def get_analyst_information(tickers):
    """Retrieves analyst information for a list of tickers.

    Args:
        tickers: A list of ticker symbols.

    Returns:
        A pandas DataFrame containing analyst information with 'Ticker' as the first column,
        and additional columns for percentage distances, or None if an error occurs.
    """
    try:
        data = []
        for ticker in tickers:
            ydata = YData(ticker)
            info = ydata.get_fundamental_data(info_keys=[
                'currentPrice', 'recommendationKey', 'numberOfAnalystOpinions',
                'targetMedianPrice', 'targetMeanPrice', 'targetLowPrice', 'targetHighPrice'
            ])
            if info:  # Check if info is not None or empty
                info['Ticker'] = ticker  # Add the ticker to the data

                # Calculate percentage distances
                current_price = info.get('currentPrice', None)
                if current_price is not None and current_price != 0:
                    info['% Distance to Median'] = ((info.get('targetMedianPrice', 0) - current_price) / current_price) * 100
                    info['% Distance to Low'] = ((info.get('targetLowPrice', 0) - current_price) / current_price) * 100
                    info['% Distance to High'] = ((info.get('targetHighPrice', 0) - current_price) / current_price) * 100
                else:
                    info['% Distance to Median'] = None
                    info['% Distance to Low'] = None
                    info['% Distance to High'] = None

                data.append(info)

        if data:  # Check if any data was retrieved
            df = pd.DataFrame(data)
            
            # Reorder columns to put 'Ticker' first
            if 'Ticker' in df.columns:
                cols = ['Ticker'] + [col for col in df.columns if col != 'Ticker']
                df = df[cols]
            return df
        else:
            return None

    except Exception as e:
        print(f"Error retrieving analyst information: {e}")
        return None

