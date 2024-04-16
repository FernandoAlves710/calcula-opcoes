import streamlit as st
import numpy as np
import yfinance as yf
from scipy.stats import norm

# Configuração inicial da página
st.set_page_config(page_title="Calculadora de Opções Avançada", layout="wide", page_icon="📈")

import yfinance as yf

def get_stock_data(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        if stock.info['quoteType'] == 'ETF':
            hist = stock.history(period="1y")  # Dados históricos do último ano
            if hist.empty:
                st.error(f"Não foi possível obter dados para o ETF com o símbolo {ticker_symbol}. Por favor, tente outro símbolo.")
                return None, None
            last_price = hist['Close'].iloc[-1]  # Último preço de fechamento
            daily_returns = hist['Close'].pct_change().dropna()  # Mudança percentual diária
            volatilidade = np.std(daily_returns) * np.sqrt(252)  # Volatilidade anualizada
            return last_price, volatilidade
        else:
            data = stock.history(period="1y")
            if data.empty:
                st.error(f"Não foi possível obter dados para a ação com o símbolo {ticker_symbol}. Por favor, tente outro símbolo.")
                return None, None
            last_price = data['Close'].iloc[-1]
            daily_returns = data['Close'].pct_change().dropna()
            volatilidade = np.std(daily_returns) * np.sqrt(252)
            return last_price, volatilidade
    except KeyError:
        st.error(f"As informações necessárias para calcular a opção não estão disponíveis para o símbolo {ticker_symbol}. Por favor, tente outro símbolo.")
        return None, None
        
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def delta(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return np.exp(-r * T) * norm.cdf(d1)
    else:
        return np.exp(-r * T) * (norm.cdf(d1) - 1)

def gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return np.exp(-r * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-r * T) * np.sqrt(T) * norm.pdf(d1)

# Interface do usuário
st.title('Calculadora de Opções Avançada')
simbolo = st.text_input("Digite o símbolo do ativo (ex: AAPL ou um ETF como AGRI11):")

if simbolo:
    S, sigma = get_stock_data(simbolo)
    if S is not None and sigma is not None:
        st.write(f"Preço Atual do Ativo: {S:.2f}")
        st.write(f"Volatilidade Anualizada: {sigma:.2%}")

        K = st.number_input("Preço de Exercício (K):", min_value=0.0, value=S, format="%.2f")
        T = st.number_input("Tempo até a Expiração (T) em anos:", min_value=0.0, value=1.0, step=0.1, format="%.2f")
        r = st.number_input("Taxa de Juros Sem Risco (r):", min_value=0.0, value=0.05, step=0.01, format="%.2f")
        opcao_tipo = st.selectbox("Tipo de Opção:", ["Europeia", "Americana", "Asiática"])

        if st.button('Calcular Preço da Opção'):
            preco_opcao = black_scholes(S, K, T, r, sigma, opcao_tipo)
            st.success(f"Preço da Opção Calculada: ${preco_opcao:.2f}")

        # Cálculo das Gregas
        st.header("Análise de Sensibilidade - Gregas")
        with st.expander("Mostrar Gregas"):
            st.write("### Delta:")
            delta_valor = delta(S, K, T, r, sigma, option_type=opcao_tipo)
            st.write(f"Delta: {delta_valor:.4f}")

            st.write("### Gamma:")
            gamma_valor = gamma(S, K, T, r, sigma)
            st.write(f"Gamma: {gamma_valor:.4f}")

            st.write("### Vega:")
            vega_valor = vega(S, K, T, r, sigma)
            st.write(f"Vega: {vega_valor:.4f}")

