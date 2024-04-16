import streamlit as st
import numpy as np
from scipy.stats import norm
import yfinance as yf
import matplotlib.pyplot as plt

# Configuração inicial da página
st.set_page_config(page_title="Calculadora de Opções", layout="wide", page_icon="📈")

# Função para obter os dados do ativo do Yahoo Finance
def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1y")  # Dados históricos do último ano
    last_price = hist['Close'].iloc[-1]  # Último preço de fechamento
    daily_returns = hist['Close'].pct_change().dropna()  # Mudança percentual diária
    volatility = np.std(daily_returns) * np.sqrt(252)  # Volatilidade anualizada
    return last_price, volatility

# Função para calcular o preço da opção usando o modelo Black-Scholes
def black_scholes(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Função para calcular o delta da opção
def delta(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

# Função para calcular o gamma da opção
def gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

# Função para calcular o vega da opção
def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) * 0.01  # Multiplicado por 0.01 para converter de % para pontos base

# Função para calcular o preço da opção usando o método de Monte Carlo
def monte_carlo_pricing(S, K, T, r, sigma, num_simulations, option_type):
    dt = T / 365
    Z = np.random.normal(0, 1, num_simulations)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    payoff = np.maximum(ST - K, 0)
    if option_type == 'call':
        option_price = np.exp(-r * T) * np.mean(payoff)
    else:
        option_price = np.exp(-r * T) * np.mean(payoff)
    return option_price

# Interface do Streamlit
st.title('Calculadora de Opções')

# Sidebar para entrada de dados
st.sidebar.title('Parâmetros da Opção')
symbol = st.sidebar.text_input('Digite o símbolo do ativo (ex: AAPL):')
strike_price = st.sidebar.number_input('Preço de Exercício (K):', min_value=0.0, format="%.2f")
expiry_time = st.sidebar.number_input('Tempo até a Expiração (T) em anos:', min_value=0.0, step=0.01, format="%.2f")
risk_free_rate = st.sidebar.number_input('Taxa de Juros Sem Risco (r):', min_value=0.0, step=0.01, format="%.2f")
option_type = st.sidebar.selectbox('Tipo de Opção:', ['Call', 'Put'])

# Obter os dados do ativo do Yahoo Finance
if symbol:
    S, volatility = get_stock_data(symbol)
    st.write(f'Preço Atual do Ativo: ${S:.2f}')
    st.write(f'Volatilidade Anualizada: {volatility:.2%}')

    # Método de cálculo do preço da opção
    option_method = st.selectbox('Escolha o Método de Cálculo:', ['Black-Scholes', 'Monte Carlo'])

    # Botão para calcular o preço da opção
    if st.button('Calcular Preço da Opção'):
        if option_method == 'Black-Scholes':
            option_price = black_scholes(S, strike_price, expiry_time, risk_free_rate, volatility, option_type.lower())
        elif option_method == 'Monte Carlo':
            option_price = monte_carlo_pricing(S, strike_price, expiry_time, risk_free_rate, volatility, 10000, option_type.lower())
        st.success(f'Preço da Opção Calculado: ${option_price:.2f}')

    # Calcular e exibir o delta, gamma e vega
    st.write('### Análise de Sensibilidade - Greeks')
    st.write(f'Delta: {delta(S, strike_price, expiry_time, risk_free_rate, volatility, option_type.lower()):.4f}')
    st.write(f'Gamma: {gamma(S, strike_price, expiry_time, risk_free_rate, volatility):.4f}')
    st.write(f'Vega: {vega(S, strike_price, expiry_time, risk_free_rate, volatility):.4f}')

    # Plotar o gráfico de previsão do preço
    st.write('### Gráfico de Previsão do Preço do Ativo')
    times = np.linspace(0, expiry_time, int(expiry_time * 365))
    prices = S * np.exp((risk_free_rate - 0.5 * volatility ** 2) * times + volatility * np.sqrt(times) * np.random.normal(size=len(times)))
    plt.figure(figsize=(10, 5))
    plt.plot(times, prices)
    plt.title('Previsão do Preço do Ativo')
    plt.xlabel('Tempo (anos)')
    plt.ylabel('Preço do Ativo')
    plt.grid(True)
    st.pyplot(plt)
