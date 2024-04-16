import streamlit as st
import numpy as np
from scipy.stats import norm
import yfinance as yf
import matplotlib.pyplot as plt

# Função para buscar o preço e a volatilidade do ativo no Yahoo Finance
def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1y")  # Dados históricos do último ano
    last_price = hist['Close'].iloc[-1]  # Último preço de fechamento
    daily_returns = hist['Close'].pct_change().dropna()  # Mudança percentual diária
    volatility = np.std(daily_returns) * np.sqrt(252)  # Volatilidade anualizada
    return last_price, volatility

# Configuração inicial da página
st.set_page_config(page_title="Calculadora de Opções", layout="wide", page_icon="📈")

# Interface do Streamlit
st.title('Calculadora de Opções')

# Sidebar para entrada de dados
col1, col2 = st.columns([1, 1])
with col1:
    symbol = st.text_input('Digite o símbolo do ativo (ex: AAPL):')
    strike_price = st.number_input('Preço de Exercício (K):', min_value=0.0, format="%.2f")
    expiry_time = st.number_input('Tempo até a Expiração (T) em anos:', min_value=0.0, step=0.01, format="%.2f")
    risk_free_rate = st.number_input('Taxa de Juros Sem Risco (r):', min_value=0.0, step=0.01, format="%.2f")
    option_type = st.selectbox('Tipo de Opção:', ['Call', 'Put'])

# Obter os dados do ativo do Yahoo Finance
if symbol:
    S, volatility = get_stock_data(symbol)
    with col2:
        st.write(f'Preço Atual do Ativo: ${S:.2f}')
        st.write(f'Volatilidade Anualizada: {volatility:.2%}')
        st.write('---')

        # Método de cálculo do preço da opção
        option_method = st.selectbox('Escolha o Método de Cálculo:', ['Black-Scholes', 'Monte Carlo'])

        # Botão para calcular o preço da opção
        if st.button('Calcular Preço da Opção'):
            if option_method == 'Black-Scholes':
                # Cálculo do preço da opção usando o modelo de Black-Scholes
                d1 = (np.log(S / strike_price) + (risk_free_rate + 0.5 * volatility ** 2) * expiry_time) / (volatility * np.sqrt(expiry_time))
                d2 = d1 - volatility * np.sqrt(expiry_time)
                if option_type == 'Call':
                    option_price = S * norm.cdf(d1) - strike_price * np.exp(-risk_free_rate * expiry_time) * norm.cdf(d2)
                else:
                    option_price = strike_price * np.exp(-risk_free_rate * expiry_time) * norm.cdf(-d2) - S * norm.cdf(-d1)
            elif option_method == 'Monte Carlo':
                # Cálculo do preço da opção usando o método de Monte Carlo
                dt = expiry_time / 365
                Z = np.random.normal(0, 1, 10000)
                ST = S * np.exp((risk_free_rate - 0.5 * volatility ** 2) * dt + volatility * np.sqrt(dt) * Z)
                payoff = np.maximum(ST - strike_price, 0)
                option_price = np.exp(-risk_free_rate * expiry_time) * np.mean(payoff)

            st.success(f'Preço da Opção Calculado: ${option_price:.2f}')

        # Exibir o delta, gamma e vega
        d1 = (np.log(S / strike_price) + (risk_free_rate + 0.5 * volatility ** 2) * expiry_time) / (volatility * np.sqrt(expiry_time))
        delta = norm.cdf(d1) if option_type == 'Call' else norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * volatility * np.sqrt(expiry_time))
        vega = S * norm.pdf(d1) * np.sqrt(expiry_time) / 100  # Vega em termos de 1% de mudança na volatilidade
        st.write(f'Delta: {delta:.4f}')
        st.write(f'Gamma: {gamma:.4f}')
        st.write(f'Vega: {vega:.4f}')

        # Plotar o gráfico de previsão do preço
        times = np.linspace(0, expiry_time, int(expiry_time * 365))
        prices = S * np.exp((risk_free_rate - 0.5 * volatility ** 2) * times + volatility * np.sqrt(times) * np.random.normal(size=len(times)))
        plt.figure(figsize=(10, 5))
        plt.plot(times, prices)
        plt.title('Previsão do Preço do Ativo')
        plt.xlabel('Tempo (anos)')
        plt.ylabel('Preço do Ativo')
        plt.grid(True)
        st.pyplot(plt)

