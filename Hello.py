import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Configuração inicial da página
st.set_page_config(page_title="Calculadora de Opções Avançada", layout="wide", page_icon="📈")

# Estilos personalizados
st.markdown("""
<style>
    .big-font {
        font-size:25px !important;
        font-weight: bold;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        padding: 10px 10px 10px 10px;
    }
    h1 {
        color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title('Calculadora de Opções Avançada')

# Sidebar - Parâmetros de Entrada
with st.sidebar:
    st.markdown("## Parâmetros de Entrada")
    mercado = st.selectbox("Escolha o Mercado:", ["Ações", "Moedas", "ETFs", "Offshore"])
    simbolo = st.text_input("Símbolo do Ativo:", placeholder="Ex: AAPL ou EURUSD")
    S = st.number_input("Preço do Ativo (S):", min_value=0.0, value=100.0, format="%.2f")
    K = st.number_input("Preço de Exercício (K):", min_value=0.0, value=100.0, format="%.2f")
    T = st.number_input("Tempo até a Expiração (T) em anos:", min_value=0.0, value=1.0, step=0.1, format="%.2f")
    r = st.number_input("Taxa de Juros Sem Risco (r):", min_value=0.0, value=0.05, step=0.01, format="%.2f")
    sigma = st.number_input("Volatilidade (σ):", min_value=0.0, value=0.20, step=0.01, format="%.2f")
    opcao_tipo = st.selectbox("Tipo de Opção:", ["Europeia", "Americana", "Asiática"])

# Tab layout
tab1, tab2, tab3 = st.tabs(["Cálculo de Opção", "Greeks", "Simulação de Preço"])

with tab1:
    st.header("Calculadora de Opção")
    if st.button('Calcular Preço da Opção'):
        preco_opcao = S * np.exp(-r * T)  # Example calculation
        st.success(f"Preço da Opção Calculado: ${preco_opcao:.2f}")

with tab2:
    st.header("Análise de Sensibilidade - Greeks")
    if st.button('Calcular Greeks'):
        delta = delta(S, K, T, r, sigma, opcao_tipo)
        gamma = gamma(S, K, T, r, sigma)
        vega = vega(S, K, T, r, sigma)
        st.write(f"Delta: {delta:.4f}, Gamma: {gamma:.4f}, Vega: {vega:.4f}")

with tab3:
    st.header("Simulação de Preço do Ativo")
    if st.button('Simular Preço do Ativo'):
        plot_interactive_price_simulation(S, sigma, T)

# Function definitions moved to appropriate locations in the script

def plot_interactive_price_simulation(S, sigma, T):
    prices = S * np.exp(np.cumsum(np.random.normal(0, sigma, int(T*365)))*np.sqrt(1/365))
    fig = go.Figure(data=[go.Line(x=list(range(int(T*365))), y=prices)])
    fig.update_layout(title='Simulação de Preço do Ativo', xaxis_title='Dias', yaxis_title='Preço do Ativo')
    st.plotly_chart(fig)
