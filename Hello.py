import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

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
st.title('Calculadora de Opções Avançada', anchor=None)

# Sidebar - Parâmetros de Entrada
with st.sidebar:
    st.markdown("## Parâmetros de Entrada")
    mercado = st.selectbox("Escolha o Mercado:", ["Ações", "Moedas", "ETFs", "Offshore"])
    simbolo = st.text_input("Símbolo do Ativo:")
    S = st.number_input("Preço do Ativo (S):", min_value=0.0, value=100.0, format="%.2f")
    K = st.number_input("Preço de Exercício (K):", min_value=0.0, value=100.0, format="%.2f")
    T = st.number_input("Tempo até a Expiração (T) em anos:", min_value=0.0, value=1.0, step=0.1, format="%.2f")
    r = st.number_input("Taxa de Juros Sem Risco (r):", min_value=0.0, value=0.05, step=0.01, format="%.2f")
    sigma = st.number_input("Volatilidade (σ):", min_value=0.0, value=0.20, step=0.01, format="%.2f")
    opcao_tipo = st.selectbox("Tipo de Opção:", ["Europeia", "Americana", "Asiática"])

# Main Area - Displaying Results and Graphs
col1, col2 = st.columns(2)

with col1:
    if st.button('Calcular Preço da Opção'):
        with st.spinner('Calculando...'):
            # Placeholder for calculation logic
            preco_opcao = S * np.exp(-r * T)  # Example calculation
            st.success(f"Preço da Opção Calculado: ${preco_opcao:.2f}")

with col2:
    st.write("### Gráfico de Simulação do Preço do Ativo")
    fig, ax = plt.subplots()
    ax.plot(np.random.normal(S, sigma, 100), marker='', color='skyblue', linewidth=2)
    ax.set_title("Simulação de Preço do Ativo ao Longo do Tempo")
    ax.grid(True)
    st.pyplot(fig)

