def calcular_preco_opcao_BS(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    preco_opcao = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return preco_opcao

# Função para calcular o preço da opção usando o método de Monte Carlo
def calcular_preco_opcao_MonteCarlo(S, K, T, r, sigma, num_simulacoes=10000):
    dt = T / 365
    Z = np.random.normal(0, 1, num_simulacoes)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    payoff = np.maximum(ST - K, 0)
    preco_opcao = np.exp(-r * T) * np.mean(payoff)
    return preco_opcao

# Função para calcular o preço da opção usando o método binomial
def calcular_preco_opcao_Binomial(S, K, T, r, sigma, n=1000):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    prices = np.zeros((n+1, n+1))
    prices[0, 0] = S
    for i in range(1, n+1):
        prices[i, 0] = prices[i-1, 0] * u
        for j in range(1, i+1):
            prices[i, j] = prices[i-1, j-1] * d
    option_values = np.maximum(prices - K, 0)
    for j in range(n-1, -1, -1):
        for i in range(j+1):
            option_values[i, j] = np.exp(-r * dt) * (p * option_values[i, j+1] + (1 - p) * option_values[i+1, j+1])
    return option_values[0, 0]

# Obtendo o mercado escolhido pelo usuário
print("Escolha o mercado que deseja analisar:")
print("1. Ações")
print("2. Moedas")
mercado_escolhido = int(input("Digite o número correspondente ao mercado desejado: "))

if mercado_escolhido == 1:
    mercado = "Ações"
elif mercado_escolhido == 2:
    mercado = "Moedas"

# Obtendo o símbolo do ativo ou paridade de moeda
simbolo = input("Digite o símbolo do ativo ou a paridade de moeda (ex: AAPL ou EURUSD): ")

# Obtendo os parâmetros da opção
S = float(input("Preço do Ativo (S): "))
K = float(input("Preço de Exercício (K): "))
T = float(input("Tempo até a Expiração (T) em anos: "))
r = float(input("Taxa de Juros Sem Risco (r): "))
sigma = float(input("Volatilidade (sigma): "))

# Escolha do método de solução
print("Escolha o método de solução:")
print("1. Black-Scholes")
print("2. Monte Carlo")
print("3. Binomial")

opcao_metodo = int(input("Digite o número correspondente ao método de solução desejado: "))

# Calcula o preço da opção com base no método selecionado
if opcao_metodo == 1:
    preco_opcao = calcular_preco_opcao_BS(S, K, T, r, sigma)
elif opcao_metodo == 2:
    preco_opcao = calcular_preco_opcao_MonteCarlo(S, K, T, r, sigma)
elif opcao_metodo == 3:
    preco_opcao = calcular_preco_opcao_Binomial(S, K, T, r, sigma)

# Exibe o preço da opção
print(f"Preço da opção calculado: {preco_opcao}")
