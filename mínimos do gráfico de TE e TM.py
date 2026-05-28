import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from google.colab import files

# ==========================================
# 1. UPLOAD DO ARQUIVO CSV
# ==========================================
print("Clique abaixo para carregar o seu arquivo CSV:")
uploaded = files.upload()
file_name = list(uploaded.keys())[0]

# ==========================================
# 2. CARREGAR E ORGANIZAR OS DADOS
# ==========================================
df = pd.read_csv(file_name)

freq = df.iloc[:, 0].values
c1 = df.iloc[:, 1].values
c2 = df.iloc[:, 2].values

# Nomes exatos das curvas para usar nas legendas e eixos
legend_c1 = "S(2:1,1:1)"
legend_c2 = "S(2:2,1:2)"

# ==========================================
# 3. DETECÇÃO DE MÍNIMOS COM FILTRO DE PROMINÊNCIA
# ==========================================
# Multiplicamos por -1 porque a função busca 'picos'. Procurar picos no sinal invertido
# é matematicamente idêntico a encontrar os vales (mínimos) do sinal original.
# O parâmetro 'prominence=1.0' garante que variações menores que 1 dB sejam ignoradas.
minima_idx1, _ = find_peaks(-c1, prominence=1.0)
minima_idx2, _ = find_peaks(-c2, prominence=1.0)

# ==========================================
# 4. EXIBIR OS RESULTADOS FORMATADOS
# ==========================================
print("\n" + "="*50)
print(f" MÍNIMOS REGIONAIS RELEVANTES - {legend_c1}")
print("="*50)
for f, val in zip(freq[minima_idx1], c1[minima_idx1]):
    print(f"Frequência: {f:7.4f} GHz  |  Magnitude: {val:8.4f} dB")

print("\n" + "="*50)
print(f" MÍNIMOS REGIONAIS RELEVANTES - {legend_c2}")
print("="*50)
for f, val in zip(freq[minima_idx2], c2[minima_idx2]):
    print(f"Frequência: {f:7.4f} GHz  |  Magnitude: {val:8.4f} dB")

# ==========================================
# 5. PLOTAR O GRÁFICO CORRIGIDO
# ==========================================
plt.figure(figsize=(12, 6.5))

# Plot das curvas completas com os nomes corretos
plt.plot(freq, c1, label=legend_c1, color='#1f77b4', linewidth=1.5, alpha=0.8)
plt.plot(freq, c2, label=legend_c2, color='#ff7f0e', linewidth=1.5, alpha=0.8)

# Marcação dos mínimos significativos filtrados
plt.scatter(freq[minima_idx1], c1[minima_idx1], color='blue', edgecolor='black', s=60, zorder=5, label=f'Mínimos {legend_c1}')
plt.scatter(freq[minima_idx2], c2[minima_idx2], color='darkorange', edgecolor='black', s=60, zorder=5, label=f'Mínimos {legend_c2}')

# Configurações do Layout
plt.xlabel("Frequency [GHz]", fontsize=11)
plt.ylabel("Magnitude [dB]", fontsize=11)
plt.title("Parâmetros S - Identificação de Mínimos Regionais (Filtrados)", fontsize=13, fontweight='bold', pad=15)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower left', frameon=True, shadow=True, fontsize=10)

# Ajusta as margens para não cortar as legendas
plt.tight_layout()
plt.show()
