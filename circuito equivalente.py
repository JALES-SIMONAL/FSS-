import numpy as np

# =============================================================================
# PARÂMETROS DE ENTRADA — preencha os valores abaixo
# =============================================================================

p      = 17.5e-3    # Periodicidade da célula unitária [m]
w      = 0.5e-3     # Largura do gap [m]
lam    = 30.0e-3    # Comprimento de onda no meio [m]
phi    = 0.0        # Ângulo de incidência para reatância indutiva [radianos]
Z0     = 377.0      # Impedância de onda do meio [Ohm] (377 Ω no vácuo)

g      = 0.5e-3     # Largura do gap capacitivo (separação entre patches) [m]
theta  = 0.0        # Ângulo de incidência para susceptância capacitiva [radianos]
eps_r  = 4.4        # Permissividade relativa do substrato (susceptância corrigida)

valores_d = [2.5e-3, 2.5e-3, 5.0e-3, 7.5e-3, 12.5e-3]  # lado da espira [m] - u da FSS Fibonacci

# --- frequência derivada de λ ---
c     = 3e8
f     = c / lam
omega = 2 * np.pi * f

# =============================================================================
# FUNÇÃO AUXILIAR: F(p, w_eff, λ, angle)
# =============================================================================

def calcular_F(p, w_eff, lam, angle):
    beta = np.sin(np.pi * w_eff / (2 * p))

    def C_pm(sign):
        radicando = 1 + sign * (2 * p * np.sin(angle) / lam) - (p * np.cos(angle) / lam)**2
        return 1.0 / np.sqrt(radicando) - 1

    C_plus  = C_pm(+1)
    C_minus = C_pm(-1)

    num_G = (
        0.5 * (1 - beta**2)**2
        * ((1 - beta**2 / 4) * (C_plus + C_minus) + 4 * beta**2 * C_plus * C_minus)
    )
    den_G = (
        (1 - beta**2 / 4)
        + beta**2 * (1 + beta**2 / 2 - beta**4 / 8) * (C_plus + C_minus)
        + 2 * beta**6 * C_plus * C_minus
    )
    G = num_G / den_G

    F = (p * np.cos(angle) / lam) * (np.log(1.0 / np.sin(np.pi * w_eff / (2 * p))) + G)

    return F, beta, C_plus, C_minus, G

# =============================================================================
# PRÉ-CÁLCULO DOS F (independentes de d)
# =============================================================================

F_trilha, beta_t, Cp_t, Cm_t, G_t = calcular_F(p, w,   lam, phi)
F_espira, beta_e, Cp_e, Cm_e, G_e = calcular_F(p, 2*w, lam, phi)
F_cap,    beta_c, Cp_c, Cm_c, G_c = calcular_F(p, g,   lam, theta)

# =============================================================================
# IMPRESSÃO
# =============================================================================

sep  = "=" * 62
sep2 = "-" * 62

print(sep)
print("  CIRCUITO EQUIVALENTE FSS Fibonacci Retangular")
print(sep)
print(f"  p={p*1e3:.2f} mm  w={w*1e3:.2f} mm  g={g*1e3:.2f} mm  λ={lam*1e3:.2f} mm")
print(f"  f={f/1e9:.4f} GHz  ω={omega:.4e} rad/s  Z0={Z0:.1f} Ω  εr={eps_r}")
print(f"  φ={np.degrees(phi):.1f}°  θ={np.degrees(theta):.1f}°")

# ── 1) Reatância — Usando modelo da espira quadrada-──────────────────────────
print()
print("  ── 1) REATÂNCIA — PELO MODELO DA ESPIRA QUADRADA ──")
print("   [ X_L/Z0 = (d/p)·F(p,2w,λ,φ) ] ")
print()
print(f"    β={beta_e:.6f}  C+={Cp_e:.6f}  C−={Cm_e:.6f}  G={G_e:.6f}  F={F_espira:.6f}")
print(f"  {'d=u [mm]':>8}  {'X_L [Ω]':>10}  {'L [nH]':>10}")
print(f"  {sep2}")
for d in valores_d:
    XL  = (d / p) * F_espira * Z0
    L   = XL / omega
    print(f"  {d*1e3:>8.2f}  {XL:>10.4f}  {L*1e9:>10.6f}")

# ── 2) Susceptância — Usando modelo da espira quadrada corrigida ──────────────────────────────
print()
print("  ── 2) SUSCEPTÂNCIA CAPACITIVA — PELO MODELO DA ESPIRA QUADRADA CORRIGIDA ──")
print("   [ B_C/Z0 = 4·εr·(d/p)·F(p,g,λ,θ) ] ")
print()
print(f"    εr={eps_r}  β={beta_c:.6f}  G={G_c:.6f}  F={F_cap:.6f}")
print(f"  {'d=u [mm]':>8}  {'B_C [mS]':>10}  {'C [pF]':>10}")
print(f"  {sep2}")
for d in valores_d:
    BC_norm = 4 * eps_r * (d / p) * F_cap
    BC      = BC_norm / Z0
    C       = BC / omega
    print(f"  {d*1e3:>8.2f}  {BC*1e3:>10.6f}  {C*1e12:>10.6f}")

# ── 3) Frequência de ressonância ───────────────────────────────────
print()
print("  ── 5) FREQUÊNCIA DE RESSONÂNCIA ──")
print("   [ f0 = 1/(2π√LC) ] ")
print()
print(f"  {'d=u [mm]':>8}  {'L [nH]':>10}  {'C [pF]':>10}  {'f0 [GHz]':>12}")
print(f"  {sep2}")
for d in valores_d:
    L  = (d / p) * F_espira * Z0 / omega
    C  = 4 * eps_r * (d / p) * F_cap / Z0 / omega
    f0 = 1.0 / (2 * np.pi * np.sqrt(L * C))
    print(f"  {d*1e3:>8.2f}  {L*1e9:>10.6f}  {C*1e12:>10.6f}  {f0/1e9:>12.6f}")

# ── 4) Frequência de ressonância corrigida — pares cruzados ──────────────────
# ── Considerando somente indutancia vertical e capacitancia horizontal────────

L_vec = [(d / p) * F_espira * Z0 / omega          for d in valores_d]
C_vec = [4 * eps_r * (d / p) * F_cap / Z0 / omega for d in valores_d]

pares = [(0, 1), (2, 1), (2, 3), (4, 3)]

print()
print("  ── 4) FREQUÊNCIA DE RESSONÂNCIA CORRIGIDA — PARES CRUZADOS ──")
print("     [ f0 = 1/(2π√(Lᵢ·Cⱼ))  com Cⱼ = 4·εr·(d/p)·F/Z0/ω ]")
print(f"  {'Par':>8}  {'dL [mm]':>8}  {'dC [mm]':>8}  {'L [nH]':>10}  {'C [pF]':>10}  {'f0 [GHz]':>12}")
print(f"  {sep2}")
for (iL, iC) in pares:
    L   = L_vec[iL]
    C   = C_vec[iC]
    f0  = 1.0 / (2 * np.pi * np.sqrt(L * C))
    par = f"L{iL+1}C{iC+1}"
    print(f"  {par:>8}  {valores_d[iL]*1e3:>8.2f}  {valores_d[iC]*1e3:>8.2f}  {L*1e9:>10.6f}  {C*1e12:>10.6f}  {f0/1e9:>12.6f}")

print()
print(sep)
