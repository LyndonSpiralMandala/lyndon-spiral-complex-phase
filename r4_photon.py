import numpy as np

# === 素数リスト ===
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

# === Jacobi(-29, p) を手入力 ===
chi_m29 = {
    2: -1,
    3:  1,
    5:  1,
    7: -1,
    11: 1,
    13: -1,
    17: 1,
    19: -1,
    23: 1,
    29: 0
}

# === 仮の depth（ビット列由来） ===
depth = {
    2: 0,
    3: 0,
    5: 0,
    7: 1,
    11: 2,
    13: 1,
    17: 0,
    19: 1,
    23: 0,
    29: 0
}

# === パラメータ（R4 最適推定値） ===
q = 0.545
r = 0.589
eps_even = 1.996  # radians
eps_odd  = 3.391  # radians
c = 1.248         # radians per log(p)

# === 局所補正 δ_p ===
delta_p = {
    # 例: p=11 に局所補正を入れる場合
    11: 0.02,
    13: -0.01
    # 必要に応じて追加
}

# === 実データ (r=1) ===
tau_real_r1 = {
    2: -0.265,
    3:  0.299,
    5: -0.346,
    7: -0.188,
    11: 0.500
    # 必要なら追加
}

# === モデル式 ===
def tau_mod(p):
    chi = chi_m29[p]
    d = depth[p]

    theta_base = 0 if chi >= 0 else np.pi
    eps = eps_even if d % 2 == 0 else eps_odd
    rd = r ** d
    theta = theta_base + d * eps + c * np.log(p)
    amp = 2 * p ** 5.5 * q * rd

    base = amp * np.cos(theta)
    return base + delta_p.get(p, 0.0)

def tau_mod_p2(p, tau_p):
    return tau_p ** 2 - p ** 11

def tau_mod_p3(p, tau_p, tau_p2):
    return tau_p * tau_p2 - p ** 11 * tau_p

def tau_norm(tau, p, r):
    return tau / (2 * p ** (5.5 * r))

# === 計算・出力 ===
real_vals = []
pred_vals = []

print("=" * 40)

for p in primes:
    tau_p = tau_mod(p)
    tau_p2 = tau_mod_p2(p, tau_p)
    tau_p3 = tau_mod_p3(p, tau_p, tau_p2)

    tn_p = tau_norm(tau_p, p, 1)
    tn_p2 = tau_norm(tau_p2, p, 2)
    tn_p3 = tau_norm(tau_p3, p, 3)

    if p in tau_real_r1:
        real = tau_real_r1[p]
        real_vals.append(real)
        pred_vals.append(tn_p)

    print(f"p = {p}")
    print(f" τmod(p)   = {tn_p:.3f} (実 = {tau_real_r1.get(p, 'N/A')})")
    print(f" τmod(p²)  = {tn_p2:.3f}")
    print(f" τmod(p³)  = {tn_p3:.3f}")
    print("-" * 30)

# === 誤差指標 ===
real_vals = np.array(real_vals)
pred_vals = np.array(pred_vals)

residuals = real_vals - pred_vals

rmse = np.sqrt(np.mean(residuals ** 2))
mae = np.mean(np.abs(residuals))

print("=" * 40)
print(f"RMSE (r=1) = {rmse:.4f}")
print(f"MAE  (r=1) = {mae:.4f}")
print("=" * 40)

# === ビット列ジェネレータの例（将来拡張用） ===
def depth_from_bits(p, bitstring='0001101'):
    bits = list(map(int, bitstring))
    idx = (p - 2) % len(bits)
    count = 0
    for i in range(idx + 1):
        if bits[i] == 1:
            count += 1
    return count

# depth を自動で差し替えたいとき:
# depth[p] = depth_from_bits(p)