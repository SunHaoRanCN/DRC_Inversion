import numpy as np
from scipy.optimize import root

def find_root(x, n):
    if x < 0:
        g = 1j * pow(-x, n)
    else:
        g = pow(x, n)
    return g


def compressor(x, fs, L, R, T_v_att, T_v_rel, T_g_att, T_g_rel, p=2, normalize_after=False):
    # L The threshold in dB
    # R The compression ratio
    # p The detector type (RMS=2 or peak=1)
    # T_v_att The attack time of the envelope filter in ms
    # T_v_rel The release time of the envelope filter in ms
    # T_g_att The attack time of the gain filter in ms
    # T_g_rel The release time of the gain filter in ms

    if max(np.abs(x)) > 1:
        return print('Warning: x should be normalized!')

    BG = 1 - np.exp(-2.2 / (np.array([T_v_att, T_v_rel, T_g_att, T_g_rel]) / 1000 * fs))

    Batt = BG[0]
    # Batt = 1
    Brel = BG[1]
    # Brel = 1
    Gatt = BG[2]
    # Gatt = 1
    Grel = BG[3]
    # Grel = 1

    S = 1 - 1 / R  # Slope
    l = 10 ** (L / 20)  # l in linear scale
    K = l ** S
    N = len(x)

    B_init = 1  # Initial value
    G_init = 1  # Initial value
    x2 = np.zeros(N)  # x_tilde
    v = np.zeros(N)
    f = np.ones(N)
    g = np.zeros(N)

    for n in range(N):
        if n == 0:  # Initialization
            x2[n] = B_init * np.abs(x[n]) ** p
        else:
            if np.abs(x[n]) > v[n - 1]:  # Attack if x is greater than envelope
                x2[n] = Batt * np.abs(x[n]) ** p + (1 - Batt) * x2[n - 1]
            else:  # Release if x is lower than envelope
                x2[n] = Brel * np.abs(x[n]) ** p + (1 - Brel) * x2[n - 1]

        v[n] = find_root(x2[n], 1 / p)
        # v[n] = x2[n] ** (1 / p)

        # compress or not
        if v[n] > l:
            f[n] = K * find_root(v[n], -S)
            # f[n] = K * (v[n] ** (-S))
        else:
            f[n] = 1

        if n == 0:
            g[n] = G_init * f[n]
        else:
            if f[n] < g[n - 1]:  # Attack
                g[n] = Gatt * f[n] + (1 - Gatt) * g[n - 1]
            else:  # Release
                g[n] = Grel * f[n] + (1 - Grel) * g[n - 1]

    y = g * x

    I = np.isnan(y)
    I2 = np.where(g < 0)
    if np.any(I) or len(I2[0]) > 0:
        print('Warning: Invalid parameters')
        y[I] = 0
        y = np.abs(y) * np.sign(x)

    if normalize_after:
        y = y - np.mean(y)
        y = y / np.max(np.abs(y))

    return y


def Xip(v, eps, B, G, K, S, p, g, y, x_tilde):
    return find_root(G * K * find_root(v, -S) + (1 - G) * g, p) * (find_root(v, p) - (1 - B) * x_tilde) - B * find_root(np.abs(y), p) + eps

### original charzero function
# def CHARFZERO(vn, eps, B, G, K, S, p, g, y, x_tilde):  # find the root of Xip
#     vi = vn
#     xipvi = Xip(vi, eps, B, G, K, S, p, g, y, x_tilde)
#     continuer = True
#     while continuer:
#         Di = np.abs(xipvi)
#         xipvi_Di = Xip(vi + Di, eps, B, G, K, S, p, g, y, x_tilde)
#         vi = vi - Di * xipvi / (xipvi_Di - xipvi + eps)
#         xipvi = Xip(vi, eps, B, G, K, S, p, g, y, x_tilde)
#         if np.abs(xipvi) > Di:
#             v0 = vn
#             return v0
#         vn = vi
#         continuer = np.abs(xipvi) > eps
#     v0 = vi
#     return v0

# ### Scipy function
def CHARFZERO(vn, eps, B, G, K, S, p, g, y, x_tilde):
    root_result = root(Xip, vn, args=(eps, B, G, K, S, p, g, y, x_tilde))
    return root_result.x


def decompressor(y, Fs, L, R, T_v_att, T_v_rel, T_g_att, T_g_rel, p, eps=np.finfo(float).eps, normalize_after=False):

    if max(np.abs(y)) > 1:
        y = y - np.mean(y)
        y = y / np.max(np.abs(y))

    # Convert time to filter coefficients
    BG = 1 - np.exp(-2.2 / (np.array([T_v_att, T_v_rel, T_g_att, T_g_rel]) / 1000 * Fs))

    S = 1 - 1 / R
    l = 10 ** (L / 20)
    K = l ** S  # root(l, S)
    N = len(y)
    x = np.zeros(N)
    g = np.zeros(N)
    v = np.zeros(N)
    x2 = np.zeros(N)

    # x_tilde = 0
    x[0] = y[0]
    g[0] = 1

    for n in range(N):
        if np.abs(y[n]) > find_root(x2[n], 1 / p) * g[n]:
            B = BG[0]
        else:  # Release
            B = BG[1]

        if np.abs(y[n]) > find_root((find_root((K / g[n]), p / S) - (1 - B) * x2[n]) / B, 1 / p) * g[n]:  # Attack
            G = BG[2]
        else:
            G = BG[3]
        if np.abs(y[n]) > find_root((find_root(l, p) - (1 - B) * x2[n]) / B, 1 / p) * (G + (1 - G) * g[n]):
            v[n] = find_root(B * find_root(np.abs(y[n]) / (G + (1 - G) * g[n]), p) + (1 - B) * x2[n], 1 / p)
            if n > 0:
                v0 = CHARFZERO(v[n], eps, B, G, K, S, p, g[n - 1], y[n], x2[n])
            else:
                v0 = CHARFZERO(v[n], eps, B, G, K, S, p, 1, y[n], x2[n])
            x[n] = abs(find_root((find_root(v0, p) - (1 - B) * x2[n]) / B, 1 / p))
            x2[n] = v0 ** p
            g[n] = np.abs(y[n]) / np.abs(x[n])

        else:
            g[n] = G + (1 - G) * g[n]
            x[n] = np.abs(y[n]) / g[n]
            x2[n] = B * find_root(np.abs(x[n]), p) + (1 - B) * x2[n]

        x[n] = np.sign(y[n]) * np.abs(x[n])

        if n < N - 1:
            g[n + 1] = g[n]
            v[n + 1] = v[n]
            x2[n + 1] = x2[n]

    if normalize_after:
        x = x - np.mean(x)
        x = x / np.max(np.abs(x))

    return x
