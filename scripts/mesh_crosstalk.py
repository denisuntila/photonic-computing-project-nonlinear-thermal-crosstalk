#!/usr/bin/env python3
"""
Previsione del crosstalk termico in una mesh di Clements 6x6.

Topologia (dallo schizzo): 6 colonne di MZI, alternate tra le coppie di
guide (0,1),(2,3),(4,5) e (1,2),(3,4): 15 MZI. Ogni MZI ha due heater sul
braccio superiore: phi prima del primo accoppiatore, theta tra i due
accoppiatori. In uscita uno schermo di 6 heater (uno per guida). Totale:
36 heater, disposti in 13 file alla stessa z:

    fila 2c    : heater phi degli MZI della colonna c
    fila 2c+1  : heater theta degli MZI della colonna c
    fila 12    : schermo di uscita (tutte le guide)

Confronto tra due modelli della mesh:
  semplice  ogni heater ha risposta lineare (P = psi / (kappa A_jj)) e
            scalda solo la propria guida: e' il modello con cui la mesh
            viene programmata, e secondo cui la matrice realizzata
            coinciderebbe con quella voluta (F = 1)
  non lineare + crosstalk
            modello termico non lineare con crosstalk (dalle simulazioni
            FEM): le stesse potenze producono fasi diverse, anche sulle
            guide senza heater

Si prevedono quindi, per le potenze calcolate con il modello semplice:
  * lo sfasamento indesiderato su ogni tratto di guida della mesh;
  * la fedelta' della matrice realizzata rispetto a quella voluta;
  * il contributo separato della non linearita' degli heater e del
    crosstalk.

Ipotesi: accoppiatori ideali 50:50, nessuna perdita, passo uniforme di
20 um come nella simulazione FEM, file di heater diverse separate lungo z
abbastanza da essere termicamente indipendenti.

Uso:
    python3 mesh_crosstalk.py [--model crosstalk_out] [--samples 1000]
                              [--seed 0] [--outdir mesh_out]
                              [--example switch|median]
"""

import argparse
import csv
import math
import os
import time

import numpy as np
from scipy.stats import unitary_group

N = 6
N_COLS = 6
N_SUB = 2 * N_COLS + 1          # 12 sotto-colonne di MZI + schermo di uscita
LAMBDA_UM = 1.55
DN_DT = 1.86e-4
KAPPA = 2.0 * math.pi / LAMBDA_UM * DN_DT      # rad / (K um)
P_MAX_MODEL = 50.0                              # mW, limite del fit


# ======================================================================
# Modello termico
# ======================================================================

class ThermalModel:
    """I(P) [K um] per una sotto-colonna, P [mW] vettore di 6 potenze."""

    def __init__(self, A, S, X, name=""):
        self.A = np.asarray(A, float)          # (N, N)
        self.S = np.asarray(S, float)          # (3, N, N): P^2, P^3, P^4
        self.X = np.asarray(X, float)          # (3, N, N, N): coppie j < l
        self.name = name
        self.pairs = [(j, l) for j in range(N) for l in range(j + 1, N)
                      if np.any(self.X[:, :, j, l] != 0)]

    def linear(self):
        return ThermalModel(self.A, np.zeros_like(self.S),
                            np.zeros_like(self.X), self.name + " (solo A)")

    def I(self, P):
        P = np.asarray(P, float)
        I = self.A @ P
        for k in range(3):
            I = I + self.S[k] @ P**(k + 2)
        for j, l in self.pairs:
            a, b = P[j], P[l]
            I = I + (self.X[0, :, j, l] * a * b
                     + self.X[1, :, j, l] * a * b * (a + b) / 2
                     + self.X[2, :, j, l] * a * b * (a * a + b * b) / 2)
        return I

    def jacobian(self, P):
        P = np.asarray(P, float)
        J = self.A.copy()
        for k in range(3):
            J = J + self.S[k] * ((k + 2) * P**(k + 1))[np.newaxis, :]
        for j, l in self.pairs:
            a, b = P[j], P[l]
            X0, X1, X2 = self.X[0, :, j, l], self.X[1, :, j, l], self.X[2, :, j, l]
            J[:, j] += X0 * b + X1 * (2 * a * b + b * b) / 2 + X2 * (3 * a * a * b + b**3) / 2
            J[:, l] += X0 * a + X1 * (2 * a * b + a * a) / 2 + X2 * (3 * b * b * a + a**3) / 2
        return J


def _read_matrix_csv(path):
    rows = [r for r in csv.reader(open(path)) if r and not r[0].startswith("#")]
    return np.array([[float(x) for x in r[1:]] for r in rows[1:]])


def load_model(directory):
    """Legge i CSV scritti da analyze_crosstalk.py."""
    A = _read_matrix_csv(os.path.join(directory, "A_K_um_per_mW.csv"))
    S = np.zeros((3, N, N))
    for k, name in enumerate(["B_self.csv", "C_self.csv", "D_self.csv"]):
        path = os.path.join(directory, name)
        if os.path.exists(path):
            S[k] = np.nan_to_num(_read_matrix_csv(path))
    X = np.zeros((3, N, N, N))
    path = os.path.join(directory, "B_cross.csv")
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            i, j, l = int(r["WG_i"]), int(r["heater_j"]), int(r["heater_k"])
            for k, col in enumerate(["B_K_um_per_mW2", "C_K_um_per_mW3",
                                     "D_K_um_per_mW4"]):
                if col in r and r[col] not in ("", "nan"):
                    X[k, i, j, l] = float(r[col])
    return ThermalModel(A, S, X, f"CSV da {directory}/")


def default_model():
    """
    Valori dall'output di analyze_crosstalk.py della campagna (125 run).
    Incompleti: C e D solo in diagonale (dai pesi riportati), interazione
    solo tra heater adiacenti. Con i CSV completi usare --model.
    """
    A = np.array([
        [188.7,  2.847,  1.3,   0.681,  0.3649, 0.2005],
        [2.847,  188.6,  2.78,  1.267,  0.667,  0.3648],
        [1.3,    2.781,  188.5, 2.765,  1.267,  0.681],
        [0.681,  1.267,  2.765, 188.6,  2.779,  1.3],
        [0.3648, 0.667,  1.267, 2.779,  188.5,  2.846],
        [0.2005, 0.3649, 0.681, 1.3,    2.846,  188.7]])
    B = np.array([
        [-0.1072,   7.128e-05, 1.745e-05, 4.312e-06, 1.191e-06, 2.629e-07],
        [7.193e-05, -0.1072,   6.814e-05, 1.654e-05, 4.442e-06, 1.248e-06],
        [1.75e-05,  6.792e-05, -0.1072,   6.694e-05, 1.663e-05, 4.46e-06],
        [4.629e-06, 1.662e-05, 6.735e-05, -0.1073,   6.826e-05, 1.753e-05],
        [1.351e-06, 4.365e-06, 1.672e-05, 6.838e-05, -0.1071,   7.201e-05],
        [3.532e-07, 1.331e-06, 4.47e-06,  1.757e-05, 7.137e-05, -0.1075]])
    S = np.zeros((3, N, N))
    S[0] = B
    d = np.arange(N)
    # Pesi riportati a 20 mW: 3o ordine +0.0243 %, 4o ordine -0.0005 %.
    S[1][d, d] = 2.43e-4 * A[d, d] / 20.0**2
    S[2][d, d] = -5e-6 * A[d, d] / 20.0**3
    X = np.zeros((3, N, N, N))
    for i, j, l, b in [(1, 0, 1, -2.4379e-3), (0, 0, 1, -2.4311e-3),
                       (2, 1, 2, -2.3937e-3), (1, 1, 2, -2.3912e-3),
                       (3, 2, 3, -2.3837e-3), (2, 2, 3, -2.3825e-3),
                       (3, 3, 4, -2.3921e-3), (4, 3, 4, -2.3884e-3),
                       (4, 4, 5, -2.4356e-3), (5, 4, 5, -2.4338e-3)]:
        X[0, i, j, l] = b
    return ThermalModel(A, S, X, "valori della campagna (default)")


# ======================================================================
# Mesh: MZI, decomposizione di Clements, matrice realizzata
# ======================================================================

BS = np.array([[1, 1j], [1j, 1]]) / math.sqrt(2)     # accoppiatore 50:50


def mzi(theta, phi):
    """MZI = BS * diag(e^{i theta},1) * BS * diag(e^{i phi},1)."""
    s, c = math.sin(theta / 2), math.cos(theta / 2)
    g = 1j * np.exp(1j * theta / 2)
    return g * np.array([[s * np.exp(1j * phi), c],
                         [c * np.exp(1j * phi), -s]])


def embed(T2, a):
    T = np.eye(N, dtype=complex)
    T[a:a + 2, a:a + 2] = T2
    return T


def column_pairs(c):
    return [0, 2, 4] if c % 2 == 0 else [1, 3]


def heater_rows(s):
    return list(range(N)) if s == N_SUB - 1 else column_pairs(s // 2)


def clements(U):
    """
    Decomposizione di Clements. Ritorna (mzis, out_phases) con
    mzis[(colonna, modo_superiore)] = (theta, phi).
    """
    U = U.astype(complex).copy()
    right, left = [], []
    for i in range(1, N):
        if i % 2 == 1:
            for j in range(i):
                r, a = N - 1 - j, i - 1 - j
                x, y = U[r, a], U[r, a + 1]
                theta = 2 * math.atan2(abs(y), abs(x))
                phi = (np.angle(x) - np.angle(-y)) if abs(x) > 1e-14 and abs(y) > 1e-14 else 0.0
                U = U @ embed(mzi(theta, phi), a).conj().T
                right.append((a, theta, phi))
        else:
            for j in range(1, i + 1):
                a, col = N + j - i - 2, j - 1
                x, y = U[a, col], U[a + 1, col]
                theta = 2 * math.atan2(abs(x), abs(y))
                phi = (np.angle(y) - np.angle(x)) if abs(x) > 1e-14 and abs(y) > 1e-14 else 0.0
                U = embed(mzi(theta, phi), a) @ U
                left.append((a, theta, phi))
    D = np.diag(np.diag(U))

    # Porta gli MZI applicati da sinistra a destra di D:
    # T^dagger D = D' T'(theta', phi').
    moved = []
    for a, theta, phi in reversed(left):
        X = embed(mzi(theta, phi), a).conj().T @ D
        x = X[a:a + 2, a:a + 2]
        th = 2 * math.atan2(abs(x[0, 0]), abs(x[0, 1]))
        s, c = math.sin(th / 2), math.cos(th / 2)
        g = 1j * np.exp(1j * th / 2)
        if c < 1e-12:                          # theta' = pi
            ph, d0, d1 = 0.0, x[0, 0] / g, -x[1, 1] / g
        elif s < 1e-12:                        # theta' = 0
            ph, d0, d1 = 0.0, x[0, 1] / g, x[1, 0] / g
        else:
            d0, d1 = x[0, 1] / (g * c), -x[1, 1] / (g * s)
            ph = np.angle(x[0, 0] / (d0 * g * s))
        D = D.copy()
        D[a, a], D[a + 1, a + 1] = d0, d1
        moved.append((a, th, ph))              # ordine: M'_p, ..., M'_1

    sequence = right + moved                   # ordine di attraversamento
    last = [-1] * N
    mzis = {}
    for a, theta, phi in sequence:
        c = max(last[a], last[a + 1]) + 1
        if c % 2 != a % 2:
            c += 1
        last[a] = last[a + 1] = c
        mzis[(c, a)] = (theta % (2 * math.pi), phi % (2 * math.pi))
    assert all(c < N_COLS for c, _ in mzis) and len(mzis) == N * (N - 1) // 2
    return mzis, np.angle(np.diag(D)) % (2 * math.pi)


def ideal_phases(mzis, out_phases):
    """Fasi obiettivo psi[s, riga] (13 x 6): zero dove non c'e' heater."""
    psi = np.zeros((N_SUB, N))
    for (c, a), (theta, phi) in mzis.items():
        psi[2 * c, a] = phi
        psi[2 * c + 1, a] = theta
    psi[N_SUB - 1] = out_phases
    return psi


def mesh_matrix(psi):
    """Matrice della mesh date le fasi reali su tutte le guide (13 x 6)."""
    U = np.eye(N, dtype=complex)
    for c in range(N_COLS):
        DC = np.eye(N, dtype=complex)
        for a in column_pairs(c):
            DC[a:a + 2, a:a + 2] = BS
        U = DC @ (np.exp(1j * psi[2 * c])[:, None] * U)
        U = DC @ (np.exp(1j * psi[2 * c + 1])[:, None] * U)
    return np.exp(1j * psi[N_SUB - 1])[:, None] * U


def fidelity(U, V):
    return abs(np.trace(U.conj().T @ V))**2 / N**2


# ======================================================================
# Programmazione e previsione
# ======================================================================

def realized_phases(model, P):
    """Fasi reali su tutte le guide di tutte le file (13 x 6)."""
    return np.array([KAPPA * model.I(P[s]) for s in range(N_SUB)])


def solve_single(model, j, target, P0):
    """Potenza dell'heater j, acceso da solo, per la fase target sulla sua guida."""
    P = P0
    for _ in range(30):
        e = np.zeros(N)
        e[j] = P
        f = KAPPA * model.I(e)[j] - target
        step = f / (KAPPA * model.jacobian(e)[j, j])
        P -= step
        if abs(step) < 1e-12:
            break
    return P


def program_linear(model, psi):
    """Potenze con il modello semplice: P = psi / (kappa A_jj)."""
    P = np.zeros((N_SUB, N))
    for s in range(N_SUB):
        for a in heater_rows(s):
            P[s, a] = psi[s, a] / (KAPPA * model.A[a, a])
    return P


def program_calibrated(model, psi):
    """
    Potenze con calibrazione non lineare a heater singolo: ogni heater,
    acceso da solo, da' esattamente la fase voluta sulla propria guida.
    Serve a isolare il contributo del solo crosstalk.
    """
    P = np.zeros((N_SUB, N))
    for s in range(N_SUB):
        for a in heater_rows(s):
            P[s, a] = solve_single(model, a, psi[s, a],
                                   psi[s, a] / (KAPPA * model.A[a, a]))
    return P


def without_crosstalk(model):
    """Modello non lineare senza crosstalk: ogni heater scalda solo la sua guida."""
    D = np.eye(N, dtype=bool)
    return ThermalModel(np.where(D, model.A, 0.0),
                        np.where(D[np.newaxis], model.S, 0.0),
                        np.zeros_like(model.X), "senza crosstalk")


# ======================================================================
# Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="crosstalk_out",
                    help="cartella con i CSV di analyze_crosstalk.py")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="mesh_out")
    ap.add_argument("--example", choices=["switch", "median"], default="switch",
                    help="configurazione mostrata nella mappa: 'switch' = tutti "
                         "gli MZI in stato bar (heater theta a P_pi), phi a pi/2 e "
                         "schermo di uscita alternato acceso/spento; 'median' = "
                         "matrice casuale con 1-F mediana")
    args = ap.parse_args()

    if os.path.exists(os.path.join(args.model, "A_K_um_per_mW.csv")):
        model = load_model(args.model)
    else:
        model = default_model()
    print(f"Modello termico: {model.name}")
    print(f"kappa = {KAPPA:.4e} rad/(K um), P_pi (heater 0) = "
          f"{math.pi / (KAPPA * model.A[0, 0]):.2f} mW")

    print("\nTopologia: Clements 6x6, 15 MZI, 36 heater in 13 file")
    for s in range(N_SUB):
        kind = ("uscita" if s == N_SUB - 1 else
                f"colonna {s // 2}, {'phi' if s % 2 == 0 else 'theta'}")
        print(f"  fila {s:2d} ({kind:18s}): heater sulle guide {heater_rows(s)}")

    rng = np.random.default_rng(args.seed)
    cases = {
        "nl_xt":       "modello non lineare + crosstalk",
        "solo_nl":     "solo non linearita' degli heater",
        "solo_xt":     "solo crosstalk (heater calibrati singolarmente)",
    }
    infid = {k: [] for k in cases}
    err_signed = np.zeros((args.samples, N_SUB, N))   # fase (non lineare + crosstalk) - voluta
    psi_ideal_all = np.zeros((args.samples, N_SUB, N))
    psi_real_all = np.zeros((args.samples, N_SUB, N))
    check_simple = 0.0
    P_max = 0.0
    nl_only = without_crosstalk(model)

    t0 = time.time()
    for n in range(args.samples):
        U = unitary_group.rvs(N, random_state=rng)
        psi = ideal_phases(*clements(U))
        P = program_linear(model, psi)
        P_max = max(P_max, P.max())

        # Modello semplice: la fase su ogni guida con heater e' kappa A_jj P,
        # cioe' esattamente quella voluta -> la mesh e' perfetta.
        psi_simple = np.zeros((N_SUB, N))
        for s_ in range(N_SUB):
            for a in heater_rows(s_):
                psi_simple[s_, a] = KAPPA * model.A[a, a] * P[s_, a]
        check_simple = max(check_simple,
                           1 - fidelity(U, mesh_matrix(psi_simple)))

        psi_real = realized_phases(model, P)
        infid["nl_xt"].append(1 - fidelity(U, mesh_matrix(psi_real)))
        infid["solo_nl"].append(
            1 - fidelity(U, mesh_matrix(realized_phases(nl_only, P))))
        infid["solo_xt"].append(
            1 - fidelity(U, mesh_matrix(realized_phases(
                model, program_calibrated(model, psi)))))
        err_signed[n] = (psi_real - psi + math.pi) % (2 * math.pi) - math.pi
        psi_ideal_all[n] = psi
        psi_real_all[n] = psi_real
    elapsed = time.time() - t0

    err_abs = np.abs(err_signed)
    has_heater = np.zeros((N_SUB, N), bool)
    for s in range(N_SUB):
        has_heater[s, heater_rows(s)] = True
    mean_map = err_abs.mean(axis=0) * 1e3            # mrad
    max_map = err_abs.max(axis=0) * 1e3
    mean_signed = err_signed.mean(axis=0) * 1e3

    print(f"\n{args.samples} matrici unitarie casuali (Haar), potenza max "
          f"{P_max:.1f} mW  [{elapsed:.0f} s]")
    print(f"Controllo: con il modello semplice 1-F = {check_simple:.1e} "
          "(mesh perfetta, come atteso)")

    print("\n=== Differenza di fase: non lineare + crosstalk meno semplice [mrad] ===")
    for label, mask in [("guide con heater", has_heater),
                        ("guide senza heater", ~has_heater)]:
        v = err_signed[:, mask] * 1e3
        print(f"  {label:20s} media {v.mean():+8.2f}   media |.| "
              f"{np.abs(v).mean():7.2f}   max |.| {np.abs(v).max():7.2f}")
    print("  (sulle guide con heater prevale il difetto di fase dovuto alla")
    print("   sublinearita' dell'heater, su quelle senza l'eccesso da crosstalk)")
    print("\n  Media di |differenza| per fila e guida [mrad] (* = tratto con heater):")
    print("         " + "".join(f"   guida {i}" for i in range(N)))
    for s in range(N_SUB):
        name = "uscita" if s == N_SUB - 1 else f"c{s // 2} {'phi' if s % 2 == 0 else 'th '}"
        print(f"  {name:7s}" + "".join(
            f"{mean_map[s, i]:9.1f}{'*' if has_heater[s, i] else ' '} "
            for i in range(N)))

    print("\n=== Fedelta' della matrice realizzata ===")
    print(f"  {'caso':50s} {'1-F mediana':>12s} {'1-F media':>12s} "
          f"{'1-F max':>12s} {'F mediana':>10s}")
    print(f"  {'modello semplice (lineare, senza crosstalk)':50s} "
          f"{0:12.3e} {0:12.3e} {0:12.3e} {1:10.5f}")
    for k in cases:
        v = np.array(infid[k])
        print(f"  {cases[k]:50s} {np.median(v):12.3e} {v.mean():12.3e} "
              f"{v.max():12.3e} {1 - np.median(v):10.5f}")

    # --- File ------------------------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "infidelity.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample"] + [f"1-F_{k}" for k in cases])
        for n in range(args.samples):
            w.writerow([n] + [f"{infid[k][n]:.6e}" for k in cases])
    # Configurazione di esempio per la mappa.
    v = np.array(infid["nl_xt"])
    if args.example == "switch":
        # Configurazione a switch: tutti gli MZI in stato bar (theta = pi,
        # heater a P_pi), heater phi a pi/2 (P_pi/2), schermo di uscita
        # alternato (guide pari a pi, dispari spente). Idealmente la mesh
        # realizza l'identita' in potenza: la luce resta nella propria
        # guida, con fasi diverse da guida a guida.
        ex_ideal = np.zeros((N_SUB, N))
        for c in range(N_COLS):
            for a in column_pairs(c):
                ex_ideal[2 * c, a] = math.pi / 2
                ex_ideal[2 * c + 1, a] = math.pi
        ex_ideal[N_SUB - 1, 0::2] = math.pi
        ex_P = program_linear(model, ex_ideal)
        ex_real = realized_phases(model, ex_P)
        ex_label = ("configurazione a switch: MZI in stato bar, "
                    "phi = pi/2, uscita alternata")
    else:
        ex = int(np.argmin(np.abs(v - np.median(v))))
        ex_ideal, ex_real = psi_ideal_all[ex], psi_real_all[ex]
        ex_label = f"matrice casuale n. {ex} (1-F mediana)"
    U_ex = mesh_matrix(ex_ideal)
    V_ex = mesh_matrix(ex_real)
    # Sfasamento indesiderato accumulato lungo ciascuna guida: somma su
    # tutte le file della differenza tra fase reale e ideale.
    ex_dev = (ex_real - ex_ideal + math.pi) % (2 * math.pi) - math.pi
    ex_accum = ex_dev.sum(axis=0)
    ex_infid = 1 - fidelity(U_ex, V_ex)
    leak = 1 - np.abs(np.diag(V_ex))**2 if args.example == "switch" else None
    print(f"\n=== Esempio per la mappa: {ex_label} ===")
    print(f"  1-F = {ex_infid:.3e}")
    print("  sfasamento indesiderato accumulato per guida 0..5 [rad]: " +
          ", ".join(f"{x:+.3f}" for x in ex_accum))
    if leak is not None:
        print("  potenza che esce da guide diverse da quella d'ingresso, per "
              "ingresso 0..5: " + ", ".join(f"{100 * x:.2f}%" for x in leak))
        print("  matrice di trasferimento in potenza |V|^2 (riga = uscita):")
        for row in np.abs(V_ex)**2:
            print("    " + "  ".join(f"{x:7.4f}" for x in row))

    with open(os.path.join(args.outdir, "example_phases.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fila", "guida", "heater", "fase_ideale_rad",
                    "fase_nonlineare_crosstalk_rad"])
        # Ultime righe: accumulato per guida (fila = "somma").
        for s in range(N_SUB):
            for i in range(N):
                w.writerow([s, i, int(has_heater[s, i]),
                            f"{ex_ideal[s, i]:.6f}",
                            f"{ex_real[s, i]:.6f}"])
        for i in range(N):
            w.writerow(["somma", i, "", "0.000000", f"{ex_accum[i]:.6f}"])

    with open(os.path.join(args.outdir, "crosstalk_map.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fila", "guida", "heater", "differenza_media_mrad",
                    "differenza_media_abs_mrad", "differenza_max_abs_mrad"])
        for s in range(N_SUB):
            for i in range(N):
                w.writerow([s, i, int(has_heater[s, i]),
                            f"{mean_signed[s, i]:.4f}",
                            f"{mean_map[s, i]:.4f}", f"{max_map[s, i]:.4f}"])

    # --- Grafici ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\nmatplotlib non disponibile: salvati solo i CSV in {args.outdir}/")
        return

    # 1. Mappa delle fasi su ogni tratto di guida per la matrice di
    #    esempio: modello ideale (lineare, senza crosstalk) e modello non
    #    lineare + crosstalk. File in orizzontale nell'ordine attraversato
    #    dalla luce, guide in verticale.
    xlabels = [("out" if s == N_SUB - 1 else
                f"c{s // 2}\n{'phi' if s % 2 == 0 else 'theta'}")
               for s in range(N_SUB)]
    vmax = max(ex_ideal.max(), ex_real.max())
    acc_lim = max(np.abs(ex_accum).max(), 1e-12)
    fig = plt.figure(figsize=(11.5, 7.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[N_SUB, 1.3, 0.35],
                          wspace=0.08, hspace=0.25)
    panels = [(ex_ideal, np.zeros(N), "Modello ideale (lineare, senza crosstalk)"),
              (ex_real, ex_accum, "Modello non lineare + crosstalk")]
    for row, (ph, acc, title) in enumerate(panels):
        ax = fig.add_subplot(gs[row, 0])
        im = ax.imshow(ph.T, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
        for s in range(N_SUB):
            for i in range(N):
                if has_heater[s, i]:
                    ax.add_patch(plt.Rectangle((s - 0.5, i - 0.5), 1, 1,
                                               fill=False, ec="white", lw=1.5))
                ax.text(s, i, f"{ph[s, i]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="black" if ph[s, i] > 0.6 * vmax else "white")
        ax.set_yticks(range(N))
        ax.set_yticklabels([f"guida {i}" for i in range(N)], fontsize=8)
        ax.set_xticks(range(N_SUB))
        ax.set_xticklabels(xlabels if row == 1 else [""] * N_SUB, fontsize=8)
        ax.set_title(title, fontsize=10)

        # Colonna dell'accumulato (scala propria, divergente).
        axa = fig.add_subplot(gs[row, 1], sharey=ax)
        axa.imshow(acc[:, None], cmap="RdBu_r", vmin=-acc_lim, vmax=acc_lim,
                   aspect="auto")
        for i in range(N):
            axa.text(0, i, f"{acc[i]:+.2f}", ha="center", va="center",
                     fontsize=7.5,
                     color="white" if abs(acc[i]) > 0.6 * acc_lim else "black")
        axa.set_xticks([0])
        axa.set_xticklabels(["accum."] if row == 1 else [""], fontsize=8)
        axa.tick_params(axis="y", labelleft=False)
        axa.set_title(r"$\Sigma\,\Delta\psi$", fontsize=10)

        cax = fig.add_subplot(gs[row, 2])
        fig.colorbar(im, cax=cax, label="fase [rad]")
    fig.suptitle(f"Fasi su ogni tratto di guida, {ex_label} "
                 f"(1-F = {ex_infid:.1e}); riquadri: tratti con heater;\n"
                 r"$\Sigma\,\Delta\psi$: sfasamento indesiderato accumulato "
                 "lungo la guida [rad]", fontsize=10)
    fig.savefig(os.path.join(args.outdir, "crosstalk_map.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # 1b. Matrice di trasferimento in potenza dell'esempio: ideale e reale.
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, (M, title) in zip(axes, [(np.abs(U_ex)**2, "Modello ideale"),
                                     (np.abs(V_ex)**2, "Non lineare + crosstalk")]):
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        for r in range(N):
            for c in range(N):
                ax.text(c, r, f"{M[r, c]:.3f}", ha="center", va="center",
                        fontsize=7, color="white" if M[r, c] > 0.5 else "black")
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xlabel("guida d'ingresso")
        ax.set_ylabel("guida d'uscita")
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=axes, label="frazione di potenza", shrink=0.85)
    fig.suptitle(f"Trasferimento in potenza $|U_{{ij}}|^2$, {ex_label}",
                 fontsize=10)
    fig.savefig(os.path.join(args.outdir, "example_transfer.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # 2. Distribuzione di 1-F: modello non lineare + crosstalk e contributi separati.
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    all_v = np.concatenate([infid[k] for k in cases])
    bins = np.logspace(np.log10(all_v.min()), np.log10(all_v.max()), 50)
    for k, c, ls in (("nl_xt", "black", "-"), ("solo_nl", "tab:blue", "--"),
                     ("solo_xt", "tab:red", "--")):
        ax.hist(infid[k], bins=bins, histtype="step", lw=1.5, color=c, ls=ls,
                label=f"{cases[k]} ({np.median(infid[k]):.1e})")
    ax.set_xscale("log")
    ax.set_xlabel("infedelta' 1 - F rispetto al modello semplice")
    ax.set_ylabel("numero di matrici")
    ax.set_title(f"Mesh di Clements 6x6: {args.samples} matrici casuali "
                 "(tra parentesi la mediana)", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "infidelity_hist.png"), dpi=150)
    plt.close(fig)

    print(f"\nRisultati salvati in {args.outdir}/")


if __name__ == "__main__":
    main()


