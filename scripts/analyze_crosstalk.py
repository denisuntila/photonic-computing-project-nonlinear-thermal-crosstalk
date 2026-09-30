#!/usr/bin/env python3
"""
Analisi del crosstalk termico dai risultati di run_campaign.sh.

Modello, per ogni guida i (I_i = integrale di Delta T lungo la guida):

    I_i = sum_j A_ij P_j  +  sum_j B_i,jj P_j^2  +  sum_{j<k} B_i,jk P_j P_k
          + (termini di ordine superiore)

  A_ij     [K um / mW]    crosstalk lineare (solver lineare, linearizzazione
                          del non lineare attorno a T_amb)
  B_i,jj   [K um / mW^2]  auto-non-linearita' dell'heater j
  B_i,jk   [K um / mW^2]  interazione tra gli heater j e k (j != k)

Estrazione, per ogni potenza P disponibile:
  A_ij      = I_i / P                         heater j da solo, solver lineare
  B_i,jj(P) = (I_i - A_ij P) / P^2            heater j da solo, non lineare
  B_i,jk(P) = (I_i(j+k) - I_i(j) - I_i(k)) / P^2
                                              coppia j,k e singoli alla stessa P
Poi B(P) viene fittato con un polinomio in P ed estrapolato a P -> 0:
  B(P) = B + C P            (2-3 potenze)
  B(P) = B + C P + D P^2    (>= 4 potenze)
B e' il coefficiente del secondo ordine; C e D sono il terzo e il quarto
ordine. Nella previsione "completa" i termini di interazione di ordine
superiore sono attribuiti in forma simmetrica, P_j P_k (P_j + P_k)/2 e
P_j P_k (P_j^2 + P_k^2)/2, che coincide con i dati a potenze uguali.

Lo sfasamento si ottiene con  dphi = (2 pi / lambda) (dn/dT) I.

Uso:
    python3 analyze_crosstalk.py results.csv [--outdir crosstalk_out]
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np

N = 6                 # heater / guide
PITCH_UM = 20.0       # passo tra le guide [um] (dal .geo)
H_SUB_UM = 50.0       # spessore del substrato [um] (dal .geo)
LAMBDA_UM = 1.55      # lunghezza d'onda [um]
DN_DT = 1.86e-4       # coefficiente termo-ottico del Si a 1550 nm [1/K]
K_PHI = 2.0 * math.pi / LAMBDA_UM * DN_DT   # [rad / (K um)]
T_LAW_MAX = 500.0     # limite superiore di validita' delle leggi k(T) [K]
P_REF = 20.0          # potenza di riferimento per i pesi relativi [mW]


# ----------------------------------------------------------------------
# Lettura
# ----------------------------------------------------------------------

def load(path):
    """Ritorna {(solver, mask, P_mW): vettore I} (l'ultima riga vince)."""
    data = {}
    T_max_seen = 0.0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["solver"], int(row["mask"], 0), float(row["P_mW"]))
            I = np.array([float(row[f"I_{i}_K_um"]) for i in range(N)])
            data[key] = I

            P_inj = float(row["P_injected_mW"])
            n_on = bin(key[1]).count("1")
            if abs(P_inj - n_on * key[2]) > 1e-6 * max(1.0, P_inj):
                print(f"  ATTENZIONE: {key}: potenza iniettata {P_inj} mW "
                      f"invece di {n_on * key[2]} mW")

            T_max = float(row["T_max_K"])
            T_max_seen = max(T_max_seen, T_max)
            if key[0] == "nonlinear" and T_max > T_LAW_MAX:
                print(f"  ATTENZIONE: {key}: T_max = {T_max:.0f} K oltre "
                      f"la validita' delle leggi k(T) ({T_LAW_MAX:.0f} K)")
    print(f"Letti {len(data)} run, T_max massima = {T_max_seen:.1f} K")
    return data


def active(mask):
    return [j for j in range(N) if (mask >> j) & 1]


def fit_B(powers, B_values):
    """
    Fit polinomiale di B(P) su ciascuna guida. Ritorna (coeff, dev):
    coeff[k] = coefficiente di P^k (coeff[0] = B estrapolato a P -> 0),
    array di forma (3, N) con zeri per i gradi non fittati; dev = massimo
    scarto del fit relativo a |B| (0 se il fit e' esatto).
    """
    powers = np.asarray(powers, dtype=float)
    B_values = np.asarray(B_values, dtype=float)       # (n_P, N)
    n = len(powers)
    deg = 0 if n == 1 else (1 if n <= 3 else 2)
    coeff = np.zeros((3, N))
    if deg == 0:
        coeff[0] = B_values[0]
        return coeff, 0.0
    p = np.polyfit(powers, B_values, deg)              # (deg+1, N), P^deg prima
    for k in range(deg + 1):
        coeff[k] = p[deg - k]
    dev = 0.0
    if n > deg + 1:
        fitted = sum(np.outer(powers**k, coeff[k]) for k in range(3))
        scale = np.maximum(np.abs(B_values).max(axis=0), 1e-300)
        dev = float((np.abs(B_values - fitted) / scale).max())
    return coeff, dev


# ----------------------------------------------------------------------
# Estrazione
# ----------------------------------------------------------------------

def extract(data):
    # --- A -----------------------------------------------------------
    A = np.full((N, N), np.nan)
    for (solver, mask, P), I in data.items():
        if solver == "linear" and bin(mask).count("1") == 1:
            A[:, active(mask)[0]] = I / P
    missing = [j for j in range(N) if np.isnan(A[0, j])]
    if missing:
        sys.exit(f"Mancano i run lineari per gli heater {missing}")

    # --- run non lineari, per numero di heater accesi -----------------------
    single = defaultdict(dict)              # single[j][P] = I
    pair = defaultdict(dict)                # pair[(j,k)][P] = I
    others = []                             # (mask, P, I) per validazione
    for (solver, mask, P), I in data.items():
        if solver != "nonlinear":
            continue
        on = active(mask)
        if len(on) == 1:
            single[on[0]][P] = I
        elif len(on) == 2:
            pair[tuple(on)][P] = I
        else:
            others.append((mask, P, I))

    # --- B_i,jj --------------------------------------------------------
    S = np.full((3, N, N), np.nan)          # S[k, i, j]: coeff. di P^k in B_i,jj(P)
    B_self_byP = {}                         # (j, P) -> vettore B(P)
    dev_self = 0.0
    for j in range(N):
        if not single[j]:
            print(f"  nota: nessun run non lineare singolo per HEATER_{j}")
            continue
        powers = sorted(single[j])
        Bp = [(single[j][P] - A[:, j] * P) / P**2 for P in powers]
        for P, b in zip(powers, Bp):
            B_self_byP[(j, P)] = b
        S[:, :, j], dev = fit_B(powers, Bp)
        dev_self = max(dev_self, dev)

    # --- B_i,jk --------------------------------------------------------
    X = np.full((3, N, N, N), np.nan)       # X[k, i, j, l]: coppia (j,l), j < l
    B_cross_byP = {}                        # (j, l, P) -> vettore B(P)
    for (j, l), runs in sorted(pair.items()):
        powers, Bp = [], []
        for P in sorted(runs):
            if P not in single[j] or P not in single[l]:
                print(f"  nota: coppia ({j},{l}) a P={P:g} mW senza i run "
                      f"singoli alla stessa potenza, ignorata")
                continue
            b = (runs[P] - single[j][P] - single[l][P]) / P**2
            powers.append(P)
            Bp.append(b)
            B_cross_byP[(j, l, P)] = b
        if powers:
            X[:, :, j, l], _ = fit_B(powers, Bp)

    return dict(A=A, S=S, X=X, B_self_byP=B_self_byP,
                B_cross_byP=B_cross_byP, dev_self=dev_self,
                single=single, others=others)


def predict(res, P_vec, order):
    """
    I previsto dal modello per un vettore di potenze [mW].
    order = 1: solo A;  2: A + B (quadratico);  4: fino al quarto ordine.
    """
    A, S, X = res["A"], res["S"], res["X"]
    I = A @ P_vec
    if order < 2:
        return I
    kmax = 0 if order == 2 else 2
    for k in range(kmax + 1):
        I = I + np.nansum(S[k] * P_vec**(k + 2), axis=1)
    for j in range(N):
        for l in range(j + 1, N):
            if np.isnan(X[0, 0, j, l]):
                continue
            Pj, Pl = P_vec[j], P_vec[l]
            I = I + X[0, :, j, l] * Pj * Pl
            if kmax >= 1:
                I = I + X[1, :, j, l] * Pj * Pl * (Pj + Pl) / 2
                I = I + X[2, :, j, l] * Pj * Pl * (Pj**2 + Pl**2) / 2
    return I


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------

def fmt_matrix(M, fmt="{:11.4g}"):
    header = "        " + "".join(f"   heater {j}" for j in range(M.shape[1]))
    lines = [header]
    for i in range(M.shape[0]):
        lines.append(f"  WG {i} " + "".join(fmt.format(v) for v in M[i]))
    return "\n".join(lines)


def save_matrix(path, M, note):
    with open(path, "w", newline="") as f:
        f.write(f"# {note}\n")
        w = csv.writer(f)
        w.writerow(["WG \\ heater"] + [f"heater_{j}" for j in range(M.shape[1])])
        for i in range(M.shape[0]):
            w.writerow([f"WG_{i}"] + [f"{v:.10g}" for v in M[i]])


def crosstalk_vs_distance(A_norm):
    """Media e dispersione di A_ij/A_jj per ciascuna distanza |i-j| >= 1."""
    d_list, mean, lo, hi = [], [], [], []
    for d in range(1, N):
        vals = [A_norm[i, j] for i in range(N) for j in range(N)
                if abs(i - j) == d]
        d_list.append(d * PITCH_UM)
        mean.append(np.mean(vals))
        lo.append(np.min(vals))
        hi.append(np.max(vals))
    return (np.array(d_list), np.array(mean), np.array(lo), np.array(hi))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--outdir", default="crosstalk_out")
    args = ap.parse_args()

    data = load(args.csv)
    res = extract(data)
    A, S, X = res["A"], res["S"], res["X"]
    B_self = S[0]
    os.makedirs(args.outdir, exist_ok=True)
    np.set_printoptions(linewidth=140)

    # --- Lineare --------------------------------------------------------------
    print("\n=== Matrice lineare A [K um / mW] ===")
    print(fmt_matrix(A))
    A_norm = A / np.diag(A)[np.newaxis, :]
    print("\n=== Crosstalk normalizzato A_ij / A_jj ===")
    print(fmt_matrix(A_norm, "{:11.3e}"))
    A_phi = K_PHI * A
    print(f"\n=== Sfasamento lineare [rad / mW]  (lambda = {LAMBDA_UM} um, "
          f"dn/dT = {DN_DT:g} 1/K) ===")
    print(fmt_matrix(A_phi))
    print("\n  P_pi (stima lineare, heater j sulla sua guida): " +
          ", ".join(f"{math.pi / A_phi[j, j]:.2f}" for j in range(N)) + " mW")

    d_um, xt_mean, xt_lo, xt_hi = crosstalk_vs_distance(A_norm)
    sel = d_um >= 2 * PITCH_UM            # oltre il primo vicino
    slope, intercept = np.polyfit(d_um[sel], np.log(xt_mean[sel]), 1)
    L_decay = -1.0 / slope
    print("\n=== Crosstalk in funzione della distanza ===")
    for d, m, lo, hi in zip(d_um, xt_mean, xt_lo, xt_hi):
        print(f"  {d:5.0f} um: {m:.3e}  (min {lo:.3e}, max {hi:.3e})")
    print(f"  Decadimento esponenziale (d >= {2 * PITCH_UM:g} um): "
          f"L = {L_decay:.1f} um   [stima 2 h_sub / pi = "
          f"{2 * H_SUB_UM / math.pi:.1f} um]")

    # --- Auto-non-linearita' --------------------------------------------------
    if not np.all(np.isnan(B_self)):
        powers_self = sorted({P for (_, P) in res["B_self_byP"]})
        print(f"\n=== Auto-non-linearita' B_i,jj [K um / mW^2] "
              f"(estrapolata a P->0, potenze {powers_self} mW) ===")
        print(fmt_matrix(B_self))
        print(f"\n  Peso di ciascun ordine rispetto al lineare a {P_REF:g} mW "
              f"(guida dell'heater, media sui 6 heater):")
        diag = np.arange(N)
        for k, name in [(0, "2o ordine  B P / A      "),
                        (1, "3o ordine  C P^2 / A    "),
                        (2, "4o ordine  D P^3 / A    ")]:
            w = S[k][diag, diag] * P_REF**(k + 1) / A[diag, diag]
            if np.all(w == 0):
                continue
            print(f"    {name}: {100 * np.mean(w):+.4f}%")
        P_max = max(powers_self)
        w2 = S[0][diag, diag] * P_max / A[diag, diag]
        w3 = S[1][diag, diag] * P_max**2 / A[diag, diag]
        print(f"  A {P_max:g} mW: 2o ordine {100 * np.mean(w2):+.3f}%, "
              f"3o ordine {100 * np.mean(w3):+.4f}%")
        if res["dev_self"] > 0:
            print(f"  Scarto massimo del fit B(P): "
                  f"{100 * res['dev_self']:.3f}% di |B|")

    # --- Interazione ------------------------------------------------------------
    n_pairs = int(np.sum(~np.isnan(X[0, 0])))
    if n_pairs:
        print(f"\n=== Interazione B_i,jk [K um / mW^2], {n_pairs} coppie ===")
        rows = [(i, j, l, X[0, i, j, l])
                for j in range(N) for l in range(j + 1, N)
                if not np.isnan(X[0, 0, j, l]) for i in range(N)]
        for i, j, l, b in sorted(rows, key=lambda r: -abs(r[3]))[:10]:
            print(f"  WG {i}, heater {j}+{l}: {b: .4e}")
        print("  (le 10 piu' grandi in valore assoluto; tutte nel CSV)")

    # --- Validazione -------------------------------------------------------------
    if res["others"]:
        print("\n=== Validazione (combinazioni con >= 3 heater accesi) ===")
        print("  errore max sulle 6 guide, in mrad di fase")
        print(f"  {'maschera':>10} {'P [mW]':>7} {'solo A':>10} "
              f"{'A + B':>10} {'completo':>10}")
        for mask, P, I_sim in sorted(res["others"], key=lambda r: (r[0], r[1])):
            P_vec = np.array([P if (mask >> j) & 1 else 0.0 for j in range(N)])
            errs = [K_PHI * np.abs(predict(res, P_vec, o) - I_sim).max() * 1e3
                    for o in (1, 2, 4)]
            print(f"  0b{mask:0{N}b} {P:7g} {errs[0]:10.3f} {errs[1]:10.3f} "
                  f"{errs[2]:10.3f}")

    # --- File -------------------------------------------------------------------
    save_matrix(os.path.join(args.outdir, "A_K_um_per_mW.csv"), A,
                "A_ij [K um / mW]: riga = guida i, colonna = heater j")
    save_matrix(os.path.join(args.outdir, "A_rad_per_mW.csv"), A_phi,
                f"A_ij in sfasamento [rad / mW], lambda={LAMBDA_UM} um, dn/dT={DN_DT}")
    save_matrix(os.path.join(args.outdir, "A_normalized.csv"), A_norm,
                "A_ij / A_jj")
    for k, name in [(0, "B"), (1, "C"), (2, "D")]:
        if np.all(np.nan_to_num(S[k]) == 0):
            continue
        save_matrix(os.path.join(args.outdir, f"{name}_self.csv"), S[k],
                    f"{name}_i,jj: coefficiente di P^{k + 2} [K um / mW^{k + 2}], "
                    "riga = guida i, colonna = heater j")
    with open(os.path.join(args.outdir, "B_cross.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["WG_i", "heater_j", "heater_k", "B_K_um_per_mW2",
                    "C_K_um_per_mW3", "D_K_um_per_mW4", "B_rad_per_mW2"])
        for j in range(N):
            for l in range(j + 1, N):
                if np.isnan(X[0, 0, j, l]):
                    continue
                for i in range(N):
                    w.writerow([i, j, l] +
                               [f"{X[k, i, j, l]:.10g}" for k in range(3)] +
                               [f"{K_PHI * X[0, i, j, l]:.10g}"])

    # --- Grafici ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\nmatplotlib non disponibile: salvati solo i CSV in {args.outdir}/")
        return

    colors = plt.cm.tab10(np.arange(N))

    # 1. Matrici: crosstalk lineare e non linearita' fuori diagonale.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    im = axes[0].imshow(np.log10(np.abs(A_norm)), cmap="viridis")
    axes[0].set_title(r"$\log_{10}(A_{ij}/A_{jj})$")
    fig.colorbar(im, ax=axes[0])
    if not np.all(np.isnan(B_self)):
        rel = B_self / A
        off = np.ma.masked_where(np.eye(N, dtype=bool), rel)
        lim = np.nanmax(np.abs(off))
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad("0.75")
        im = axes[1].imshow(off, cmap=cmap, vmin=-lim, vmax=lim)
        axes[1].set_title(r"$B_{i,jj}/A_{ij}$ fuori diagonale [1/mW]" + "\n"
                          + rf"(diagonale, in grigio: {np.mean(np.diag(rel)):.2e})",
                          fontsize=10)
        fig.colorbar(im, ax=axes[1])
    for ax in axes:
        ax.set_xlabel("heater j")
        ax.set_ylabel("guida i")
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "matrices.png"), dpi=150)
    plt.close(fig)

    # 2. Crosstalk in funzione della distanza.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    yerr = np.maximum([xt_mean - xt_lo, xt_hi - xt_mean], 0.0)
    ax.errorbar(d_um, xt_mean, yerr=yerr,
                fmt="o", capsize=3, label="simulazione (media, min-max)")
    d_fit = np.linspace(d_um[0], d_um[-1], 100)
    ax.plot(d_fit, np.exp(intercept + slope * d_fit), "--",
            label=rf"fit esponenziale, $L$ = {L_decay:.1f} $\mu$m")
    ax.set_yscale("log")
    ax.set_xlabel(r"distanza tra heater e guida [$\mu$m]")
    ax.set_ylabel(r"$A_{ij}/A_{jj}$")
    ax.set_title(rf"Crosstalk lineare ($2h_{{sub}}/\pi$ = "
                 rf"{2 * H_SUB_UM / math.pi:.1f} $\mu$m)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "crosstalk_vs_distance.png"), dpi=150)
    plt.close(fig)

    # 3. B(P): guida dell'heater e primo vicino, in due pannelli.
    if res["B_self_byP"]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        P_fine = np.linspace(0, max(P for (_, P) in res["B_self_byP"]), 100)
        for j in range(N):
            powers = sorted(P for (jj, P) in res["B_self_byP"] if jj == j)
            if len(powers) < 2:
                continue
            neighbour = j + 1 if j + 1 < N else j - 1
            for ax, i in [(axes[0], j), (axes[1], neighbour)]:
                vals = [res["B_self_byP"][(j, P)][i] / A[i, j] for P in powers]
                fit = sum(S[k][i, j] * P_fine**k for k in range(3)) / A[i, j]
                ax.plot(powers, vals, "o", color=colors[j])
                ax.plot(P_fine, fit, "-", color=colors[j], lw=1,
                        label=f"heater {j} -> WG {i}")
        axes[0].set_title("Sulla guida dell'heater")
        axes[1].set_title("Sulla guida vicina")
        for ax in axes:
            ax.set_xlabel("P [mW]")
            ax.set_ylabel(r"$B_{i,jj}(P)/A_{ij}$ [1/mW]")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0),
                                useOffset=False)
        fig.suptitle("B(P): punti = simulazioni, linee = fit "
                     "(pendenza = ordini superiori)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "B_vs_P.png"), dpi=150)
        plt.close(fig)

    print(f"\nGrafici e CSV salvati in {args.outdir}/")


if __name__ == "__main__":
    main()

