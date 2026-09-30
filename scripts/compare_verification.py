#!/usr/bin/env python3
"""
Confronto dei run di verifica (fase 1) con i run della campagna.

Per ogni run di verifica cerca nel CSV di riferimento il run con stesso
solver, maschera e potenza, e confronta gli integrali I_i = int Delta T dz:

Le verifiche numeriche (margini, mesh fine) hanno un criterio di
superamento; quelle sui parametri fisici (h, k_TiN) sono sensibilita' da
riportare, senza criterio.

  diagonale      guida dell'heater acceso           criterio: < 0.5 %
  primo vicino   guide a 20 um                       criterio: < 2 %
  lontane        guide a >= 40 um (solo informativo)
  B_jj           auto-non-linearita' (se nel file ci sono sia il run
                 lineare sia quello non lineare dello stesso heater)
                                                     criterio: < 2 %

Uso:
    python3 compare_verification.py results.csv verify_*.csv
"""

import csv
import os
import sys

import numpy as np

N = 6
CRIT_DIAG = 0.5   # %
CRIT_NN = 2.0     # %
CRIT_B = 2.0      # %

# Due categorie:
#   numerica    (margini, mesh fine): errore del modello discreto, deve
#               stare entro i criteri
#   sensibilita' (h, k_TiN): effetto di un parametro fisico incerto, e'
#               un'informazione da riportare, non un superamento
DESCRIPTION = {
    "verify_margin": ("numerica", "margini 100 um invece di 50"),
    "verify_fine": ("numerica", "mesh fine (lc / 1.5)"),
    "verify_h0": ("sensibilita'", "h = 0 invece di 10 W/(m^2 K)"),
    "verify_h100": ("sensibilita'", "h = 100 invece di 10 W/(m^2 K)"),
    "verify_kheater20": ("sensibilita'", "k_TiN = 20 invece di 67.7 W/(m K)"),
}


def load(path):
    data = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["solver"], int(row["mask"], 0), float(row["P_mW"]))
            data[key] = np.array([float(row[f"I_{i}_K_um"]) for i in range(N)])
    return data


def single_heater(mask):
    on = [j for j in range(N) if (mask >> j) & 1]
    return on[0] if len(on) == 1 else None


def rel(a, b):
    """Differenza relativa percentuale di a rispetto a b."""
    return 100.0 * (a - b) / np.abs(b)


def B_diag(data, j, P):
    """B_jj(P) = (I_NL - A P) / P^2 sulla guida j, dal run lineare stesso file."""
    lin_keys = [k for k in data if k[0] == "linear" and k[1] == (1 << j)]
    nl = data.get(("nonlinear", 1 << j, P))
    if not lin_keys or nl is None:
        return None
    lin_key = lin_keys[-1]
    A_col = data[lin_key] / lin_key[2]
    B = (nl - A_col * P) / P**2
    return B


def flag(value, limit):
    return "ok" if abs(value) < limit else "OLTRE IL CRITERIO"


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ref = load(sys.argv[1])
    all_ok = True

    for path in sys.argv[2:]:
        if not os.path.exists(path):
            print(f"\n{path}: non trovato, saltato")
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        ver = load(path)
        kind, desc = DESCRIPTION.get(name, ("numerica", ""))
        numeric = kind == "numerica"
        print(f"\n=== {name}: {desc}  [verifica {kind}] ===")

        for (solver, mask, P), I_v in sorted(ver.items()):
            j = single_heater(mask)
            I_r = ref.get((solver, mask, P))
            label = f"{solver:9s} heater {j} P={P:g} mW"
            if j is None:
                print(f"  {label}: non e' un run a heater singolo, saltato")
                continue
            if I_r is None:
                print(f"  {label}: manca il run corrispondente in "
                      f"{sys.argv[1]}")
                continue

            d = rel(I_v, I_r)
            nn = [i for i in (j - 1, j + 1) if 0 <= i < N]
            far = [i for i in range(N) if abs(i - j) >= 2]
            d_nn = max(d[nn], key=abs)
            d_far = max(d[far], key=abs)
            if numeric:
                all_ok &= abs(d[j]) < CRIT_DIAG and abs(d_nn) < CRIT_NN
                print(f"  {label}:  diagonale {d[j]:+7.3f}% "
                      f"[{flag(d[j], CRIT_DIAG)}]"
                      f"   primo vicino {d_nn:+7.3f}% [{flag(d_nn, CRIT_NN)}]"
                      f"   lontane (max) {d_far:+7.3f}%")
            else:
                print(f"  {label}:  diagonale {d[j]:+7.3f}%"
                      f"   primo vicino {d_nn:+7.3f}%"
                      f"   lontane (max) {d_far:+7.3f}%")

            if solver == "nonlinear":
                B_v = B_diag(ver, j, P)
                B_r = B_diag(ref, j, P)
                if B_v is None or B_r is None:
                    print("    B_jj: manca il run lineare dello stesso "
                          "heater, non confrontabile")
                    continue
                dB = rel(B_v[j], B_r[j])
                if numeric:
                    all_ok &= abs(dB) < CRIT_B
                tag = f" [{flag(dB, CRIT_B)}]" if numeric else ""
                print(f"    B_jj(P={P:g}) = {B_v[j]:.5e} contro {B_r[j]:.5e}:"
                      f" {dB:+.3f}%{tag}")

    print("\n" + ("Verifiche numeriche: tutte entro i criteri." if all_ok else
                  "ATTENZIONE: almeno una verifica numerica e' oltre il "
                  "criterio (vedi sopra)."))
    print(f"Criteri (solo verifiche numeriche): diagonale < {CRIT_DIAG}%, "
          f"primo vicino < {CRIT_NN}%, B_jj < {CRIT_B}%.")
    print("Le sensibilita' (h, k_TiN) non hanno criterio: sono l'incertezza "
          "del modello dovuta a quel parametro, da riportare.")


if __name__ == "__main__":
    main()

