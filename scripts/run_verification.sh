#!/usr/bin/env bash
# ================================================================
# FASE 1: VERIFICHE DEL MODELLO
# ================================================================
#
# 10 run di controllo, confrontati con i run della campagna
# (results.csv) da compare_verification.py:
#
#   margini      mesh con margin = z_ext = 100 um     -> verify_margin.csv
#   mesh fine    mesh con tutte le lc / 1.5            -> verify_fine.csv
#   convezione   mesh di riferimento, h = 0 e 100      -> verify_h0.csv,
#                                                         verify_h100.csv
#   heater       mesh di riferimento, k_TiN = 20       -> verify_kheater20.csv
#
# Prima di lanciare: generare le due mesh di verifica con Gmsh da
#   soi_mesh_margin100_geo.txt  e  soi_mesh_fine_geo.txt
# (rinominati in .geo).
#
# Uso:
#   LIN=./simulation NL=./non_linear ./run_verification.sh
# Variabili d'ambiente (con i default):
#   MESH_REF=../mesh/soi_mesh.msh
#   MESH_MARGIN=../mesh/soi_mesh_margin100.msh
#   MESH_FINE=../mesh/soi_mesh_fine.msh
#   NP=4        processi MPI
#   NP_FINE=2   processi MPI per la mesh fine (ogni processo legge
#               l'intera mesh: meno processi = meno memoria)
#   REF_CSV=results.csv   CSV della campagna, per il confronto finale
#
# I run gia' presenti nei CSV di verifica vengono saltati.
# ================================================================

set -euo pipefail

MESH_REF="${MESH_REF:-../mesh/soi_mesh.msh}"
MESH_MARGIN="${MESH_MARGIN:-../mesh/soi_mesh_margin100.msh}"
MESH_FINE="${MESH_FINE:-../mesh/soi_mesh_fine.msh}"
NP="${NP:-4}"
NP_FINE="${NP_FINE:-2}"
LIN="${LIN:-./heat_linear}"
NL="${NL:-./heat_nonlinear}"
REF_CSV="${REF_CSV:-results.csv}"
LOGDIR="${LOGDIR:-logs_verification}"
N_HEATERS=6

mkdir -p "$LOGDIR"

for m in "$MESH_REF" "$MESH_MARGIN" "$MESH_FINE"; do
    [[ -f "$m" ]] || { echo "Mesh mancante: $m"; exit 1; }
done

to_bin() {
    local m=$1 s="" b
    for (( b = N_HEATERS - 1; b >= 0; b-- )); do
        s+=$(( (m >> b) & 1 ))
    done
    echo "$s"
}

n_run=0
n_tot=10

run() {
    # run <np> <eseguibile> <mesh> <maschera> <P_mW> <csv> [opzioni...]
    local np="$1" exe="$2" mesh="$3" mask="$4" power="$5" csv="$6"
    shift 6
    local solver="nonlinear"
    [[ "$exe" == "$LIN" ]] && solver="linear"
    n_run=$(( n_run + 1 ))

    printf "[%2d/%2d] %-22s %-9s mask=%2d P=%3s mW %s ... " \
           "$n_run" "$n_tot" "$csv" "$solver" "$mask" "$power" "$*"

    if [[ -f "$csv" ]] &&
       grep -q "^${solver},0b$(to_bin "$mask"),${power}," "$csv"; then
        echo "gia' fatto"
        return
    fi

    local log="$LOGDIR/${csv%.csv}_${solver}_mask${mask}_P${power}mW.log"
    local t0=$SECONDS
    if mpirun -n "$np" "$exe" "$mesh" "$mask" "$power" \
              --csv "$csv" --no-vtu "$@" > "$log" 2>&1; then
        echo "ok ($(( SECONDS - t0 )) s)"
    else
        echo "ERRORE, vedi $log"
        exit 1
    fi
}

# Margini: heater di bordo (0) e interno (2); B sul bordo a 40 mW.
run "$NP" "$LIN" "$MESH_MARGIN" 1 20 verify_margin.csv
run "$NP" "$LIN" "$MESH_MARGIN" 4 20 verify_margin.csv
run "$NP" "$NL"  "$MESH_MARGIN" 1 40 verify_margin.csv

# Mesh fine: A su heater 0 e 2; B e terzo ordine su heater 2.
run "$NP_FINE" "$LIN" "$MESH_FINE" 1 20 verify_fine.csv
run "$NP_FINE" "$LIN" "$MESH_FINE" 4 20 verify_fine.csv
run "$NP_FINE" "$NL"  "$MESH_FINE" 4 20 verify_fine.csv
run "$NP_FINE" "$NL"  "$MESH_FINE" 4 40 verify_fine.csv

# Sensibilita' a h (riferimento: 10 W/(m^2 K)).
run "$NP" "$LIN" "$MESH_REF" 4 20 verify_h0.csv   --h 0
run "$NP" "$LIN" "$MESH_REF" 4 20 verify_h100.csv --h 100

# Sensibilita' a k del TiN (riferimento: 67.7 W/(m K)).
run "$NP" "$LIN" "$MESH_REF" 4 20 verify_kheater20.csv --k-heater 20

echo
python3 "$(dirname "$0")/compare_verification.py" "$REF_CSV" \
    verify_margin.csv verify_fine.csv verify_h0.csv verify_h100.csv \
    verify_kheater20.csv

