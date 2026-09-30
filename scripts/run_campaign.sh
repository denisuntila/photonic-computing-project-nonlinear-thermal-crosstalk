#!/usr/bin/env bash
# ================================================================
# CAMPAGNA DI RUN PER LA MATRICE DI CROSSTALK TERMICO
# ================================================================
#
# Tutti i run usano la STESSA mesh e scrivono una riga ciascuno nel
# CSV. Poi analyze_crosstalk.py ricava:
#   A_ij     matrice lineare                 (solver lineare)
#   B_i,jj   auto-non-linearita'             (non lineare, heater singoli)
#   B_i,jk   interazione tra heater j e k    (non lineare, coppie)
# e valida il modello su combinazioni con 3-6 heater accesi.
#
# Se il CSV esiste gia', i run presenti vengono saltati: si puo'
# estendere la campagna (nuove potenze) senza rifare quelli fatti.
#
# Uso:
#   ./run_campaign.sh
# Parametri modificabili con variabili d'ambiente, per esempio:
#   NP=8 MESH=../mesh/soi_mesh.msh CSV=results.csv ./run_campaign.sh
#
# ATTENZIONE: i run vanno eseguiti uno alla volta (come fa questo
# script): piu' processi che scrivono insieme sullo stesso CSV
# possono mescolare le righe.
# ================================================================

set -euo pipefail

MESH="${MESH:-../mesh/soi_mesh.msh}"
NP="${NP:-4}"
LIN="${LIN:-./heat_linear}"
NL="${NL:-./heat_nonlinear}"
CSV="${CSV:-results.csv}"
LOGDIR="${LOGDIR:-logs}"

N_HEATERS=6

# Potenze [mW] per heater acceso. Intervallo fino a ~2 P_pi (~44 mW),
# cioe' tutta la dinamica 0-2pi di un phase shifter. Limite superiore:
# a 50 mW T_max ~ 430 K, dentro la validita' delle leggi k(T) (250-500 K).
P_LIN=20                           # matrice lineare: qualsiasi valore va bene
P_SELF=(2.5 5 10 15 20 30 40 50)   # auto-non-linearita': fit di B(P)
P_PAIR=(10 20 30 40)               # interazione (sottoinsieme di P_SELF!)
P_ALL=(5 10 20 30 40)              # validazione: tutti gli heater accesi

# Validazione aggiuntiva: combinazioni con 3-5 heater accesi.
#   7  = 0b000111  tre adiacenti al bordo
#   21 = 0b010101  alternati
#   59 = 0b111011  tutti tranne HEATER_2
MASK_VAL=(7 21 59)
P_VAL=(20 40)

mkdir -p "$LOGDIR"

if [[ -e "$CSV" ]]; then
    echo "$CSV esiste gia': i run gia' presenti vengono saltati,"
    echo "i nuovi aggiunti in fondo."
fi

n_run=0
n_skip=0
n_tot=$(( N_HEATERS
        + N_HEATERS * ${#P_SELF[@]}
        + N_HEATERS * (N_HEATERS - 1) / 2 * ${#P_PAIR[@]}
        + ${#P_ALL[@]}
        + ${#MASK_VAL[@]} * ${#P_VAL[@]} ))

# Maschera decimale -> stringa binaria a N_HEATERS cifre (come nel CSV).
to_bin() {
    local m=$1 s="" b
    for (( b = N_HEATERS - 1; b >= 0; b-- )); do
        s+=$(( (m >> b) & 1 ))
    done
    echo "$s"
}

run() {
    # run <eseguibile> <maschera decimale> <P_mW> <etichetta>
    local exe="$1" mask="$2" power="$3" label="$4"
    local solver="nonlinear"
    [[ "$label" == linear ]] && solver="linear"
    n_run=$(( n_run + 1 ))

    # Ripresa: salta i run gia' presenti nel CSV.
    if [[ -f "$CSV" ]] &&
       grep -q "^${solver},0b$(to_bin "$mask"),${power}," "$CSV"; then
        n_skip=$(( n_skip + 1 ))
        printf "[%3d/%3d] %-10s mask=%2d P=%4s mW ... gia' fatto\n" \
               "$n_run" "$n_tot" "$label" "$mask" "$power"
        return
    fi

    local log="$LOGDIR/${label}_mask${mask}_P${power}mW.log"
    printf "[%3d/%3d] %-10s mask=%2d P=%4s mW ... " \
           "$n_run" "$n_tot" "$label" "$mask" "$power"
    local t0=$SECONDS
    if mpirun -n "$NP" "$exe" "$MESH" "$mask" "$power" \
              --csv "$CSV" --no-vtu > "$log" 2>&1; then
        echo "ok ($(( SECONDS - t0 )) s)"
    else
        echo "ERRORE, vedi $log"
        exit 1
    fi
}

echo "Campagna: $n_tot run, mesh $MESH, $NP processi MPI, CSV $CSV"

# 1. Matrice lineare A: un heater alla volta, solver lineare.
for (( j = 0; j < N_HEATERS; j++ )); do
    run "$LIN" $(( 1 << j )) "$P_LIN" linear
done

# 2. Auto-non-linearita' B_i,jj: un heater alla volta, piu' potenze.
for P in "${P_SELF[@]}"; do
    for (( j = 0; j < N_HEATERS; j++ )); do
        run "$NL" $(( 1 << j )) "$P" self
    done
done

# 3. Interazione B_i,jk: tutte le coppie j < k, stesse potenze.
for P in "${P_PAIR[@]}"; do
    for (( j = 0; j < N_HEATERS; j++ )); do
        for (( k = j + 1; k < N_HEATERS; k++ )); do
            run "$NL" $(( (1 << j) | (1 << k) )) "$P" pair
        done
    done
done

# 4. Validazione: tutti gli heater accesi.
for P in "${P_ALL[@]}"; do
    run "$NL" $(( (1 << N_HEATERS) - 1 )) "$P" all
done

# 5. Validazione: altre combinazioni con 3-5 heater accesi.
for P in "${P_VAL[@]}"; do
    for mask in "${MASK_VAL[@]}"; do
        run "$NL" "$mask" "$P" validation
    done
done

echo "Fatto: $(( n_run - n_skip )) run eseguiti, $n_skip gia' presenti."
echo "Ora: python3 analyze_crosstalk.py $CSV"

