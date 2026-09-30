# Progetto Finale - Photonic Computing

Questo repository contiene il lavoro per il progetto finale del corso di **Photonic Computing** presso il **Politecnico di Milano**.

## Descrizione
Le simulazioni FEM (Finite Element Method) incluse in questo progetto sono state implementate e risolte utilizzando la libreria **deal.II**.

La toolchain e le dipendenze di supporto fanno riferimento alla repository ufficiale del corso:
- [pcafrica/mk](https://github.com/pcafrica/mk.git)

## Istruzioni per la Compilazione

Per compilare correttamente il codice, assicurati di seguire questi passaggi sul sistema di calcolo/ambiente configurato:

1. Carica il modulo necessario per la libreria `deal.II`:
   ```bash
   module load dealii
   ```

2. Spostati nella cartella di build (creala se non esiste):
   ```bash
   mkdir -p build && cd build
   ```

3. Esegui CMake per configurare il progetto:
   ```bash
   cmake ..
   ```

4. Compila il codice specificando il numero di core desiderati (sostituisci `<np>` con il numero di thread, es. `4` o `8`):
   ```bash
   make -j <np>
   ```
