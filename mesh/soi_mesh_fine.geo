// ================================================================
// VARIANTE DI VERIFICA - MESH FINE (tutte le lc / 1.5)
//   lc_wg 0.50 -> 0.33, lc_heater 0.75 -> 0.50, lc_stack 4.0 -> 2.7,
//   lc_sub = lc_bulk 6.0 -> 4.0, MeshSizeMin 0.5 -> 0.33,
//   MeshSizeMax 6.0 -> 4.0. Geometria identica alla mesh di
//   riferimento. Serve a verificare la convergenza della
//   discretizzazione (fase 1). ~1.04M nodi, ~6.1M tetraedri: con
//   deal.II ogni processo MPI legge la mesh intera, quindi conviene
//   lanciarla con pochi processi (es. -n 2).
// ================================================================

// ================================================================
// FILE .geo - MESH 3D DI UNA SEZIONE SOI
// ================================================================
//
// Geometria in micrometri
// Kernel OpenCASCADE
// Compatibile con Gmsh
//
// STRATEGIA DI MESH: MASSIMA OMOGENEITA' DELLA SIZE, A COSTO DI
// PIU' NODI (simulazione stazionaria, un solo solve).
//
//   Waveguide                 ~ 0.50 um
//   Heater                    ~ 0.75 um
//   BOX / Cladding            ~ 4.0 um
//   Parte alta del substrato  ~ 4.0 -> 6.0 um
//   Bulk profondo Si          ~ 6.0 um
//
// IMPORTANTE:
// la mesh fine delle waveguide NON viene estesa indefinitamente
// nel substrato.
//
// MODIFICHE PER MASSIMIZZARE L'OMOGENEITA':
//   - lc_sub / lc_bulk / MeshSizeMax abbassati ulteriormente a
//     6.0 um (molto vicini a lc_stack=4.0 um): il salto massimo
//     nel dominio e' ora ~1.5x invece di 3x, la fonte principale
//     di disomogeneita' residua.
//   - Transizioni Distance/Threshold e Box.Thickness allargate
//     ancora di piu' (dist_max_heater=60, dist_max_wg=30,
//     stack_transition=20 um), cosi' il gradiente di size e'
//     spalmato su una distanza molto maggiore.
//   - Rimosso lo stadio di transizione intermedio (Field 7/8):
//     non serve piu', il salto diretto lc_heater/lc_wg -> lc_bulk
//     e' ora molto piu' contenuto (8-12x contro 33-50x) e la
//     transizione e' gia' ampia di suo.
//   - NESSUNA ottimizzazione post-mesh (rimossi Mesh.Smoothing e
//     Mesh.OptimizeNetgen): l'omogeneita' viene tutta dai field,
//     non da uno smoothing successivo.
//
// MODIFICHE PER LO STUDIO DEL CROSSTALK:
//   - Margine laterale portato da 15 a 50 um (margin), per
//     allontanare le pareti adiabatiche X_MIN/X_MAX (che agiscono
//     come specchi) dalle guide di bordo.
//   - Margine longitudinale parametrico z_ext = 50 um oltre
//     ciascuna estremita' dell'heater: L = L_heater + 2*z_ext,
//     z_heater = z_ext. Le guide coprono tutto il dominio, quindi
//     si estendono di z_ext oltre l'heater da entrambi i lati.
//   - Una Physical Volume per guida (WG_i, tag 20+i) al posto
//     dell'unico WAVEGUIDES_SI (tag 4), per poter integrare
//     Delta T separatamente su ciascuna guida.
// ================================================================

SetFactory("OpenCASCADE");


// ================================================================
// PARAMETRI CONFIGURABILI
// ================================================================


// ================================================================
// GUIDE D'ONDA IN SILICIO
// ================================================================

N_wg = 6;

pitch = 20.0;              // Passo tra le guide [um]

w_wg = 0.5;                // Larghezza waveguide [um]

h_wg = 0.22;               // Altezza waveguide [um]


// ================================================================
// HEATER METALLICI
// ================================================================

w_heater = 3.0;            // Larghezza heater [um]

h_heater = 0.15;           // Spessore heater [um]

L_heater = 150.0;          // Lunghezza heater [um]

z_ext = 50.0;              // Margine longitudinale: estensione di guide e
                           // dominio oltre ciascuna estremita' dell'heater [um]

z_heater = z_ext;          // Posizione iniziale lungo z [um]


// ================================================================
// STACK SOI
// ================================================================

h_sub = 50.0;              // Spessore substrato Si [um]

h_box = 2.0;               // Spessore BOX [um]

h_clad = 2.0;              // Spessore cladding [um]

margin = 50.0;             // Margine laterale [um]

L = L_heater + 2 * z_ext;  // Lunghezza totale [um] (= 250 um)


// ================================================================
// PARAMETRI MESH
// ================================================================


// ------------------------------------------------
// Waveguide
// ------------------------------------------------
//
// Nel modello originale avevi lc_wg=0.15 um,
// ma Mesh.MeshSizeMin=0.5 um impediva a Gmsh
// di scendere realmente a 0.15 um.
//
// Qui rendiamo esplicita la dimensione effettiva.
// ------------------------------------------------

lc_wg = 0.33;              // VERIFICA: era 0.50


// ------------------------------------------------
// Heater
// ------------------------------------------------

lc_heater = 0.50;          // VERIFICA: era 0.75


// ------------------------------------------------
// BOX + CLADDING + regione alta del substrato
// ------------------------------------------------

lc_stack = 2.7;            // VERIFICA: era 4.0


// ------------------------------------------------
// Bulk profondo del substrato
//
// Abbassato ulteriormente a 6.0 um, molto vicino a
// lc_stack=4.0: il rapporto max/min nel dominio e'
// ora ~1.5x, minimizzando il gradiente residuo.
// ------------------------------------------------

lc_sub = 4.0;              // VERIFICA: era 6.0

lc_bulk = 4.0;             // VERIFICA: era 6.0


// ================================================================
// DISTANCE FIELD - HEATER
// ================================================================
//
// Transizione allargata ulteriormente per massima omogeneita':
// il salto lc_heater -> lc_bulk (ora solo 8x invece di 33x)
// viene spalmato su una distanza ampia.
// ================================================================

dist_min_heater = 3.0;

dist_max_heater = 60.0;


// ================================================================
// DISTANCE FIELD - WAVEGUIDE
// ================================================================
//
// Era la transizione piu' brusca di tutte (4 um per un salto
// 50x, ora ridotto a 12x): allargata ulteriormente.
// ================================================================

dist_min_wg = 1.0;

dist_max_wg = 30.0;


// ================================================================
// PROFONDITÀ DELLA REGIONE RAFFINATA DELLO STACK
// ================================================================

stack_depth = 5.0;

stack_transition = 20.0;


// ================================================================
// TOLLERANZA PER SELEZIONI GEOMETRICHE (BoundingBox)
// ================================================================

eps = 1e-3;


// ================================================================
// CALCOLI DERIVATI
// ================================================================

W_total =
    (N_wg - 1) * pitch
    + w_wg
    + 2 * margin;


start_x =
    -((N_wg - 1) * pitch) / 2.0;


box_start_x =
    start_x
    - w_wg / 2.0
    - margin;


// ================================================================
// BOUNDING BOX GLOBALE
// ================================================================

y_bottom = -h_box - h_sub;

y_top = h_clad;

x_min = box_start_x;

x_max = box_start_x + W_total;

z_min = 0;

z_max = L;


// ================================================================
// COSTRUZIONE GEOMETRIA
// ================================================================


// ================================================================
// 1. SUBSTRATO SILICIO
// ================================================================

v_sub = newv;

Box(v_sub) = {
    box_start_x,
    -h_box - h_sub,
    0,
    W_total,
    h_sub,
    L
};


// ================================================================
// 2. BOX SiO2
// ================================================================

v_box = newv;

Box(v_box) = {
    box_start_x,
    -h_box,
    0,
    W_total,
    h_box,
    L
};


// ================================================================
// 3. CLADDING SUPERIORE SiO2
// ================================================================

v_clad = newv;

Box(v_clad) = {
    box_start_x,
    0,
    0,
    W_total,
    h_clad,
    L
};


// ================================================================
// 4. WAVEGUIDE IN SILICIO
// ================================================================

wg_vols[] = {};

For i In {0:N_wg-1}

    v_wg = newv;

    Box(v_wg) = {
        start_x + i * pitch - w_wg / 2.0,
        0,
        0,
        w_wg,
        h_wg,
        L
    };

    wg_vols[] += v_wg;

EndFor


// ================================================================
// 5. HEATER METALLICI
// ================================================================

heater_vols[] = {};

For i In {0:N_wg-1}

    v_h = newv;

    Box(v_h) = {
        start_x + i * pitch - w_heater / 2.0,
        h_clad,
        z_heater,
        w_heater,
        h_heater,
        L_heater
    };

    heater_vols[] += v_h;

EndFor


// ================================================================
// OPERAZIONE BOOLEAN DIFFERENCE
// ================================================================

clad_parts() =
    BooleanDifference {
        Volume{v_clad};
        Delete;
    } {
        Volume{wg_vols[]};
    };


// ================================================================
// BOOLEAN FRAGMENTS
// ================================================================

in_vols[] = {
    v_sub,
    v_box,
    clad_parts[],
    wg_vols[],
    heater_vols[]
};


n_sub = 1;

n_box = 1;

n_clad = #clad_parts[];

n_wg = #wg_vols[];

n_heater = #heater_vols[];


out[] =
    BooleanFragments {
        Volume{in_vols[]};
        Delete;
    } {};


// ================================================================
// RICOSTRUZIONE DEI GRUPPI DI VOLUME
// ================================================================

idx = 0;

sub_out[] =
    out[{idx : idx + n_sub - 1}];

idx += n_sub;

box_out[] =
    out[{idx : idx + n_box - 1}];

idx += n_box;

clad_out[] =
    out[{idx : idx + n_clad - 1}];

idx += n_clad;

wg_out[] =
    out[{idx : idx + n_wg - 1}];

idx += n_wg;

heater_out[] =
    out[{idx : idx + n_heater - 1}];

idx += n_heater;


// ================================================================
// PHYSICAL VOLUMES
// ================================================================

Physical Volume("SILICON_SUBSTRATE", 1) =
    sub_out[];

Physical Volume("BOX_SIO2", 2) =
    box_out[];

Physical Volume("CLADDING_SIO2", 3) =
    clad_out[];


// ------------------------------------------------
// WAVEGUIDE - UN Physical Volume PER GUIDA
// ------------------------------------------------
//
// Tag da 20 a 20+N_wg-1 (sostituisce il vecchio WAVEGUIDES_SI
// con tag 4), per integrare Delta T separatamente su ciascuna
// guida. Identificazione per BoundingBox sulla posizione di
// ciascuna guida, come per gli heater: l'ordine di wg_out[]
// non e' garantito da BooleanFragments.
// ------------------------------------------------

For i In {0:N_wg-1}

    x_wg_c = start_x + i * pitch;

    wg_i_vols() =
        Volume In BoundingBox{
            x_wg_c - w_wg / 2.0 - eps,
            -eps,
            z_min - eps,
            x_wg_c + w_wg / 2.0 + eps,
            h_wg + eps,
            z_max + eps
        };

    Physical Volume(Sprintf("WG_%g", i), 20 + i) =
        wg_i_vols();

EndFor


// ------------------------------------------------
// HEATER METALLICI - UN Physical Volume PER HEATER
// ------------------------------------------------
//
// Tag da 10 a 10+N_wg-1. Identificazione per BoundingBox sulla
// posizione originale di ciascun heater, non sull'ordine di
// heater_out[] (non garantito da BooleanFragments).
//
// NUOVO: accumuliamo anche heater_all_vols[], che serve sotto
// per calcolare la vera pelle esterna del dominio con
// CombinedBoundary.
// ------------------------------------------------

heater_all_vols[] = {};

For i In {0:N_wg-1}

    x_heater_c = start_x + i * pitch;

    heater_i_vols() =
        Volume In BoundingBox{
            x_heater_c - w_heater / 2.0 - eps,
            h_clad - eps,
            z_heater - eps,
            x_heater_c + w_heater / 2.0 + eps,
            h_clad + h_heater + eps,
            z_heater + L_heater + eps
        };

    Physical Volume(Sprintf("HEATER_%g", i), 10 + i) =
        heater_i_vols();

    heater_all_vols[] += heater_i_vols();

EndFor


// ================================================================
// PHYSICAL SURFACES
// ================================================================


// ------------------------------------------------
// BOTTOM SUBSTRATE, X_MIN, X_MAX, Z_MIN, Z_MAX
// ------------------------------------------------
//
// Invariati: nessun volume sporge su questi 5 lati, quindi la
// selezione per BoundingBox resta corretta e non ambigua.
// ------------------------------------------------

bottom_surf() =
    Surface In BoundingBox{
        x_min - eps,
        y_bottom - eps,
        z_min - eps,
        x_max + eps,
        y_bottom + eps,
        z_max + eps
    };

xmin_surf() =
    Surface In BoundingBox{
        x_min - eps,
        y_bottom - eps,
        z_min - eps,
        x_min + eps,
        y_top + eps,
        z_max + eps
    };

xmax_surf() =
    Surface In BoundingBox{
        x_max - eps,
        y_bottom - eps,
        z_min - eps,
        x_max + eps,
        y_top + eps,
        z_max + eps
    };

zmin_surf() =
    Surface In BoundingBox{
        x_min - eps,
        y_bottom - eps,
        z_min - eps,
        x_max + eps,
        y_top + eps,
        z_min + eps
    };

zmax_surf() =
    Surface In BoundingBox{
        x_min - eps,
        y_bottom - eps,
        z_max - eps,
        x_max + eps,
        y_top + eps,
        z_max + eps
    };


// ------------------------------------------------
// TOP_CLADDING - CALCOLO CORRETTO (FIX)
// ------------------------------------------------
//
// PROBLEMA ORIGINALE: una selezione per BoundingBox a y=h_clad
// cattura sia il vero bordo esterno del cladding, sia la faccia
// INTERNA condivisa con gli heater che vi poggiano sopra
// (dopo BooleanFragments). Assegnare boundary_id a una faccia
// interna manda in crash GridIn::read_msh in deal.II.
//
// FIX: calcoliamo prima la pelle ESTERNA VERA di tutto il
// dominio con CombinedBoundary (che cancella automaticamente le
// interfacce interne condivise tra volumi adiacenti, heater
// compresi), poi definiamo TOP_CLADDING come tutto cio' che sta
// in questa pelle esterna e non e' gia' stato classificato come
// uno dei 5 bordi sopra. Questo include correttamente sia il
// cladding esterno (dove non ci sono heater) sia le pareti e il
// tetto degli heater (che sporgono e sono quindi parte del bordo
// reale del dominio), escludendo pero' l'interfaccia interna.
// ------------------------------------------------

all_vols[] = {
    sub_out[],
    box_out[],
    clad_out[],
    wg_out[],
    heater_all_vols[]
};

skin_surf() = CombinedBoundary{ Volume{ all_vols[] }; };

already_classified[] = {
    bottom_surf(),
    xmin_surf(),
    xmax_surf(),
    zmin_surf(),
    zmax_surf()
};

top_surf_list[] = {};

For i In {0 : #skin_surf[]-1}

    s = skin_surf[i];

    is_classified = 0;

    For j In {0 : #already_classified[]-1}
        If (s == already_classified[j])
            is_classified = 1;
        EndIf
    EndFor

    If (is_classified == 0)
        top_surf_list[] += s;
    EndIf

EndFor


// ================================================================
// PHYSICAL SURFACES - ASSEGNAZIONE TAG
// ================================================================

Physical Surface("BOTTOM_SUBSTRATE", 1) =
    bottom_surf();

Physical Surface("TOP_CLADDING", 2) =
    top_surf_list[];

Physical Surface("X_MIN", 3) =
    xmin_surf();

Physical Surface("X_MAX", 4) =
    xmax_surf();

Physical Surface("Z_MIN", 5) =
    zmin_surf();

Physical Surface("Z_MAX", 6) =
    zmax_surf();


// ================================================================
// CAMPI DI MESH
// ================================================================
//
// Field 1/2: Distance/Threshold sull'heater, direttamente
//            lc_heater -> lc_bulk su una transizione ampia
//            (dist_max_heater = 60 um). Nessuno stadio
//            intermedio: con lc_bulk cosi' vicino a lc_stack
//            il salto e' gia' contenuto (~8x) e ben spalmato.
// Field 3/4: Distance/Threshold sulla waveguide, stesso
//            principio (lc_wg -> lc_bulk, dist_max_wg = 30 um).
// Field 5:   Box sullo stack (BOX/cladding/parte alta substrato).
// Field 6:   Min di tutti i field sopra = background field.
// ================================================================

heater_surf() =
    Boundary{
        Volume{heater_out[]};
    };

Field[1] = Distance;

Field[1].SurfacesList =
    {heater_surf()};

Field[1].NNodesByEdge = 50;

Field[2] = Threshold;

Field[2].InField = 1;

Field[2].SizeMin = lc_heater;

Field[2].SizeMax = lc_bulk;

Field[2].DistMin = dist_min_heater;

Field[2].DistMax = dist_max_heater;

wg_surf() =
    Boundary{
        Volume{wg_out[]};
    };

Field[3] = Distance;

Field[3].SurfacesList =
    {wg_surf()};

Field[3].NNodesByEdge = 50;

Field[4] = Threshold;

Field[4].InField = 3;

Field[4].SizeMin = lc_wg;

Field[4].SizeMax = lc_bulk;

Field[4].DistMin = dist_min_wg;

Field[4].DistMax = dist_max_wg;

Field[5] = Box;

Field[5].VIn = lc_stack;

Field[5].VOut = lc_bulk;

Field[5].XMin = x_min;

Field[5].XMax = x_max;

Field[5].YMin =
    -h_box - stack_depth;

Field[5].YMax =
    h_clad + h_heater;

Field[5].ZMin = z_min;

Field[5].ZMax = z_max;

Field[5].Thickness =
    stack_transition;

Field[6] = Min;

Field[6].FieldsList = {
    2,
    4,
    5
};

Background Field = 6;


// ================================================================
// DISABILITAZIONE SORGENTI AUTOMATICHE DI SIZE
// ================================================================

Mesh.MeshSizeExtendFromBoundary = 0;

Mesh.MeshSizeFromPoints = 0;

Mesh.MeshSizeFromCurvature = 0;


// ================================================================
// LIMITI GLOBALI
// ================================================================
//
// MeshSizeMax abbassato a 6.0 um in coerenza con lc_bulk, per
// eliminare qualunque salto oltre il cap globale.
// ================================================================

Mesh.MeshSizeMin = 0.33;   // VERIFICA: era 0.5 (senza, le lc piu piccole verrebbero ignorate)

Mesh.MeshSizeMax = 4.0;    // VERIFICA: era 6.0


// ================================================================
// PARAMETRI OPZIONALI DI MESH
// ================================================================

Mesh 3;


// ================================================================
// FINE FILE
//
// Nessuna ottimizzazione post-mesh (ne' Mesh.Smoothing ne'
// Mesh.OptimizeNetgen): l'omogeneita' della size e' interamente
// demandata ai field sopra.
// ================================================================

