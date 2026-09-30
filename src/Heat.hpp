#ifndef HEAT_HPP
#define HEAT_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <fstream>
#include <iostream>

namespace MaterialId
{
  constexpr dealii::types::material_id silicon_substrate = 1;
  constexpr dealii::types::material_id box_sio2           = 2;
  constexpr dealii::types::material_id cladding_sio2      = 3;

  // Un material_id per ciascun heater (10..15 per N_wg = 6),
  // coerente con Physical Volume(Sprintf("HEATER_%g", i), 10 + i).
  constexpr dealii::types::material_id heater_base = 10;
  constexpr unsigned int               n_heaters   = 6;

  inline bool
  is_heater(const dealii::types::material_id material)
  {
    return material >= heater_base && material < heater_base + n_heaters;
  }

  inline unsigned int
  heater_index(const dealii::types::material_id material)
  {
    return static_cast<unsigned int>(material - heater_base);
  }

  inline bool
  is_heater_active(const std::uint8_t mask, const unsigned int index)
  {
    return (mask & (static_cast<std::uint8_t>(1) << index)) != 0;
  }

  // Una Physical Volume per guida (WG_i, tag 20..25 per N_wg = 6),
  // coerente con Physical Volume(Sprintf("WG_%g", i), 20 + i).
  constexpr dealii::types::material_id waveguide_base = 20;
  constexpr unsigned int               n_waveguides   = 6;

  inline bool
  is_waveguide(const dealii::types::material_id material)
  {
    return material >= waveguide_base &&
           material < waveguide_base + n_waveguides;
  }

  // Indice 0-based della guida. Chiamare solo se is_waveguide() e' vero.
  inline unsigned int
  waveguide_index(const dealii::types::material_id material)
  {
    return static_cast<unsigned int>(material - waveguide_base);
  }
}

// Tag delle Physical Surface nel .geo (read_msh li legge come
// boundary_id), allineati al .geo: 1 = BOTTOM_SUBSTRATE, 2 = TOP_CLADDING.
// Le facce senza Physical Surface (boundary_id = 0) restano
// adiabatiche.
namespace BoundaryId
{
  constexpr dealii::types::boundary_id top_cladding     = 2; // Robin
  constexpr dealii::types::boundary_id bottom_substrate = 1; // Dirichlet T = T_amb
}

using namespace dealii;

// Problema di conduzione termica STAZIONARIO LINEARE (k costante
// per materiale):
//
//   -div(k grad T) = Q                 in Omega
//   T = T_amb                          su BOTTOM_SUBSTRATE
//   -k dT/dn = h (T - T_amb)           su TOP_CLADDING
//   -k dT/dn = 0                       altrove
class Heat
{
public:
  static constexpr unsigned int dim = 3;

  // Valori di default dei parametri modificabili da riga di comando.
  //   h:     convezione naturale verso l'aria sul TOP_CLADDING
  //          [W/(m^2 K)], da Jacques et al., Opt. Express 27, 10456 (2019).
  //   k_TiN: conducibilita' dell'heater in TiN [W/(m K)], da
  //          arXiv 2412.03951, Tab. I (film sottili: ~20-70).
  static constexpr double h_conv_default   = 10.0;
  static constexpr double k_heater_default = 67.7;

  // Coefficiente h della Robin BC su TOP_CLADDING [W/(m^2 K)], uniforme.
  class FunctionH : public Function<dim>
  {
  public:
    explicit FunctionH(const double h_ = h_conv_default)
      : h(h_)
    {}

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return h;
    }

  private:
    const double h;
  };

  // active_heaters_mask_: bit i acceso -> HEATER_i attivo.
  // power_W_:            potenza di CIASCUN heater attivo [W].
  // csv_file_name_:      file CSV a cui aggiungere una riga di risultati.
  // write_vtu_:          se false non scrive l'output per ParaView.
  // h_conv_:             coefficiente h della Robin [W/(m^2 K)].
  // k_heater_:           conducibilita' dell'heater [W/(m K)].
  Heat(const std::string  &mesh_file_name_,
       const unsigned int &r_,
       const std::uint8_t  active_heaters_mask_,
       const double        power_W_,
       const std::string  &csv_file_name_,
       const bool          write_vtu_,
       const double        h_conv_   = h_conv_default,
       const double        k_heater_ = k_heater_default)
    : active_heaters_mask(active_heaters_mask_)
    , P_active(power_W_)
    , csv_file_name(csv_file_name_)
    , write_vtu(write_vtu_)
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , function_h(h_conv_)
    , mesh_file_name(mesh_file_name_)
    , r(r_)
    , mesh(MPI_COMM_WORLD)
    , k_metal(k_heater_)
  {}

  void
  setup();

  void
  solve();

protected:
  // Assembla matrice (stiffness + Robin) e rhs, poi applica la
  // Dirichlet su BOTTOM_SUBSTRATE.
  void
  assemble_system();

  void
  solve_linear_system();

  void
  output() const;

  // Volume di ciascun heater, integrato sulla mesh.
  void
  compute_heater_volumes();

  // Integrali di Delta T sulle guide, potenza iniettata e T_max:
  // stampa a schermo e aggiunge una riga al CSV.
  void
  postprocess() const;

  // Conducibilita' termica k, convertita in W/(um K).
  double
  get_k(const types::material_id material) const;

  // Densita' di potenza Q = P / V_heater in W/um^3; zero fuori dagli
  // heater attivi.
  double
  get_Q(const types::material_id material) const;

  // Selezione degli heater attivi. ///////////////////////////////////////////

  // bit i acceso -> HEATER_i attivo. Passata dal costruttore.
  // Dichiarata prima di mpi_size: l'ordine di inizializzazione
  // segue quello di dichiarazione.
  const std::uint8_t active_heaters_mask;

  // Potenza di ciascun heater attivo [W]. La densita' Q = P / V_heater
  // usa il volume dell'heater integrato sulla mesh, cosi' la potenza
  // iniettata e' esattamente P qualunque sia la discretizzazione.
  const double P_active;

  // File CSV dei risultati e flag per l'output VTU.
  const std::string csv_file_name;
  const bool        write_vtu;

  // Volume di ciascun heater [um^3], calcolato in setup().
  std::array<double, MaterialId::n_heaters> heater_volume{};

  // Iterazioni del solver (CG nel lineare, Newton nel non lineare),
  // riportate nel CSV.
  unsigned int n_iterations = 0;

  // MPI parallel. /////////////////////////////////////////////////////////////

  const unsigned int mpi_size;
  const unsigned int mpi_rank;
  ConditionalOStream pcout;

  // Problem definition. ///////////////////////////////////////////////////////

  FunctionH function_h;

  // Temperatura ambiente [K]: T esterna nella Robin e valore della
  // Dirichlet su BOTTOM_SUBSTRATE.
  double T_amb = 293.15;

  // Discretization. ///////////////////////////////////////////////////////////

  const std::string  mesh_file_name;
  const unsigned int r;

  parallel::fullydistributed::Triangulation<dim> mesh;

  std::unique_ptr<FiniteElement<dim>>  fe;
  std::unique_ptr<Quadrature<dim>>     quadrature;
  std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;

  DoFHandler<dim> dof_handler;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  TrilinosWrappers::SparseMatrix system_matrix;
  TrilinosWrappers::MPI::Vector  system_rhs;

  // Soluzione senza ghost (per il solver).
  TrilinosWrappers::MPI::Vector solution_owned;

  // Soluzione con ghost (per l'output).
  TrilinosWrappers::MPI::Vector solution;

  // Material properties (bulk, SI). ///////////////////////////////////////////

  // Conducibilita' costanti, valutate in T_amb con le STESSE leggi k(T)
  // del solver non lineare (NonLinear.hpp). Cosi' questo problema e'
  // esattamente la linearizzazione del non lineare attorno a T_amb, e
  // la matrice A che se ne ricava e' la pendenza iniziale dI/dP del
  // modello non lineare: condizione necessaria per estrarre i termini
  // di secondo ordine come (I_NL - A P) / P^2.
  static constexpr double T_ref     = 300.0;  // K, riferimento delle leggi
  static constexpr double k_Si_ref  = 148.0;  // W/(m K) a T_ref, Si bulk
  static constexpr double alpha_Si  = 1.3;    // k_Si(T) = k_Si_ref (T/T_ref)^(-alpha_Si)
  static constexpr double k_SiO2_ref = 1.4;   // W/(m K) a T_ref
  static constexpr double beta_SiO2 = 1.0e-3; // 1/K, k_SiO2(T) = k_SiO2_ref (1 + beta (T - T_ref))
  // Conducibilita' dell'heater (TiN) [W/(m K)], a T_ref: default
  // k_heater_default, modificabile con --k-heater.
  const double k_metal;

  // Coordinate della mesh in micron: 1 um = 1e-6 m.
  //   k [W/(m K)] -> * um,  Q [W/m^3] -> * um^3,  h [W/(m^2 K)] -> * um^2
  static constexpr double um = 1e-6;
};

#endif

