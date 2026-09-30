#include "Heat.hpp"

#include <bitset>

void
Heat::setup()
{
  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);

    std::ifstream grid_in_file(mesh_file_name);
    grid_in.read_msh(grid_in_file);

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature          = std::make_unique<QGaussSimplex<dim>>(r + 1);
    quadrature_boundary = std::make_unique<QGaussSimplex<dim - 1>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
    pcout << "  Quadrature points per face = " << quadrature_boundary->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    system_matrix.reinit(sparsity);
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  }

  pcout << "-----------------------------------------------" << std::endl;

  compute_heater_volumes();
}

double
Heat::get_k(const types::material_id material) const
{
  // Leggi k(T) valutate una volta in T_amb (vedi Heat.hpp).
  const double k_Si   = k_Si_ref * std::pow(T_amb / T_ref, -alpha_Si);
  const double k_SiO2 = k_SiO2_ref * (1.0 + beta_SiO2 * (T_amb - T_ref));

  double k = 0.0;

  if (MaterialId::is_heater(material))
    k = k_metal;
  else if (MaterialId::is_waveguide(material))
    k = k_Si; // guide in Si (per ora stessa legge del substrato)
  else
    switch (material)
      {
        case MaterialId::silicon_substrate:
          k = k_Si;
          break;
        case MaterialId::box_sio2:
        case MaterialId::cladding_sio2:
          k = k_SiO2;
          break;
        default:
          AssertThrow(false,
                      ExcMessage("material_id sconosciuto: " +
                                 std::to_string(material)));
      }

  return k * um; // W/(m K) -> W/(um K)
}

double
Heat::get_Q(const types::material_id material) const
{
  if (MaterialId::is_heater(material) &&
      MaterialId::is_heater_active(active_heaters_mask,
                                   MaterialId::heater_index(material)))
    return P_active /
           heater_volume[MaterialId::heater_index(material)]; // W/um^3

  return 0.0;
}

void
Heat::assemble_system()
{
  pcout << "Assembling the linear system" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();
  const unsigned int n_q_face      = quadrature_boundary->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_JxW_values);

  FEFaceValues<dim> fe_face_values(*fe,
                                   *quadrature_boundary,
                                   update_values | update_quadrature_points |
                                     update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_matrix = 0.0;
  system_rhs    = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      // k e Q sono costanti a tratti: basta valutarli una volta per cella.
      const types::material_id material = cell->material_id();
      const double             k_loc    = get_k(material);
      const double             Q_loc    = get_Q(material);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              // Stiffness: int k grad(phi_j) . grad(phi_i).
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i, j) += k_loc *
                                     scalar_product(fe_values.shape_grad(j, q),
                                                    fe_values.shape_grad(i, q)) *
                                     fe_values.JxW(q);

              // Sorgente: int Q phi_i (solo negli heater attivi).
              if (Q_loc != 0.0)
                cell_rhs(i) +=
                  Q_loc * fe_values.shape_value(i, q) * fe_values.JxW(q);
            }
        }

      // Robin su TOP_CLADDING: -k dT/dn = h (T - T_amb)
      //   -> + int h T phi_i       nella matrice
      //   -> + int h T_amb phi_i   nel rhs
      if (cell->at_boundary())
        {
          for (const unsigned int f : cell->face_indices())
            {
              if (!cell->face(f)->at_boundary() ||
                  cell->face(f)->boundary_id() != BoundaryId::top_cladding)
                continue;

              fe_face_values.reinit(cell, f);

              for (unsigned int q = 0; q < n_q_face; ++q)
                {
                  const double h_loc =
                    function_h.value(fe_face_values.quadrature_point(q)) * um *
                    um; // W/(m^2 K) -> W/(um^2 K)

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        cell_matrix(i, j) += h_loc *
                                             fe_face_values.shape_value(j, q) *
                                             fe_face_values.shape_value(i, q) *
                                             fe_face_values.JxW(q);

                      cell_rhs(i) += h_loc * T_amb *
                                     fe_face_values.shape_value(i, q) *
                                     fe_face_values.JxW(q);
                    }
                }
            }
        }

      cell->get_dof_indices(dof_indices);

      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  // Dirichlet su BOTTOM_SUBSTRATE: T = T_amb (substrato termostatato).
  {
    std::map<types::global_dof_index, double> boundary_values;

    Functions::ConstantFunction<dim> dirichlet_function(T_amb);

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    boundary_functions[BoundaryId::bottom_substrate] = &dirichlet_function;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, solution_owned, system_rhs, false);
  }
}

void
Heat::solve_linear_system()
{
  pcout << "Solving the linear system" << std::endl;

  SolverControl solver_control(10000, 1e-10 * system_rhs.l2_norm());

  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

  // AMG: con ~1e5-1e6 DoF in 3D e contrasto k_Si / k_SiO2 ~ 100,
  // SSOR richiederebbe moltissime iterazioni.
  TrilinosWrappers::PreconditionAMG                 preconditioner;
  TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
  amg_data.elliptic              = true;
  amg_data.higher_order_elements = (r > 1);
  amg_data.smoother_sweeps       = 2;
  amg_data.aggregation_threshold = 0.02;
  preconditioner.initialize(system_matrix, amg_data);

  solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
  n_iterations = solver_control.last_step();
  pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;

  solution = solution_owned;
}

void
Heat::output() const
{
  pcout << "Writing output" << std::endl;

  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "T");

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  // material_id come campo di cella: comodo in ParaView per isolare
  // guide e heater con un Threshold.
  Vector<float> material(mesh.n_active_cells());
  for (const auto &cell : mesh.active_cell_iterators())
    if (cell->is_locally_owned())
      material[cell->active_cell_index()] = cell->material_id();
  data_out.add_data_vector(material, "material_id");

  data_out.build_patches();

  // Nome file con solver, maschera (bit 0 a destra) e potenza, cosi'
  // run diversi non si sovrascrivono, per esempio
  // output_linear_mask_000101_P20mW_0.pvtu.
  std::ostringstream name;
  name << "output_linear_mask_"
       << std::bitset<MaterialId::n_heaters>(active_heaters_mask) << "_P"
       << P_active * 1e3 << "mW";
  const std::string output_name = name.str();

  data_out.write_vtu_with_pvtu_record(
    "./", output_name, 0, MPI_COMM_WORLD, 3);
}

void
Heat::solve()
{
  pcout << "===============================================" << std::endl;

  assemble_system();
  solve_linear_system();
  postprocess();

  if (write_vtu)
    output();

  pcout << "===============================================" << std::endl;
}

void
Heat::compute_heater_volumes()
{
  FEValues<dim> fe_values(*fe, *quadrature, update_JxW_values);

  std::array<double, MaterialId::n_heaters> local_volume{};

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned() ||
          !MaterialId::is_heater(cell->material_id()))
        continue;

      fe_values.reinit(cell);

      const unsigned int index = MaterialId::heater_index(cell->material_id());
      for (unsigned int q = 0; q < quadrature->size(); ++q)
        local_volume[index] += fe_values.JxW(q);
    }

  pcout << "Heater volumes (integrati sulla mesh)" << std::endl;

  for (unsigned int i = 0; i < MaterialId::n_heaters; ++i)
    {
      heater_volume[i] = Utilities::MPI::sum(local_volume[i], MPI_COMM_WORLD);

      const bool active = MaterialId::is_heater_active(active_heaters_mask, i);

      pcout << "  HEATER_" << i << ": " << std::fixed << std::setprecision(4)
            << heater_volume[i] << " um^3" << (active ? "  [attivo]" : "")
            << std::endl;

      AssertThrow(!active || heater_volume[i] > 0.0,
                  ExcMessage("HEATER_" + std::to_string(i) +
                             " attivo ma assente nella mesh"));
    }

  pcout << "  Potenza per heater attivo = " << P_active * 1e3 << " mW"
        << std::defaultfloat << std::endl;
  pcout << "  h (TOP_CLADDING)          = "
        << function_h.value(Point<dim>()) << " W/(m^2 K)" << std::endl;
  pcout << "  k heater                  = " << k_metal << " W/(m K)"
        << std::endl;
}

void
Heat::postprocess() const
{
  constexpr unsigned int n_wg = MaterialId::n_waveguides;

  FEValues<dim> fe_values(*fe, *quadrature, update_values | update_JxW_values);

  const unsigned int  n_q = quadrature->size();
  std::vector<double> T_loc(n_q);

  // Per ogni guida: integrale di volume di Delta T [K um^3], volume
  // [um^3] ed estensione in z [um].
  std::array<double, n_wg> local_int{};
  std::array<double, n_wg> local_vol{};
  std::array<double, n_wg> local_zmin;
  std::array<double, n_wg> local_zmax;
  local_zmin.fill(std::numeric_limits<double>::max());
  local_zmax.fill(std::numeric_limits<double>::lowest());

  // Potenza effettivamente iniettata [W]: controllo di get_Q.
  double local_power = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      const types::material_id material = cell->material_id();
      const bool is_wg     = MaterialId::is_waveguide(material);
      const bool is_heater = MaterialId::is_heater(material);

      if (!is_wg && !is_heater)
        continue;

      fe_values.reinit(cell);

      if (is_heater)
        {
          const double Q_loc = get_Q(material);
          for (unsigned int q = 0; q < n_q; ++q)
            local_power += Q_loc * fe_values.JxW(q);
        }
      else
        {
          const unsigned int i = MaterialId::waveguide_index(material);

          fe_values.get_function_values(solution, T_loc);
          for (unsigned int q = 0; q < n_q; ++q)
            {
              local_int[i] += (T_loc[q] - T_amb) * fe_values.JxW(q);
              local_vol[i] += fe_values.JxW(q);
            }

          for (const unsigned int v : cell->vertex_indices())
            {
              const double z = cell->vertex(v)[2];
              local_zmin[i]  = std::min(local_zmin[i], z);
              local_zmax[i]  = std::max(local_zmax[i], z);
            }
        }
    }

  const double P_injected = Utilities::MPI::sum(local_power, MPI_COMM_WORLD);
  const double T_max      = solution_owned.max();

  // Integrale di linea lungo la guida, int Delta T dz [K um]: e' la
  // grandezza che determina lo sfasamento e non dipende da quanto la
  // guida si estende oltre l'heater (una volta catturato il campo).
  std::array<double, n_wg> line_int;
  std::array<double, n_wg> mean_dT;

  for (unsigned int i = 0; i < n_wg; ++i)
    {
      const double integral = Utilities::MPI::sum(local_int[i], MPI_COMM_WORLD);
      const double volume   = Utilities::MPI::sum(local_vol[i], MPI_COMM_WORLD);
      const double z_min    = Utilities::MPI::min(local_zmin[i], MPI_COMM_WORLD);
      const double z_max    = Utilities::MPI::max(local_zmax[i], MPI_COMM_WORLD);

      AssertThrow(volume > 0.0 && z_max > z_min,
                  ExcMessage("WG_" + std::to_string(i) +
                             " assente nella mesh"));

      const double area = volume / (z_max - z_min); // sezione [um^2]

      line_int[i] = integral / area;
      mean_dT[i]  = integral / volume;
    }

  pcout << "Postprocessing" << std::endl
        << "  Potenza iniettata = " << std::setprecision(6)
        << P_injected * 1e3 << " mW" << std::endl
        << "  T_max             = " << T_max << " K (Delta T = "
        << T_max - T_amb << " K)" << std::endl;

  for (unsigned int i = 0; i < n_wg; ++i)
    pcout << "  WG_" << i << ": int Delta T dz = " << std::setw(12)
          << line_int[i] << " K um,  <Delta T> = " << std::setw(12)
          << mean_dT[i] << " K" << std::endl;

  // Una riga nel CSV (solo rank 0). Intestazione se il file e' nuovo.
  if (mpi_rank == 0)
    {
      bool write_header = true;
      {
        std::ifstream in(csv_file_name);
        write_header =
          !in.good() || in.peek() == std::ifstream::traits_type::eof();
      }

      std::ofstream out(csv_file_name, std::ios::app);
      AssertThrow(out.good(),
                  ExcMessage("impossibile aprire " + csv_file_name));

      if (write_header)
        {
          out << "solver,mask,P_mW,P_injected_mW,iterations,T_max_K";
          for (unsigned int i = 0; i < n_wg; ++i)
            out << ",I_" << i << "_K_um";
          for (unsigned int i = 0; i < n_wg; ++i)
            out << ",dTmean_" << i << "_K";
          out << "\n";
        }

      out << std::setprecision(12) << "linear,0b"
          << std::bitset<MaterialId::n_heaters>(active_heaters_mask) << ","
          << P_active * 1e3 << "," << P_injected * 1e3 << "," << n_iterations
          << "," << T_max;
      for (unsigned int i = 0; i < n_wg; ++i)
        out << "," << line_int[i];
      for (unsigned int i = 0; i < n_wg; ++i)
        out << "," << mean_dT[i];
      out << "\n";
    }
}
