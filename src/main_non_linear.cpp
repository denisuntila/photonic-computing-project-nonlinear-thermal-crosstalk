#include "NonLinear.hpp"

#include <bitset>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
  void
  print_usage(const char *program)
  {
    std::cerr
      << "Uso: " << program
      << " [mesh.msh] [heater_mask] [P_mW] [--csv file.csv] [--no-vtu]\n"
      << "       [--h h_W_m2K] [--k-heater k_W_mK]\n"
      << "  mesh.msh     file di mesh Gmsh (default: ../mesh/soi_mesh.msh)\n"
      << "  heater_mask  heater accesi, bit i -> HEATER_i (default: 1)\n"
      << "               formati: decimale (5), esadecimale (0x5),\n"
      << "               binario (0b000101)\n"
      << "  P_mW         potenza di CIASCUN heater acceso in mW (default: 20)\n"
      << "  --csv file   CSV a cui aggiungere i risultati (default: results.csv)\n"
      << "  --no-vtu     non scrive l'output per ParaView\n"
      << "  --h          coefficiente h della Robin sul top [W/(m^2 K)]\n"
      << "               (default: " << NonLinear::h_conv_default << ")\n"
      << "  --k-heater   conducibilita' dell'heater [W/(m K)]\n"
      << "               (default: " << NonLinear::k_heater_default << ")\n"
      << "Esempio: mpirun -n 4 " << program
      << " ../mesh/soi_mesh.msh 0b000101 10 --no-vtu\n";
  }

  // Converte la maschera da stringa. Niente base 8 implicita:
  // "010" e' dieci, non otto.
  std::uint8_t
  parse_heater_mask(const std::string &arg)
  {
    int         base   = 10;
    std::size_t offset = 0;

    if (arg.size() > 2 && arg[0] == '0' && (arg[1] == 'b' || arg[1] == 'B'))
      {
        base   = 2;
        offset = 2;
      }
    else if (arg.size() > 2 && arg[0] == '0' &&
             (arg[1] == 'x' || arg[1] == 'X'))
      {
        base   = 16;
        offset = 2;
      }

    const std::string digits = arg.substr(offset);
    if (digits.empty() || digits[0] == '-' || digits[0] == '+')
      throw std::invalid_argument("maschera non valida: '" + arg + "'");

    std::size_t         parsed = 0;
    const unsigned long value  = std::stoul(digits, &parsed, base);

    if (parsed != digits.size())
      throw std::invalid_argument("maschera non valida: '" + arg + "'");

    const unsigned long max_mask = (1ul << MaterialId::n_heaters) - 1;

    if (value == 0)
      throw std::out_of_range("maschera nulla: nessun heater acceso");

    if (value > max_mask)
      throw std::out_of_range("maschera '" + arg + "' fuori range: con " +
                              std::to_string(MaterialId::n_heaters) +
                              " heater il massimo e' " +
                              std::to_string(max_mask));

    return static_cast<std::uint8_t>(value);
  }

  // Numero reale finito e non negativo (positivo se !allow_zero), per le
  // opzioni --h e --k-heater.
  double
  parse_positive(const std::string &arg,
                 const std::string &name,
                 const bool         allow_zero)
  {
    std::size_t  parsed = 0;
    const double value  = std::stod(arg, &parsed);

    if (parsed != arg.size() || !std::isfinite(value) || value < 0.0 ||
        (!allow_zero && value == 0.0))
      throw std::invalid_argument("valore non valido per " + name + ": '" +
                                  arg + "'");

    return value;
  }

  // Potenza in mW, strettamente positiva e finita.
  double
  parse_power_mW(const std::string &arg)
  {
    std::size_t  parsed = 0;
    const double value  = std::stod(arg, &parsed);

    if (parsed != arg.size() || !std::isfinite(value) || value <= 0.0)
      throw std::invalid_argument("potenza non valida: '" + arg +
                                  "' (serve un numero > 0 in mW)");

    return value;
  }
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  // Tutti i rank leggono gli stessi argomenti e prendono le stesse
  // decisioni, quindi possono uscire insieme senza deadlock; solo il
  // rank 0 stampa.
  const bool is_root = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;

  std::string        mesh_file_name = "../mesh/soi_mesh.msh";
  std::uint8_t       heater_mask    = 1;
  double             power_mW       = 20.0;
  std::string        csv_file_name  = "results.csv";
  bool               write_vtu      = true;
  double             h_conv         = NonLinear::h_conv_default;
  double             k_heater       = NonLinear::k_heater_default;
  const unsigned int degree         = 1;

  try
    {
      std::vector<std::string> positional;

      for (int a = 1; a < argc; ++a)
        {
          const std::string arg = argv[a];

          if (arg == "-h" || arg == "--help")
            {
              if (is_root)
                print_usage(argv[0]);
              return 0;
            }
          else if (arg == "--no-vtu")
            write_vtu = false;
          else if (arg == "--csv")
            {
              if (a + 1 >= argc)
                throw std::invalid_argument("--csv richiede un nome di file");
              csv_file_name = argv[++a];
            }
          else if (arg == "--h" || arg == "--k-heater")
            {
              if (a + 1 >= argc)
                throw std::invalid_argument(arg + " richiede un valore");
              if (arg == "--h")
                h_conv = parse_positive(argv[++a], "--h", true);
              else
                k_heater = parse_positive(argv[++a], "--k-heater", false);
            }
          else if (arg.rfind("--", 0) == 0)
            throw std::invalid_argument("opzione sconosciuta: '" + arg + "'");
          else
            positional.push_back(arg);
        }

      if (positional.size() > 3)
        throw std::invalid_argument("troppi argomenti posizionali");
      if (positional.size() >= 1)
        mesh_file_name = positional[0];
      if (positional.size() >= 2)
        heater_mask = parse_heater_mask(positional[1]);
      if (positional.size() >= 3)
        power_mW = parse_power_mW(positional[2]);
    }
  catch (const std::exception &e)
    {
      if (is_root)
        {
          std::cerr << "Errore: " << e.what() << "\n\n";
          print_usage(argv[0]);
        }
      return 1;
    }

  if (!std::ifstream(mesh_file_name))
    {
      if (is_root)
        std::cerr << "Errore: impossibile aprire la mesh '" << mesh_file_name
                  << "'" << std::endl;
      return 1;
    }

  if (is_root)
    std::cout << "Mesh        = " << mesh_file_name << "\n"
              << "Heater mask = 0b"
              << std::bitset<MaterialId::n_heaters>(heater_mask) << " ("
              << static_cast<unsigned int>(heater_mask) << ")\n"
              << "Potenza     = " << power_mW << " mW per heater acceso\n"
              << "CSV         = " << csv_file_name << "\n"
              << "Output VTU  = " << (write_vtu ? "si" : "no") << "\n"
              << "h           = " << h_conv << " W/(m^2 K)\n"
              << "k heater    = " << k_heater << " W/(m K)" << std::endl;

  NonLinear problem(mesh_file_name,
                    degree,
                    heater_mask,
                    power_mW * 1e-3,
                    csv_file_name,
                    write_vtu,
                    h_conv,
                    k_heater);

  problem.setup();
  problem.solve();

  return 0;
}



