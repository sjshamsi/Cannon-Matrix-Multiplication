#include <cstring>

// Which Cannon variant to build (set by CMake; defaults to the non-blocking Isend/Irecv one)
#ifndef CANNON_HEADER
#define CANNON_HEADER "CanonM.hpp"
#endif
#include CANNON_HEADER
#include "Print2DMatrix.hpp"
#include "MPI_Timer.hpp"

using TimeUnits = std::chrono::milliseconds;


int nearest_squareroot(int n) {
  if (n <= 0) return 0;
  int p = static_cast<int>(std::floor(std::sqrt(static_cast<double>(n))));
  return std::max(1, p);
}

struct Options { size_t N{10}; };

Options parseArgs(int argc, char** argv) {
  Options opts{};
  for (int i = 1; i < argc; i++) {
    if ((std::strcmp(argv[i], "--matrix-dim") == 0 || std::strcmp(argv[i], "-N") == 0) && i + 1 < argc) {
      opts.N = std::stoull(argv[++i]);

    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      std::cout
        << "Usage: " << argv[0] << " [options]\n"
        << "  -N, --matrix-dim <N> Matrix order (default 10)\n";
        std::exit(0);
    }
  }
  return opts;
}


int main(int argc, char** argv) {
  Options opts = parseArgs(argc, argv);
  if (opts.N == 0) {
    std::cerr << "matrix-dim must be > 0\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::exit(EXIT_FAILURE);
  }

  MPI_Init(&argc, &argv);

  int wsize, wrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
  MPI_Comm_size(MPI_COMM_WORLD, &wsize);

  const int P{nearest_squareroot(wsize)};
  const int color = (wrank < P * P) ? 0 : MPI_UNDEFINED;
  MPI_Comm active{MPI_COMM_NULL};
  MPI_Comm_split(MPI_COMM_WORLD, color, wrank, &active);

  if (active == MPI_COMM_NULL) {
    MPI_Finalize();
    return 0;
  }

  MPITiming::Timekeeper tk(active);
  CartGrid matrix_props(opts.N, active, P);

  CanonM A(matrix_props, tk);
  CanonM B(matrix_props, tk);

  A.random_fill();
  B.random_fill();
  auto C = A.parallel_mmult(B);
    
  tk.packet_header<TimeUnits>();
  if (matrix_props.arank == 0) std::cout << "\n";
  tk.print_all_packets<TimeUnits>();
  
  MPI_Finalize();
  return 0;
}