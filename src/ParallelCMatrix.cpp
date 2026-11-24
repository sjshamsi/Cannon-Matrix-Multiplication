#include <iostream>
#include <cstring>

#include "ParallelCMatrix.hpp"


using matrix_t = double;
using TimeUnits = std::chrono::milliseconds;


struct Options {
  size_t N{10};
};

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
  MPI_Init(&argc, &argv);
  int rank, size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  Options opts = parseArgs(argc, argv);
  size_t mdim = opts.N;

  if (opts.N == 0) {
    if (rank == 0) std::cerr << "matrix-dim must be > 0\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPITiming::Timekeeper tk;

  MPITiming::Stopwatch total_sw(tk), constructor_sw(tk);
  total_sw.start("total");

  constructor_sw.start("constructor");
  CMatrixP<matrix_t> A(mdim, tk);
  CMatrixP<matrix_t> B(mdim, tk);
  A.random_fill();
  B.random_fill();
  constructor_sw.stop();

  CMatrixP<matrix_t> C = A.parallel_mmult(B);
  total_sw.stop();

  tk.packet_header<TimeUnits>();
  if (rank == 0) std::cout << "\n";
  tk.print_all_packets<TimeUnits>();
  
  MPI_Finalize();
  return 0;
}