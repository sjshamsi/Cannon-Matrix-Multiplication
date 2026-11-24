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

struct Options {
  size_t N{10};
  double tolerance{1e-8};
};

Options parseArgs(int argc, char** argv) {
  Options opts{};
  for (int i = 1; i < argc; i++) {
    if ((std::strcmp(argv[i], "--matrix-dim") == 0 || std::strcmp(argv[i], "-N") == 0) && i + 1 < argc) {
      opts.N = std::stoull(argv[++i]);
    } else if ((std::strcmp(argv[i], "--tolerance") == 0 || std::strcmp(argv[i], "-T") == 0) && i + 1 < argc) {
      opts.tolerance = std::stod(argv[++i]);
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      std::cout
        << "Usage: " << argv[0] << " [options]\n"
        << "  -N, --matrix-dim <N> Matrix order (default 10)\n"
        << "  -T, --tolerance <tol> Comparison tolerance (default 1e-8)\n";
      std::exit(0);
    }
  }

  return opts;
}


bool vectors_within_tolerance(const std::vector<double> &lhs, const std::vector<double> &rhs, double tolerance) {
  if (lhs.size() != rhs.size()) return false;
  return std::equal(lhs.begin(), lhs.end(), rhs.begin(), [tolerance](double a, double b) {
    return std::abs(a - b) <= tolerance;
  });
}


int main(int argc, char** argv) {
  Options opts = parseArgs(argc, argv);
  if (opts.N == 0) {
    std::cerr << "matrix-dim must be > 0\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::exit(EXIT_FAILURE);
  }

  if (opts.tolerance <= 0.0) {
    std::cerr << "tolerance must be > 0\n";
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


  MPITiming::Timekeeper tk;
  CartGrid matrix_props(opts.N, active, P);

  CanonM A(matrix_props, tk);
  CanonM B(matrix_props, tk);
  A.random_fill();
  B.random_fill();

  auto C = A.parallel_mmult(B);

  std::vector<double> Avec;
  std::vector<double> Bvec;
  std::vector<double> Cvec;
  std::vector<double> Ccomp;

  A.copy_matrix(Avec);
  B.copy_matrix(Bvec);
  C.copy_matrix(Cvec);

  int wrank_cartroot = A.grid.prank;
  if (wrank_cartroot == 0) {
    Ccomp.resize(opts.N * opts.N, 0);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      asINT(opts.N), asINT(opts.N), asINT(opts.N), 1.0,
      Avec.data(), asINT(opts.N),
      Bvec.data(), asINT(opts.N),
      0.0,
      Ccomp.data(), asINT(opts.N));

    const bool matches = vectors_within_tolerance(Cvec, Ccomp, opts.tolerance);
    std::cout << "Vector comparison result: " << (matches ? "PASS" : "FAIL")
              << " with tolerance " << opts.tolerance << "\n";

    const size_t ncorner_elems = std::min<size_t>(4, opts.N);
    const size_t r0{}, c0{};
    const size_t rf{r0 + ncorner_elems}, cf{c0 + ncorner_elems};

    std::cout << "\nMPI result top-right corner:\n";
    MatrixPrint::print_boxed_2D(Cvec, opts.N, opts.N, std::cout, true, MatrixPrint::DEFAULT_WIDTH, 6, r0, rf, c0, cf);

    std::cout << "\nLocal DGEMM result top-right corner:\n";
    MatrixPrint::print_boxed_2D(Ccomp, opts.N, opts.N, std::cout, true, MatrixPrint::DEFAULT_WIDTH, 6, r0, rf, c0, cf);
  }
  std::cout << "\n";

  MPI_Finalize();
  return 0;
}