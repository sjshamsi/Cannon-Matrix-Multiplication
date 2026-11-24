#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "ParallelCMatrix.hpp"

using matrix_t = double;


struct Options {
  size_t N{10};
  matrix_t tolerance{1e-8};
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


bool vectors_within_tolerance(const std::vector<matrix_t> &lhs, const std::vector<matrix_t> &rhs, matrix_t tolerance) {
  if (lhs.size() != rhs.size()) return false;
  return std::equal(lhs.begin(), lhs.end(), rhs.begin(), [tolerance](matrix_t a, matrix_t b) {
    return std::abs(a - b) <= tolerance;
  });
}


int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  Options opts = parseArgs(argc, argv);
  const size_t mdim = opts.N;

  if (mdim == 0) {
    if (rank == 0) std::cerr << "matrix-dim must be > 0\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (opts.tolerance <= 0.0) {
    if (rank == 0) std::cerr << "tolerance must be > 0\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPITiming::Timekeeper tk;

  CMatrixP<matrix_t> A(mdim, tk);
  CMatrixP<matrix_t> B(mdim, tk);
  A.random_fill();
  B.random_fill();

  CMatrixP<matrix_t> C = A.parallel_mmult(B);
  std::vector<matrix_t> A_full, B_full, mpiC_full;
  A.copy_matrix(A_full);
  B.copy_matrix(B_full);
  C.copy_matrix(mpiC_full);


  if (rank == DEFAULT_ROOT) {
    const size_t total_elems = mdim * mdim;
    if (A_full.size() != total_elems || B_full.size() != total_elems || mpiC_full.size() != total_elems) {
      std::cerr << "Root rank did not receive full matrix data.\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<matrix_t> local_C(total_elems, matrix_t{0});
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      asINT(mdim), asINT(mdim), asINT(mdim),
      1.0,
      A_full.data(), asINT(mdim),
      B_full.data(), asINT(mdim),
      0.0,
      local_C.data(), asINT(mdim));

    const bool matches = vectors_within_tolerance(mpiC_full, local_C, opts.tolerance);
    std::cout << "Vector comparison result: " << (matches ? "PASS" : "FAIL")
              << " with tolerance " << opts.tolerance << "\n";

    const size_t ncorner_elems = std::min<size_t>(4, mdim);
    const size_t r0{}, c0{};
    const size_t rf{r0 + ncorner_elems}, cf{c0 + ncorner_elems};

    std::cout << "\nMPI result top-right corner:\n";
    MatrixPrint::print_boxed_2D(mpiC_full, mdim, mdim, std::cout, true, MatrixPrint::DEFAULT_WIDTH, 6, r0, rf, c0, cf);

    std::cout << "\nLocal DGEMM result top-right corner:\n";
    MatrixPrint::print_boxed_2D(local_C, mdim, mdim, std::cout, true, MatrixPrint::DEFAULT_WIDTH, 6, r0, rf, c0, cf);
  }
  
  MPI_Finalize();
  return 0;
}