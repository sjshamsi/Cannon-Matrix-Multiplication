#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <cblas.h>
#include <string_view>
#include <vector>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <type_traits>

#include "CartGrid.hpp"
#include "Print2DMatrix.hpp"
#include "MPI_Timer.hpp"

constexpr double RAND_LOWER_BOUND{-100}, RAND_UPPER_BOUND{100.0};
constexpr int DEFAULT_WIDTH{-1}, DEFAULT_PRECISION{3};


//////////////////////////////
// # Helpers
//////////////////////////////
static int asINT(const size_t v) {
if (v > static_cast<size_t>(std::numeric_limits<int>::max())) {
  std::cerr << "Error: size_t value " << v << " exceeds the maximum value of int ("
            << std::numeric_limits<int>::max() << ")." << std::endl;
  std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(v);
}
  
// static size_t asST(const int v) {
//   if (v < 0) {
//     std::cerr << "Error: int value " << v << " is negative and cannot be converted to size_t!";
//     std::exit(EXIT_FAILURE);
//   }
//   return static_cast<size_t>(v);
// }


//////////////////////////////
// # CLASS: CanonM
//////////////////////////////
class CanonM {
public:
  CartGrid &grid;
  MPITiming::Timekeeper &tk;
  std::vector<double> arr;

  CanonM(CartGrid &grid_properties, MPITiming::Timekeeper &timekeeper);
  void random_fill(unsigned base_seed = 42);
  CanonM parallel_mmult(const CanonM &B);

  // IO METHOD(s)
  void copy_matrix(std::vector<double> &external_buff);
  std::ostream & print(std::ostream &os = std::cout, std::string_view delim = ",", int width = DEFAULT_WIDTH, const int precision = DEFAULT_PRECISION);

}; // END CLASS CMatrixP


CanonM::CanonM(CartGrid &grid_properties, MPITiming::Timekeeper &timekeeper) : grid(grid_properties), tk(timekeeper) {
  arr.resize(grid.block * grid.block, 0.0);
}


void CanonM::random_fill(unsigned base_seed) {
  MPITiming::Stopwatch sw(tk);
  sw.start("randfn");

  #pragma omp parallel
  {
    const unsigned tid  = static_cast<unsigned>(omp_get_thread_num());
    const uint64_t seed = base_seed + tid + grid.arank;
    std::minstd_rand engine(static_cast<unsigned>(seed));
    
    #pragma omp for collapse(2) schedule(static)
    for (size_t i = 0; i < grid.ndims_real[0]; ++i)
      for (size_t j = 0; j < grid.ndims_real[1]; ++j)
        arr[i * grid.block + j] = RAND_LOWER_BOUND + (RAND_UPPER_BOUND - RAND_LOWER_BOUND) * (double(engine()) / engine.max());
  }

  sw.stop();
}


void CanonM::copy_matrix(std::vector<double> &external_buff) {
  if (grid.prank) {
    MPI_Datatype send_t;
    MPI_Type_vector(asINT(grid.ndims_real[0]), asINT(grid.ndims_real[1]), asINT(grid.block), MPI_DOUBLE, &send_t);
    MPI_Type_commit(&send_t);
    MPI_Send(arr.data(), 1, send_t, 0, grid.prank, grid.cart);
    MPI_Type_free(&send_t);
    return;
  }

  external_buff.resize(grid.N * grid.N);
  std::vector<MPI_Request> reqs(grid.asize - 1, MPI_REQUEST_NULL);
  std::vector<MPI_Datatype> recvtypes(grid.asize - 1, MPI_DATATYPE_NULL);
  std::array<int, 2> curr_coords = {0, 0};
  int curr_prank;
  size_t curr_rowoffset, curr_coloffset, curr_nrows_real, curr_ncols_real;

  for (int i = 1; i < grid.asize; i++) {
    int pr = i / grid.P;
    int pc = i % grid.P;

    curr_coords[0] = pr; curr_coords[1] = pc;
    MPI_Cart_rank(grid.cart, curr_coords.data(), &curr_prank);
    curr_rowoffset = pr * grid.block;
    curr_coloffset = pc * grid.block;
    curr_nrows_real = std::max(size_t{}, std::min(grid.block, grid.N - curr_rowoffset));
    curr_ncols_real = std::max(size_t{}, std::min(grid.block, grid.N - curr_coloffset));

    MPI_Type_vector(asINT(curr_nrows_real), asINT(curr_ncols_real), asINT(grid.N), MPI_DOUBLE, &recvtypes[i - 1]);
    MPI_Type_commit(&recvtypes[i - 1]);
    MPI_Irecv(external_buff.data() + (curr_rowoffset * grid.N) + curr_coloffset, 1, recvtypes[i - 1], curr_prank, curr_prank, grid.cart, &reqs[i - 1]);
  }

  for (size_t j = 0; j < grid.ndims_real[0]; j++)
    std::memcpy(external_buff.data() + ((grid.offsets[0] + j) * grid.N) + grid.offsets[1], arr.data() + j * grid.block, sizeof(double) * grid.ndims_real[1]);

  MPI_Waitall(asINT(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

  for(int i = 0; i < grid.asize - 1; ++i) {
    if(recvtypes[i] != MPI_DATATYPE_NULL) MPI_Type_free(&recvtypes[i]);
  }
}


std::ostream & CanonM::print(std::ostream &os, std::string_view delim, int width, const int precision) {
  std::vector<double> print_buff;
  copy_matrix(print_buff);
  if (grid.prank) return os;

  if (width < 0) width = 1 + MatrixPrint::optimal_width(print_buff, precision);
  MatrixPrint::print_2D(print_buff, grid.N, grid.N, os, delim, width, precision);
  return os;
}

CanonM CanonM::parallel_mmult(const CanonM &B) {
  MPITiming::Stopwatch sw0(tk);
  sw0.start("tot_mmultfn");

  if (grid.P != B.grid.P || grid.N_pad != B.grid.N_pad || grid.block != B.grid.block) {
    std::cerr << "parallel_mmult_opt: incompatible grids.\n";
    std::exit(EXIT_FAILURE);
  }

  const int blk = asINT(grid.block);
  const int cnt = blk * blk;
  CanonM C(grid, tk);

  std::vector<double> A_cur = arr, B_cur = B.arr;
  std::vector<double> A_nxt(cnt), B_nxt(cnt);

  // --- 1. Perform Initial Skew ---
  auto skew_once = [&](std::vector<double> &buf, int dim, int disp, int tag) {
    if (disp == 0) return;
    int src{MPI_PROC_NULL}, dst{MPI_PROC_NULL};
    MPI_Cart_shift(grid.cart, dim, disp, &src, &dst);
    MPI_Sendrecv_replace(buf.data(), cnt, MPI_DOUBLE, dst, tag, src, tag, grid.cart, MPI_STATUS_IGNORE);
  };
  skew_once(A_cur, 1, grid.coords[0], 11);
  skew_once(B_cur, 0, grid.coords[1], 12);

  // --- 2. Setup for Neighborhood Collective ---
  int sendcounts[4] = {0, 0, cnt, 0}; // Send A LEFT (neighbor 2)
  int recvcounts[4] = {0, 0, 0, cnt}; // Recv A from RIGHT (neighbor 3)
  MPI_Aint sdisp[4] = {0}, rdisp[4] = {0}; // Displacements are 0 since we use MPI_BOTTOM
  MPI_Datatype types[4] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};

  MPITiming::Stopwatch sw1(tk);
  sw1.start("mmultfn_body");

  for (int step = 0; step < grid.P; ++step) {
    MPI_Request reqs[2];

    // --- 3. Use Neighborhood Collectives for A and B shifts ---
    MPI_Ineighbor_alltoallw(A_cur.data(), sendcounts, sdisp, types,
                            A_nxt.data(), recvcounts, rdisp, types,
                            grid.cart, &reqs[0]);
    
    // Reconfigure for B shift: send UP (0), recv from DOWN (1)
    sendcounts[0] = cnt; sendcounts[2] = 0;
    recvcounts[1] = cnt; recvcounts[3] = 0;
    MPI_Ineighbor_alltoallw(B_cur.data(), sendcounts, sdisp, types,
                            B_nxt.data(), recvcounts, rdisp, types,
                            grid.cart, &reqs[1]);
    // Reset for next A shift
    sendcounts[0] = 0; sendcounts[2] = cnt;
    recvcounts[1] = 0; recvcounts[3] = cnt;

    // --- 4. Overlap computation ---
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                blk, blk, blk,
                1.0, A_cur.data(), blk,
                     B_cur.data(), blk,
                1.0, C.arr.data(), blk);

    // --- 5. Wait for communication ---
    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

    // --- 6. Swap buffers for next iteration ---
    A_cur.swap(A_nxt);
    B_cur.swap(B_nxt);
  }

  sw1.stop();
  sw0.stop();
  return C;
}