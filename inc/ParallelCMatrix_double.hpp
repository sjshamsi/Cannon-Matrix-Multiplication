#pragma once

#include <mpi.h>
#include <cblas.h>
#include <iostream>
#include <iomanip>
#include <string_view>
#include <vector>
#include <random>
#include <algorithm>

#include "Print2DMatrix.hpp"
#include "MPI_Timer.hpp"


constexpr double RAND_LOWER_BOUND{-100}, RAND_UPPER_BOUND{100.0};
constexpr int DEFAULT_ROOT{}, DEFAULT_WIDTH{}, DEFAULT_PRECISION{3};


static int asINT(const size_t v) {
  if (v > static_cast<size_t>(std::numeric_limits<int>::max())) {
    std::cerr << "Error: size_t value " << v << " exceeds the maximum value of int ("
              << std::numeric_limits<int>::max() << ")." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(v);
}

static size_t asST(const int v) {
  if (v < 0) {
    std::cerr << "Error: int value " << v << " is negative and cannot be converted to size_t!";
    std::exit(EXIT_FAILURE);
  }
  return static_cast<size_t>(v);
}


//////////////////////////////
// # CLASS: CMatrixP
//////////////////////////////
class CMatrixP { // BEGIN CLASS CMatrixP
public:
// private:
  size_t N;
  MPITiming::Timekeeper &tk;
  MPI_Comm comm;
  int root, rank, csize;
  size_t rows_remaining, nrows, offset;

  std::vector<size_t> all_nrows;
  std::vector<size_t> all_offsets;
  void calc_nrows_offsets();

  std::vector<double> arr;

  std::vector<double> mmult_Barr;
  std::vector<int> recv_counts, displs;
  void gather_Ith_Bcols(const int I, std::vector<double> &Barr);

// public:
  // CONSTRUCTOR(s)
  explicit CMatrixP(const size_t order, MPITiming::Timekeeper &timekeeper, MPI_Comm mpi_comm = MPI_COMM_WORLD,
                    const int root_rank = DEFAULT_ROOT);
  
  CMatrixP(const CMatrixP &B);

  // MISC. METHOD(s)
  void random_fill();
  void identity_fill() { for (size_t i = 0; i < nrows; i++) arr[(N * i) + i + offset] = 1.0; }
  void zero_out() { std::fill(arr.begin(), arr.end(), 0.0); }
  CMatrixP parallel_mmult(CMatrixP &B);
  
  // IO METHOD(s)
  std::ostream & print_boxed(std::ostream &os = std::cout, int width = DEFAULT_WIDTH, const int precision = DEFAULT_PRECISION) const;
  std::ostream & print(std::ostream &os = std::cout, std::string_view delim = ", ", int width = DEFAULT_WIDTH, const int precision = DEFAULT_PRECISION) const;
  void copy_matrix(std::vector<double> &buffer);
}; // END CLASS CMatrixP


//////////////////////////////
// # CMatrixP CONSTRUCTOR(s)
//////////////////////////////
CMatrixP::CMatrixP(const size_t order, MPITiming::Timekeeper &timekeeper, MPI_Comm mpi_comm, const int root_rank)
  : N(order), tk(timekeeper), comm(mpi_comm), root(root_rank) {
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &csize);

  if (N < 1) {
    if (rank == root) std::cerr << "Invalid Matrix Dimensions. Matrix dims must be >= 1\n";
    std::exit(EXIT_FAILURE);
  }
  rows_remaining = N % csize;
  calc_nrows_offsets();
  nrows = all_nrows[rank];
  offset = all_offsets[rank];

  recv_counts.assign(csize, 0);
  displs.assign(csize, 0);
  arr.resize(nrows * N);
}


CMatrixP::CMatrixP(const CMatrixP &B)
  : N(B.N),
    tk(B.tk),
    comm(B.comm),
    root(B.root),
    rank(B.rank),
    csize(B.csize),
    rows_remaining(B.rows_remaining),
    nrows(B.nrows),
    offset(B.offset),
    all_nrows(B.all_nrows),
    all_offsets(B.all_offsets),
    arr(B.arr.size()),
    recv_counts(B.recv_counts.size()),
    displs(B.displs.size()) {}

  
//////////////////////////////
// # CMatrixP MMULT METHODS
//////////////////////////////
void CMatrixP::gather_Ith_Bcols(int I, std::vector<double> &Barr) {
  MPITiming::Stopwatch sw(tk);
  sw.start("gatherfn");

  displs[root] = 0;
  recv_counts[root] = asINT(all_nrows[root] * all_nrows[I]);
  for (int j = 1; j < csize; j++) {
    recv_counts[j] = asINT(all_nrows[j] * all_nrows[I]);
    displs[j] = displs[j - 1] + recv_counts[j - 1];
  }
  
  MPI_Datatype send_t;
  MPI_Type_vector(asINT(nrows), asINT(all_nrows[I]), asINT(N), MPI_DOUBLE, &send_t);
  MPI_Type_commit(&send_t);
  
  MPI_Allgatherv(arr.data() + all_offsets[I], 1, send_t, Barr.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, comm);
  MPI_Type_free(&send_t);
  sw.stop();
}


CMatrixP CMatrixP::parallel_mmult(CMatrixP &B) {
  if (N != B.N) {
    if (rank == root) std::cerr << "Matrix sizes incompatible for multiplication!\n";
    std::exit(EXIT_FAILURE);
  }

  MPITiming::Stopwatch sw1(tk), sw2(tk);

  sw1.start("cpy_constructor");
  CMatrixP C(*this);
  sw1.stop();

  mmult_Barr.resize(all_nrows[root] * N);

  for (int I = 0; I < csize; I++) {
    B.gather_Ith_Bcols(I, mmult_Barr);

    sw2.start("dgemm");
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      asINT(nrows), asINT(all_nrows[I]), asINT(N),
      1.0,
      arr.data(), asINT(N),
      mmult_Barr.data(), asINT(all_nrows[I]),
      0.0,
      C.arr.data() + all_offsets[I], asINT(N));
    sw2.stop();
  }
  
  return C;
}


//////////////////////////////
// # CMatrixP IO METHOS(s)
//////////////////////////////
std::ostream & CMatrixP::print_boxed(std::ostream &os, int width, int precision) const {
  if (rank) {
    MPI_Send(arr.data(), asINT(arr.size()), MPI_DOUBLE, root, rank, comm);
    return os;
  }

  MPI_Request req;
  width = (width == DEFAULT_WIDTH) ? MatrixPrint::optimal_width(arr, precision) : width;

  std::vector<double> buff1(arr.size()), buff2 = arr;
  for (int i = 1; i < csize; i++) {
    MPI_Irecv(buff1.data(), asINT(all_nrows[i] * N), MPI_DOUBLE, i, i, comm, &req);
    MatrixPrint::print_boxed_2D(buff2, all_nrows[i - 1], N, os, false, width, precision);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    std::swap(buff1, buff2);
  }

  MatrixPrint::print_boxed_2D(buff2, all_nrows[csize - 1], N, os, true, width, precision);
  return os;
}


std::ostream & CMatrixP::print(std::ostream &os, std::string_view delim, int width, const int precision) const {
  if (rank) {
    MPI_Send(arr.data(), asINT(arr.size()), MPI_DOUBLE, root, rank, comm);
    return os;
  }

  MPI_Request req;
  std::vector<double> buff1(arr.size()), buff2 = arr;
  for (int i = 1; i < csize; i++) {
    MPI_Irecv(buff1.data(), asINT(all_nrows[i] * N), MPI_DOUBLE, i, i, comm, &req);
    MatrixPrint::print_2D(buff2, all_nrows[i - 1], N, os, delim, width, precision);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    std::swap(buff1, buff2);
  }

  MatrixPrint::print_2D(buff2, all_nrows[csize - 1], N, os, delim, width, precision);
  return os;
}


void CMatrixP::copy_matrix(std::vector<double> &external_buff) {
  if (rank == root) external_buff.resize(N * N);

  displs[root] = 0;
  recv_counts[root] = asINT(all_nrows[root] * N);
  for (int i = 1; i < csize; i++) {
    recv_counts[i] = asINT(all_nrows[i] * N);
    displs[i] = displs[i - 1] + recv_counts[i - 1];
  }
  
  MPI_Gatherv(arr.data(), asINT(arr.size()), MPI_DOUBLE,
              external_buff.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, root, comm);
}


//////////////////////////////
// # CMatrixP MISC. METHODS
//////////////////////////////
void CMatrixP::random_fill() {
  MPITiming::Stopwatch sw(tk);
  sw.start("randfill");

  static thread_local std::mt19937 engine{std::random_device{}()};
  std::uniform_real_distribution<double> dist(RAND_LOWER_BOUND, RAND_UPPER_BOUND);
  std::generate(arr.begin(), arr.end(), [&]() { return dist(engine); });
  sw.stop();
}


void CMatrixP::calc_nrows_offsets() {
  size_t rank_nrows{(N / csize) + 1};
  all_nrows.reserve(csize);
  all_offsets.reserve(csize);

  for (size_t i = 0; i < rows_remaining; i++) {
    all_nrows.push_back(rank_nrows);
    all_offsets.push_back(i * rank_nrows);
  }

  rank_nrows--;
  for (size_t i = rows_remaining; i < asST(csize); i++) {
    all_nrows.push_back(rank_nrows);
    all_offsets.push_back( ((rank_nrows + 1) *  rows_remaining) + (rank_nrows * (i - rows_remaining)) );
  }
}