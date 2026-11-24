#pragma once

#include <iostream>
#include <mpi.h>
#include <array>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <type_traits>


template <typename T>
static size_t asST(const T v) {
  if (v < 0) {
    std::cerr << "Error: int value " << v << " is negative and cannot be converted to size_t!";
    std::exit(EXIT_FAILURE);
  }
  return static_cast<size_t>(v);
}


template <typename T>
T round_up_to_multiple(T n, T k) {
  static_assert(std::is_integral_v<T>, "perfect_squareroot(): T must be an integral type!\n");
  if (k <= 0) throw std::invalid_argument("k must be > 0");
  if (n < 0)  throw std::invalid_argument("n must be >= 0");
  const T r = n % k;
  return r ? (n + (k - r)) : n;
}


struct CartGrid {
  size_t N{};
  MPI_Comm active{MPI_COMM_NULL};             // The active comm
  int P{-1};                                  // grid dimension (P x P)
  int asize{-1}, arank{-1};
  MPI_Comm cart{MPI_COMM_NULL};               // 2D Cartesian communicator
  int prank{MPI_PROC_NULL};                   // rank in cart

  std::array<int, 2> dims = {0, 0};           // {rows, cols} = {P, P}
  std::array<int, 2> periods = {1, 1};        // periodic in both directions
  std::array<int, 2> coords = {-1, -1};       // (row, col) in the grid
  int up{-1}, down{-1}, left{-1}, right{-1};  // 4-neighbors in cart

  size_t N_pad{};
  size_t block{};                             // local tile size = N_pad / P
  std::array<size_t, 2> offsets = {0, 0};
  std::array<size_t, 2> ndims_real = {0, 0};

  CartGrid(const size_t order, MPI_Comm active_comm, const int cart_dim);
};


CartGrid::CartGrid(const size_t order, MPI_Comm active_comm, const int cart_dim) : N(order), active(active_comm), P(cart_dim) {
  MPI_Comm_size(active, &asize);
  MPI_Comm_rank(active, &arank);
  if (asize != P * P) {
    std::cerr << "The active rank comm size and the cart-dimension providedd to CartGrid are incompatible!\n";
    MPI_Finalize();
    std::exit(EXIT_FAILURE);
  }
  
  N_pad = round_up_to_multiple(N, asST(P));
  block = N_pad / P;

  MPI_Dims_create(asize, 2, dims.data());
  if (dims[0] != dims[1]) { dims[0] = P; dims[1] = P; }
  
  if (MPI_Cart_create(active, 2, dims.data(), periods.data(), 1, &cart) != MPI_SUCCESS || cart == MPI_COMM_NULL) {
    throw std::runtime_error("MPI_Cart_create failed");
  }

  MPI_Comm_rank(cart, &prank);
  MPI_Cart_coords(cart, prank, 2, coords.data());
  MPI_Cart_shift(cart, 1, 1, &left, &right);
  MPI_Cart_shift(cart, 0, 1, &up,   &down);

  offsets[0] = asST(coords[0]) * block;
  offsets[1] = asST(coords[1]) * block;
  ndims_real[0] = std::max(size_t{}, std::min(block, N - offsets[0]));
  ndims_real[1] = std::max(size_t{}, std::min(block, N - offsets[1]));
}