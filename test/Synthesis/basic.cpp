/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %cpp_std %s | FileCheck %s
// XFAIL: *
// [SKIP_TEST]: Not implemented

#include <cudaq.h>

#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

void custom_operation() __qpu__ {
  cudaq::qvector qubits(2);
  custom_h(qubits[0]);
  custom_x.ctrl(qubits[0], qubits[1]);
}

int main() {
  std::vector<std::complex<double>> my_h{{M_SQRT1_2, M_SQRT1_2},
                                         {M_SQRT1_2, -M_SQRT1_2}};
  std::vector<std::complex<double>> my_x{{0, 1}, {1, 0}};

  auto custom_h = cudaq::register_operation(my_h);
  auto custom_x = cudaq::register_operation(my_x);
  
  auto result = cudaq::sample(custom_operation);

  std::cout << result.most_probable() << '\n';
  return 0;
}
