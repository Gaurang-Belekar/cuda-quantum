

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../../../python_packages/cudaq pytest -rP  %s | FileCheck %s

import os

import pytest
import numpy as np

import cudaq

def test_tuple_assign():
    @cudaq.kernel(jit=True)
    def fancyCnot(a:cudaq.qubit, b:cudaq.qubit):
        x.ctrl(a, b)

    @cudaq.kernel(jit=True)
    def toffoli():
        q = cudaq.qvector(3)
        ctrl = q.front()
        x(ctrl, q[2])
        cudaq.control(fancyCnot, [ctrl], q[1], q[2])
    
    print(toffoli)

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__toffoli() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<3>
# CHECK:           %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<3>) -> !quake.ref
# CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_0]][2] : (!quake.veq<3>) -> !quake.ref
# CHECK:           quake.x %[[VAL_1]] : (!quake.ref) -> ()
# CHECK:           quake.x %[[VAL_2]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<3>) -> !quake.ref
# CHECK:           quake.apply @__nvqpp__mlirgen__fancyCnot [%[[VAL_1]]] %[[VAL_3]], %[[VAL_2]] : (!quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }