# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import numpy as np
import cudaq


def check_basic(entity):
    """Helper function to encapsulate checks"""
    counts = cudaq.sample(entity, shots_count=100)
    counts.dump()
    assert len(counts) == 2
    assert '00' in counts and '11' in counts


def test_basic():
    """
    Showcase user-level APIs of how to 
    (a) define a custom operation using unitary, and 
    (b) how to use it in kernel
    """
    custom_h = cudaq.register_operation(1. / np.sqrt(2.) *
                                        np.array([[1, 1], [1, -1]]))
    custom_x = cudaq.register_operation(np.array([[0, 1], [1, 0]]))

    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(2)
        custom_h(qubits[0])
        custom_x.ctrl(qubits[0], qubits[1])

    check_basic(bell)

    ## [SKIP_TEST]: Not working because 'CustomQuantumOperation' object is not callable
    ## Also works from builder
    # kernel = cudaq.make_kernel()
    # qubits = kernel.qalloc(2)
    # custom_h(qubits[0])
    # custom_x.ctrl(qubits[0], qubits[1])

    # check_basic(kernel)
