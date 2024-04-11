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


@pytest.mark.skip(reason="Not yet supported.")
def test_builder_mode():
    """
    Builder API
    """
    kernel = cudaq.make_kernel()
    # ASKME: Reuse `cudaq.register_op`?
    kernel.register_op("custom_h", 1. / np.sqrt(2.) * np.array([[1, 1], [1, -1]]))
    kernel.register_op("custom_x", np.array([[0, 1], [1, 0]]))

    qubits = kernel.qalloc(2)    
    kernel.custom_h(qubits[0])
    kernel.custom_x.ctrl(qubits[0], qubits[1])

    check_basic(kernel)


def test_custom_adjoint():
    """Test that adjoint can be called on custom operations."""

    custom_s = cudaq.register_operation(np.array([[1, 0], [0, 1j]]))

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        h(q)
        custom_s.adj(q)
        custom_s.adj(q)
        h(q)
        mz(q)

    counts = cudaq.sample(kernel)
    counts.dump()
    assert counts["1"] == 1000


def test_bad_attribute():
    """Test that unsupported attributes on custom operations raise error."""

    custom_s = cudaq.register_operation(np.array([[1, 0], [0, 1j]]))

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        custom_s.foo(q)
        mz(q)

    with pytest.raises(Exception) as error:
        counts = cudaq.sample(kernel)


@pytest.mark.skip(reason="Not yet supported.")
def test_parameterized_op():
    """
    Test ways to define and use custom quantum operations that can accept parameters.
    """

    # (a) Using lambda
    my_rx_op = cudaq.register_operation(lambda theta: np.array([[
        np.cos(theta / 2), -1j * np.sin(theta / 2)
    ], [-1j * np.sin(theta / 2), np.cos(theta / 2)]]))

    # (b) Using a regular function
    def my_unitary(theta: float):
        return (np.array([[np.exp(-1j * theta / 2), 0],
                          [0, np.exp(1j * theta / 2)]]))

    my_rz_op = cudaq.register_operation(my_unitary)

    @cudaq.kernel
    def use_op():
        qubits = cudaq.qvector(3)

        x(qubits)
        my_rx_op(np.pi, qubits)
        r1(-np.pi, qubits)
        ry(np.pi, qubits)
        my_rz_op(np.pi, qubits)

    counts = cudaq.sample(use_op)
    assert counts["111"] == 1000


## TODO:
## Test op on one target
## Test on multiple qubits