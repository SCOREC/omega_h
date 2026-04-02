"""
Pytest tests for PyOmega_h mesh tag operations.

Tests the Python bindings for add_tag, get_tag, and set_tag methods
with both numpy arrays and omega_h Read arrays.

Run with: pytest test_mesh_tags_simple.py
"""

import pytest
import numpy as np
import PyOmega_h as omega_h


# Create a global library instance to avoid Kokkos finalization issues
# The library must outlive all mesh objects
_global_lib = None


@pytest.fixture(scope="session")
def lib():
    """Session-scoped library instance shared across all tests."""
    global _global_lib
    if _global_lib is None:
        _global_lib = omega_h.OmegaHLibrary()
    return _global_lib


@pytest.fixture
def mesh(lib):
    """Create a simple 2D box mesh for testing."""
    return omega_h.build_box(lib.world(), omega_h.SIMPLEX, 1.0, 1.0, 0.0, 2, 2, 0)


def test_add_tag_int8_numpy(mesh):
    """Test adding int8 tag with numpy array."""
    nverts = mesh.nverts()
    data_i8 = np.arange(nverts, dtype=np.int8)
    mesh.add_tag(0, "test_i8", 1, data_i8)
    
    assert mesh.has_tag(0, "test_i8")
    result = mesh.get_tag(0, "test_i8")
    np.testing.assert_array_equal(result, data_i8)


def test_add_tag_int32_numpy(mesh):
    """Test adding int32 tag with numpy array."""
    nverts = mesh.nverts()
    data_i32 = np.arange(10, 10 + nverts, dtype=np.int32)
    mesh.add_tag(0, "test_i32", 1, data_i32)
    
    assert mesh.has_tag(0, "test_i32")
    result = mesh.get_tag(0, "test_i32")
    np.testing.assert_array_equal(result, data_i32)


def test_add_tag_int64_numpy(mesh):
    """Test adding int64 tag with numpy array."""
    nverts = mesh.nverts()
    data_i64 = np.arange(100, 100 + nverts, dtype=np.int64)
    mesh.add_tag(0, "test_i64", 1, data_i64)
    
    assert mesh.has_tag(0, "test_i64")
    result = mesh.get_tag(0, "test_i64")
    np.testing.assert_array_equal(result, data_i64)


def test_add_tag_float64_numpy(mesh):
    """Test adding float64 tag with numpy array."""
    nverts = mesh.nverts()
    data_f64 = np.arange(1.5, 1.5 + nverts, dtype=np.float64)
    mesh.add_tag(0, "test_f64", 1, data_f64)
    
    assert mesh.has_tag(0, "test_f64")
    result = mesh.get_tag(0, "test_f64")
    np.testing.assert_allclose(result, data_f64)


def test_add_tag_multicomponent_numpy(mesh):
    """Test adding multi-component tag with numpy array."""
    nverts = mesh.nverts()
    data_multi = np.arange(1.0, 1.0 + nverts * 2, dtype=np.float64)
    mesh.add_tag(0, "test_multi", 2, data_multi)
    
    assert mesh.has_tag(0, "test_multi")
    result = mesh.get_tag(0, "test_multi")
    np.testing.assert_allclose(result, data_multi)


def test_add_tag_int8_read(mesh):
    """Test adding int8 tag with Read_int8 array."""
    nverts = mesh.nverts()
    read_i8 = omega_h.Read_int8(nverts, 5)
    mesh.add_tag(0, "read_i8", 1, read_i8)
    
    assert mesh.has_tag(0, "read_i8")
    result = mesh.get_tag(0, "read_i8")
    expected = np.full(nverts, 5, dtype=np.int8)
    np.testing.assert_array_equal(result, expected)


def test_add_tag_int32_read(mesh):
    """Test adding int32 tag with Read_int32 array."""
    nverts = mesh.nverts()
    read_i32 = omega_h.Read_int32(nverts, 42)
    mesh.add_tag(0, "read_i32", 1, read_i32)
    
    assert mesh.has_tag(0, "read_i32")
    result = mesh.get_tag(0, "read_i32")
    expected = np.full(nverts, 42, dtype=np.int32)
    np.testing.assert_array_equal(result, expected)


def test_add_tag_int64_read(mesh):
    """Test adding int64 tag with Read_int64 array."""
    nverts = mesh.nverts()
    read_i64 = omega_h.Read_int64(nverts, 999)
    mesh.add_tag(0, "read_i64", 1, read_i64)
    
    assert mesh.has_tag(0, "read_i64")
    result = mesh.get_tag(0, "read_i64")
    expected = np.full(nverts, 999, dtype=np.int64)
    np.testing.assert_array_equal(result, expected)


def test_add_tag_float64_read(mesh):
    """Test adding float64 tag with Read_float64 array."""
    nverts = mesh.nverts()
    read_f64 = omega_h.Read_float64(nverts, 3.14)
    mesh.add_tag(0, "read_f64", 1, read_f64)
    
    assert mesh.has_tag(0, "read_f64")
    result = mesh.get_tag(0, "read_f64")
    expected = np.full(nverts, 3.14, dtype=np.float64)
    np.testing.assert_allclose(result, expected)


def test_get_array_int8(mesh):
    """Test get_array_int8 returns Read_int8."""
    nverts = mesh.nverts()
    data = np.arange(10, 10 + nverts, dtype=np.int8)
    mesh.add_tag(0, "typed_i8", 1, data)
    
    result = mesh.get_array_int8(0, "typed_i8")
    assert isinstance(result, omega_h.Read_int8)
    assert result.size() == nverts


def test_get_array_int32(mesh):
    """Test get_array_int32 returns Read_int32."""
    nverts = mesh.nverts()
    data = np.arange(100, 100 + nverts, dtype=np.int32)
    mesh.add_tag(0, "typed_i32", 1, data)
    
    result = mesh.get_array_int32(0, "typed_i32")
    assert isinstance(result, omega_h.Read_int32)
    assert result.size() == nverts


def test_get_array_int64(mesh):
    """Test get_array_int64 returns Read_int64."""
    nverts = mesh.nverts()
    data = np.arange(1000, 1000 + nverts, dtype=np.int64)
    mesh.add_tag(0, "typed_i64", 1, data)
    
    result = mesh.get_array_int64(0, "typed_i64")
    assert isinstance(result, omega_h.Read_int64)
    assert result.size() == nverts


def test_get_array_float64(mesh):
    """Test get_array_float64 returns Read_float64."""
    nverts = mesh.nverts()
    data = np.arange(1.1, 1.1 + nverts, dtype=np.float64)
    mesh.add_tag(0, "typed_f64", 1, data)
    
    result = mesh.get_array_float64(0, "typed_f64")
    assert isinstance(result, omega_h.Read_float64)
    assert result.size() == nverts


def test_set_tag_int32_numpy(mesh):
    """Test setting int32 tag with numpy array."""
    nverts = mesh.nverts()
    data1 = np.arange(nverts, dtype=np.int32)
    mesh.add_tag(0, "settable_i32", 1, data1)
    
    data2 = np.arange(100, 100 + nverts, dtype=np.int32)
    mesh.set_tag(0, "settable_i32", data2)
    
    result = mesh.get_tag(0, "settable_i32")
    np.testing.assert_array_equal(result, data2)


def test_set_tag_float64_numpy(mesh):
    """Test setting float64 tag with numpy array."""
    nverts = mesh.nverts()
    data1 = np.arange(1.0, 1.0 + nverts, dtype=np.float64)
    mesh.add_tag(0, "settable_f64", 1, data1)
    
    data2 = np.arange(10.5, 10.5 + nverts, dtype=np.float64)
    mesh.set_tag(0, "settable_f64", data2)
    
    result = mesh.get_tag(0, "settable_f64")
    np.testing.assert_allclose(result, data2)


def test_set_tag_int32_read(mesh):
    """Test setting int32 tag with Read_int32 array."""
    nverts = mesh.nverts()
    data1 = np.arange(nverts, dtype=np.int32)
    mesh.add_tag(0, "set_read_i32", 1, data1)
    
    read_array = omega_h.Read_int32(nverts, 777)
    mesh.set_tag(0, "set_read_i32", read_array)
    
    result = mesh.get_tag(0, "set_read_i32")
    expected = np.full(nverts, 777, dtype=np.int32)
    np.testing.assert_array_equal(result, expected)


def test_set_tag_float64_read(mesh):
    """Test setting float64 tag with Read_float64 array."""
    nverts = mesh.nverts()
    data1 = np.arange(1.0, 1.0 + nverts, dtype=np.float64)
    mesh.add_tag(0, "set_read_f64", 1, data1)
    
    read_array = omega_h.Read_float64(nverts, 2.718)
    mesh.set_tag(0, "set_read_f64", read_array)
    
    result = mesh.get_tag(0, "set_read_f64")
    expected = np.full(nverts, 2.718, dtype=np.float64)
    np.testing.assert_allclose(result, expected)


def test_roundtrip_numpy_to_omegah_to_numpy(mesh):
    """Test round-trip conversion: numpy -> add_tag -> get_array_* -> add_tag -> get_tag -> numpy."""
    nverts = mesh.nverts()
    original = np.arange(1.5, 1.5 + nverts, dtype=np.float64)
    mesh.add_tag(0, "rt1", 1, original)
    
    # Get as omega_h array
    omegah_array = mesh.get_array_float64(0, "rt1")
    assert isinstance(omegah_array, omega_h.Read_float64)
    
    # Use omega_h array to create new tag
    mesh.add_tag(0, "rt2", 1, omegah_array)
    
    # Get back as numpy
    result = mesh.get_tag(0, "rt2")
    np.testing.assert_allclose(result, original)


def test_has_tag_and_remove_tag(mesh):
    """Test has_tag and remove_tag functionality."""
    nverts = mesh.nverts()
    data = np.arange(nverts, dtype=np.int32)
    
    # Tag should not exist initially
    assert not mesh.has_tag(0, "removable")
    
    # Add tag
    mesh.add_tag(0, "removable", 1, data)
    assert mesh.has_tag(0, "removable")
    
    # Remove tag
    mesh.remove_tag(0, "removable")
    assert not mesh.has_tag(0, "removable")
