import pytest
import numpy as np
from scipy import ndimage
from SAR_change_detector import uniform_spatial_filter, compute_filtered_magnitude, generate_asym, detect_changes

def test_uniform_spatial_filter():
    # Test valid input
    u = np.array([[1, 2], [3, 4]])
    filter_size = (2, 2)
    result = uniform_spatial_filter(u, filter_size)
    expected = ndimage.uniform_filter(u, size=filter_size, mode="nearest")
    np.testing.assert_array_almost_equal(result, expected)

    # Test invalid input: not a 2D array
    with pytest.raises(ValueError):
        uniform_spatial_filter(np.array([1, 2, 3]), filter_size)

    # Test invalid filter size
    with pytest.raises(ValueError):
        uniform_spatial_filter(u, (2, -1))

def test_uniform_spatial_filter_edge_cases():
    # Test edge case: Single element array
    u = np.array([[10]])
    filter_size = (1, 1)
    result = uniform_spatial_filter(u, filter_size)
    np.testing.assert_array_equal(result, u)

    # Test large array performance (not checking output, just ensuring no crash)
    large_u = np.random.rand(1000, 1000)
    result = uniform_spatial_filter(large_u, (3, 3))
    assert result.shape == large_u.shape

    # Test with NaN values
    u_with_nan = np.array([[1, 2], [np.nan, 4]])
    result = uniform_spatial_filter(u_with_nan, filter_size=(2, 2))
    assert np.isnan(result).any()  # Ensure NaNs are handled correctly

def test_compute_filtered_magnitude_complex_input():
    # Test with complex input
    amp = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
    filter_size = (2, 2)
    result = compute_filtered_magnitude(amp, filter_size)
    expected = uniform_spatial_filter(np.abs(amp) ** 2, filter_size)
    np.testing.assert_array_almost_equal(result, expected)
    
    # Test edge case: Empty array
    with pytest.raises(ValueError):
        compute_filtered_magnitude(np.array([]), filter_size)

def test_generate_asym_nan_handling():
    # Test with NaN values in input arrays
    primary_amp = np.array([[1, np.nan], [3, 4]])
    secondary_amp = np.array([[4, 3], [np.nan, 1]])
    result = generate_asym((2, 2), primary_amp, secondary_amp)
    assert np.isnan(result).any()  # Ensure NaNs are propagated

def test_generate_asym_edge_cases():
    # Test edge case: Empty array
    primary_amp = np.array([[]])
    secondary_amp = np.array([[]])
    with pytest.raises(ValueError):
        generate_asym((2, 2), primary_amp, secondary_amp)

    # Test edge case: Zeros in amplitude
    primary_amp = np.zeros((2, 2))
    secondary_amp = np.zeros((2, 2))
    result = generate_asym((2, 2), primary_amp, secondary_amp)
    assert np.isinf(result).all()  # Expecting inf due to division by zero

def test_compute_filtered_magnitude():
    # Test valid input
    amp = np.array([[1, 2], [3, 4]])
    filter_size = (2, 2)
    result = compute_filtered_magnitude(amp, filter_size)
    expected = uniform_spatial_filter(np.abs(amp) ** 2, filter_size)
    np.testing.assert_array_almost_equal(result, expected)

    # Test invalid input: not a 2D array
    with pytest.raises(ValueError):
        compute_filtered_magnitude(np.array([1, 2, 3]), filter_size)

def test_generate_asym():
    # Test valid input
    primary_amp = np.array([[1, 2], [3, 4]])
    secondary_amp = np.array([[4, 3], [2, 1]])
    filter_size = (2, 2)
    result = generate_asym(filter_size, primary_amp, secondary_amp)
    assert result.shape == primary_amp.shape

    # Test invalid input: primary_amp is None
    with pytest.raises(ValueError):
        generate_asym(filter_size, None, secondary_amp)

    # Test invalid input: secondary_amp is None
    with pytest.raises(ValueError):
        generate_asym(filter_size, primary_amp, None)

    # Test invalid input: shapes don't match
    with pytest.raises(ValueError):
        generate_asym(filter_size, primary_amp, np.array([[1, 2, 3], [4, 5, 6]]))

    # Test invalid input: not 2D arrays
    with pytest.raises(ValueError):
        generate_asym(filter_size, np.array([1, 2, 3]), secondary_amp)

def test_detect_changes():
    # Test valid input
    first_image = np.array([[1, 2], [3, 4]])
    second_image = np.array([[4, 3], [2, 1]])
    filter_size = (2, 2)
    result = detect_changes(first_image, second_image, filter_size)
    assert result.shape == first_image.shape

    # Test invalid input: shapes don't match
    with pytest.raises(ValueError):
        detect_changes(first_image, np.array([[1, 2, 3], [4, 5, 6]]), filter_size)

    # Test invalid input: not 2D arrays
    with pytest.raises(ValueError):
        detect_changes(np.array([1, 2, 3]), second_image, filter_size)

    # Test invalid input: empty arrays
    with pytest.raises(ValueError):
        detect_changes(np.array([[]]), np.array([[]]), filter_size)

def test_detect_changes_identical_images():
    # Test with identical images (no changes should be detected)
    image = np.array([[1, 2], [3, 4]])
    result = detect_changes(image, image, filter_size=(2, 2))
    np.testing.assert_array_equal(result, np.zeros_like(image))  # Expecting no changes

def test_detect_changes_with_noise():
    # Test with images containing random noise
    np.random.seed(0)
    first_image = np.random.rand(10, 10)
    second_image = np.random.rand(10, 10)
    result = detect_changes(first_image, second_image, filter_size=(3, 3))
    assert result.shape == first_image.shape

def test_detect_changes_contamination_effect():
    # Test how contamination affects the output
    first_image = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])
    
    second_image = np.array([
        [16, 15, 14, 13],
        [12, 11, 10, 9],
        [8, 7, 6, 5],
        [4, 3, 2, 1]
    ])
    result_low_contam = detect_changes(first_image, second_image, contamination=0.01)
    result_high_contam = detect_changes(first_image, second_image, contamination=0.5)

    assert np.any(result_low_contam != result_high_contam)