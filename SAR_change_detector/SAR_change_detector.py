import numpy as np
from scipy import ndimage
from sklearn.ensemble import IsolationForest
from typing import Tuple, Optional

def uniform_spatial_filter(u: np.ndarray, filter_size: Tuple[int, int]) -> np.ndarray:
    """
    Applies a uniform spatial filter (mean filter) to the input array `u`.

    Args:
        u: 2D array-like input data.
        filter_size: Size of the filter window.

    Returns:
        Filtered array of the same shape as input `u`.

    Raises:
        ValueError: If `u` is not a 2D array or `filter_size` is not a tuple of two positive integers.
    """
    if u.ndim != 2:
        raise ValueError("Input array `u` must be a 2D array.")
    if not (isinstance(filter_size, tuple) and len(filter_size) == 2 and all(isinstance(x, int) and x > 0 for x in filter_size)):
        raise ValueError("`filter_size` must be a tuple of two positive integers.")
    
    try:
        return ndimage.uniform_filter(u, size=filter_size, mode="nearest")
    except Exception as e:
        raise RuntimeError(f"Error applying uniform filter: {e}")

def compute_filtered_magnitude(amp: np.ndarray, filter_size: Tuple[int, int]) -> np.ndarray:
    """
    Computes the magnitude of the filtered input by squaring the amplitude and applying the spatial filter.

    Args:
        amp: Amplitude array (can be a real or complex image).
        filter_size: Size of the uniform filter.

    Returns:
        Filtered magnitude array.

    Raises:
        ValueError: If `amp` is not a 2D array or `filter_size` is not a tuple of two positive integers.
    """
    if amp.ndim != 2:
        raise ValueError("Amplitude array `amp` must be a 2D array.")
    
    # Use magnitude squared for complex numbers
    magnitude_squared = np.abs(amp) ** 2
    
    return uniform_spatial_filter(magnitude_squared, filter_size)

def generate_asym(
    filter_size: Tuple[int, int] = (1, 4),
    primary_amp: Optional[np.ndarray] = None,
    secondary_amp: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generates the asymmetry map between two input amplitude images.

    Args:
        filter_size: Size of the uniform filter used in magnitude computation.
        primary_amp: Primary amplitude image (2D array).
        secondary_amp: Secondary amplitude image (2D array).

    Returns:
        2D asymmetry map with NaN values in regions where the input images contained NaNs.

    Raises:
        ValueError: If `primary_amp` or `secondary_amp` is not provided or their shapes don't match.
    """
    if primary_amp is None or secondary_amp is None:
        raise ValueError("Both primary and secondary amplitudes must be provided.")
    if primary_amp.size == 0 or secondary_amp.size == 0:
        raise ValueError("Amplitude arrays must not be empty.")
    if primary_amp.shape != secondary_amp.shape:
        raise ValueError("Primary and secondary amplitude arrays must have the same shape.")
    if primary_amp.ndim != 2 or secondary_amp.ndim != 2:
        raise ValueError("Amplitude arrays must be 2D.")

    nanmask = np.isnan(primary_amp) | np.isnan(secondary_amp)
    primary_amp_clean = np.nan_to_num(primary_amp)
    secondary_amp_clean = np.nan_to_num(secondary_amp)

    filtered_primary_mag = compute_filtered_magnitude(primary_amp_clean, filter_size)
    filtered_secondary_mag = compute_filtered_magnitude(secondary_amp_clean, filter_size)

    denominator = np.sqrt(filtered_primary_mag * filtered_secondary_mag) + 1e-10  # Add epsilon to avoid division by zero
    asym = (filtered_primary_mag + filtered_secondary_mag) / (2 * denominator)
    np.reciprocal(asym, out=asym)

    asym[nanmask] = np.nan

    return asym

def detect_changes(
    first_image: np.ndarray,
    second_image: np.ndarray,
    filter_size: Tuple[int, int] = (3, 3),
    contamination: float = 0.02
) -> np.ndarray:
    """
    Detects changes between two input images using asymmetry filtering and Isolation Forest for anomaly detection.

    Args:
        first_image: The first input image (2D array).
        second_image: The second input image (2D array).
        filter_size: The filter size used for generating the asymmetry map.
        contamination: The contamination parameter for Isolation Forest, indicating the proportion of anomalies.

    Returns:
        A change map with values -1, 0, and 1:
        - -1 indicates disappearance.
        - 0 indicates no change.
        - 1 indicates appearance.

    Raises:
        ValueError: If input images are not 2D or don't have matching shapes.
    """
    if first_image.shape != second_image.shape:
        raise ValueError("Input images must have the same shape.")
    if first_image.ndim != 2 or second_image.ndim != 2:
        raise ValueError("Input images must be 2D arrays.")
    if first_image.size == 0 or second_image.size == 0:
        raise ValueError("Input images must not be empty.")

    amp_first = np.abs(first_image)
    amp_second = np.abs(second_image)

    asym_map = generate_asym(filter_size=filter_size, primary_amp=amp_first, secondary_amp=amp_second)

    height, width = asym_map.shape
    flattened_asym = asym_map.ravel().reshape(-1, 1)

    try:
        isolation_forest = IsolationForest(contamination=contamination, random_state=0)
        anomaly_labels = isolation_forest.fit_predict(flattened_asym)
    except Exception as e:
        raise RuntimeError(f"Error applying Isolation Forest: {e}")

    anomalies_image = anomaly_labels.reshape(height, width)
    final_change_map = np.zeros_like(anomalies_image, dtype=np.int8)
    
    # Difference between amplitudes for post anomaly detection classification
    difference = amp_second - amp_first
    
    # If an anomaly is detected, mark appearance (1) or disappearance (-1)
    final_change_map[anomalies_image == -1] = np.where(difference[anomalies_image == -1] > 0, 1, -1)
    
    return final_change_map