import numpy as np
from scipy import ndimage
from sklearn.ensemble import IsolationForest


def uniform_spatial_filter(u, filter_size):
    return ndimage.uniform_filter(u, size=filter_size, mode="nearest")


def compute_filtered_magnitude(amp, filter_size):
    return uniform_spatial_filter(amp**2, filter_size)


def generate_asym(
    filter_size=(1, 4),
    primary_amp=None,
    secondary_amp=None
):
    # Vérification des types et des données fournies
    assert type(filter_size) == tuple, "filter size must be tuple"
    assert (primary_amp is not None) and (secondary_amp is not None), "amplitudes should be provided for asym computation"

    # Création du masque pour les valeurs NaN
    nanmask = np.isnan(primary_amp) | np.isnan(secondary_amp)

    # Mise à zéro des amplitudes dans les zones NaN
    primary_amp[nanmask] = 0
    secondary_amp[nanmask] = 0

    # Calcul du terme asymétrique (asym)
    asym = ((compute_filtered_magnitude(primary_amp, filter_size) 
             + compute_filtered_magnitude(secondary_amp, filter_size)) / 2) / (
                np.sqrt(compute_filtered_magnitude(primary_amp, filter_size) 
                * compute_filtered_magnitude(secondary_amp, filter_size)) + 1e-10
            )
    asym[nanmask] = np.nan
    asym = 1 / asym  # Inversion du résultat pour obtenir l'asymétrie correcte

    return asym


# Fonction principale de détection de changement
def detect_changes(first_image, second_image, filter_size=(3, 3), contamination=0.02):
    
    
    """
Detects changes between two input images.

Parameters:
- first_image: np.array, the first input image.
- second_image: np.array, the second input image.
- filter_size: tuple, the filter size used for generating asymmetric term.
- contamination: float, the contamination parameter for Isolation Forest.

Returns:
- final_change_map: np.array, a change map with values -1, 0, and 1.
    - -1 indicates disappearance.
    - 0 indicates no change.
    - 1 indicates appearance.
"""
    # Calculer l'amplitude des deux images
    amp_first = np.abs(first_image)
    amp_second = np.abs(second_image)

    # Générer la carte asymétrique
    asym_test = generate_asym(filter_size=filter_size, primary_amp=amp_first, secondary_amp=amp_second)

    # Appliquer Isolation Forest pour détecter les changements
    height, width = asym_test.shape
    data = asym_test.ravel().reshape(-1, 1)

    isolation_forest = IsolationForest(contamination=contamination, random_state=0)
    anomaly_labels = isolation_forest.fit_predict(data)

    # Convertir les labels d'anomalies en une image binaire
    anomalies_image = (anomaly_labels == -1).astype(np.uint8).reshape(height, width)

    # Créer l'image de changement final
    # Initialiser l'image de sortie avec des zéros
    final_change_map = np.zeros_like(anomalies_image, dtype=np.int8)

    # Différence entre la première et la deuxième image
    difference = amp_second - amp_first

    # Appliquer la règle de segmentation sur les zones détectées comme changements
    # Là où anomalies_image est 1 (c.-à-d. changements détectés)
    final_change_map[anomalies_image == 1] = np.where(difference[anomalies_image == 1] > 0, 1, -1)

    return final_change_map




