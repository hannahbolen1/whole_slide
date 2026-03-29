def get_neighbor_features(mask1, mask2, prefix=""):
    """Extract neighbor relationship features (Type 3: 2 masks).

    Args:
        mask1 (np.ndarray): First segmentation mask
        mask2 (np.ndarray): Second segmentation mask
        prefix (str): Prefix for feature names

    Returns:
        dict: Dictionary of extracted features of the form {'label':[measurements,],}
    """
    results = {}  # initializing dict of extracted features
    measurements = get_multimask_measurements().items()
    try:
        # Extract all correlation measurements
        for name, measure_func in measurements:
            features = measure_func(
                mask1,
                mask2,
            )
            # Add prefix to feature names to ID source
            if prefix:
                features = {f"{prefix}_{k}": v for k, v in features.items()}
            results.update(features)
    except Exception as e:
        print(
            f"Error calling neighbor measurment {name} with {mask1} and {mask2}: {str(e)}"
        )
    return results



### alternative feature extraction functions
mask_pairs = [
    (nuclei, nuclei, "nucleus"),
    (cytoplasms, cytoplasms, "cytoplasm"),
    (cells, cells, "cell"),
]

for mask1, mask2, prefix in mask_pairs:
    if (
        mask1 is not None
        and mask2 is not None
        and np.any(mask1 > 0)
        and np.any(mask2 > 0)
    ):
        features = get_neighbor_features(mask1, mask2, f"{prefix}_neighbor__")
        if features:
            all_features.append(pd.DataFrame(features))