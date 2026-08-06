"""AHP (Analytic Hierarchy Process) modul za racunanje tezina modula iz pairwise comparison matrice."""

import numpy as np

RANDOM_CONSISTENCY_INDEX = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
}


def compute_ahp_weights(comparison_matrix: np.ndarray, labels: list) -> dict:
    """Racuna AHP tezine, konzistentnost i detalje proracuna iz pairwise comparison matrice."""
    n = comparison_matrix.shape[0]
    if comparison_matrix.shape != (n, n):
        raise ValueError("Comparison matrix mora biti kvadratna.")
    if len(labels) != n:
        raise ValueError("Broj labela mora odgovarati dimenziji matrice.")

    eigenvalues, eigenvectors = np.linalg.eig(comparison_matrix)
    max_index = np.argmax(eigenvalues.real)
    lambda_max = eigenvalues[max_index].real
    principal_eigenvector = eigenvectors[:, max_index].real

    weights = principal_eigenvector / principal_eigenvector.sum()
    weights = np.abs(weights)

    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = RANDOM_CONSISTENCY_INDEX.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0

    return {
        "weights": {label: round(float(w), 4) for label, w in zip(labels, weights)},
        "lambda_max": round(float(lambda_max), 4),
        "consistency_index": round(float(ci), 4),
        "consistency_ratio": round(float(cr), 4),
        "is_consistent": bool(cr < 0.10),
        "comparison_matrix": comparison_matrix.tolist(),
        "labels": labels,
    }


MODULE_LABELS = ["quantitative", "authenticity", "sentiment", "brand_fit"]

# a_ij = koliko puta je red i vazniji od kolone j (Saaty skala 1-9)
MODULE_COMPARISON_MATRIX = np.array([
    #        Q       A      S      B
    [1.0,    1/2,   1.0,   2.0],   # Q
    [2.0,    1.0,   2.0,   3.0],   # A
    [1.0,    1/2,   1.0,   2.0],   # S
    [1/2,    1/3,   1/2,   1.0],   # B
])


def get_module_weights() -> dict:
    return compute_ahp_weights(MODULE_COMPARISON_MATRIX, MODULE_LABELS)


if __name__ == "__main__":
    result = get_module_weights()
    print("=== AHP TEZINE MODULA ===")
    for label, w in result["weights"].items():
        print(f"  {label}: {w}")
    print(f"Lambda max: {result['lambda_max']}")
    print(f"Consistency Index (CI): {result['consistency_index']}")
    print(f"Consistency Ratio (CR): {result['consistency_ratio']}")
    print(f"Konzistentno (CR < 0.10)? {result['is_consistent']}")
