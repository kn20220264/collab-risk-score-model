"""Shannon entropijski modul za objektivno odredjivanje tezina kriterijuma (Hwang & Yoon, 1981)."""

import numpy as np


def compute_entropy_weights(decision_matrix: np.ndarray, labels: list) -> dict:
    """Racuna Shannon entropijske tezine iz decision matrice (redovi=kreatori, kolone=kriterijumi)."""
    m, n = decision_matrix.shape
    if len(labels) != n:
        raise ValueError("Broj labela mora odgovarati broju kolona matrice.")
    if m < 2:
        raise ValueError(
            "Entropijska metoda zahtijeva najmanje 2 kreatora u uzorku "
            "da bi se izracunala varijabilnost. Za jednog kreatora "
            "koristiti AHP tezine (ahp_service.py)."
        )

    # Korak 1: normalizacija (kolone sabiraju na 1)
    col_sums = decision_matrix.sum(axis=0)
    col_sums = np.where(col_sums == 0, 1e-9, col_sums)
    p = decision_matrix / col_sums

    # Korak 2: entropija po kriterijumu
    k = 1 / np.log(m)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_ln_p = np.where(p > 0, p * np.log(p), 0.0)
    e = -k * p_ln_p.sum(axis=0)

    # Korak 3: stepen diverzifikacije (d_j = 1 - e_j) i normalizacija u tezine
    d = 1 - e
    if d.sum() == 0:
        weights = np.ones(n) / n
    else:
        weights = d / d.sum()

    return {
        "weights": {label: round(float(w), 4) for label, w in zip(labels, weights)},
        "entropy_per_criterion": {label: round(float(v), 4) for label, v in zip(labels, e)},
        "sample_size": m,
    }


def build_decision_matrix(module_scores_list: list, labels: list) -> np.ndarray:
    """Pretvara listu module_scores dict-ova u decision matricu za compute_entropy_weights."""
    return np.array([[creator_scores[label] for label in labels] for creator_scores in module_scores_list])


if __name__ == "__main__":
    example_labels = ["quantitative", "authenticity", "sentiment", "brand_fit"]
    example_matrix = np.array([
        [70, 60, 80, 40],
        [50, 90, 55, 75],
        [65, 40, 70, 90],
    ])

    result = compute_entropy_weights(example_matrix, example_labels)
    print("=== ENTROPIJSKE TEZINE (primjer) ===")
    for label, w in result["weights"].items():
        print(f"  {label}: {w}")
