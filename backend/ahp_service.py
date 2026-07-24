"""
AHP (Analytic Hierarchy Process) modul.

Izvori:
- Saaty, T.L. (1980). The Analytic Hierarchy Process. McGraw-Hill.
- Saaty, R.W. (1987). "The Analytic Hierarchy Process - What It Is and
  How It Is Used". Mathl Modelling, 9(3-5), 161-176.
- Podvezko, V. (2009). "Application of AHP technique". Journal of
  Business Economics and Management, 10(2), 181-189.

Kljucna razlika u odnosu na prethodnu verziju: ovdje se NE unose gotove
tezine (w1..w4) rucno. Umjesto toga, unosi se PAIRWISE COMPARISON
MATRICA (relativna vaznost svakog para modula, Saaty skala 1-9), a
tezine se IZRACUNAVAJU iz glavnog sopstvenog vektora te matrice. Matrica
je jos uvijek subjektivna procjena autora/eksperata - to je legitiman
AHP ulaz - ali finalni brojevi (tezine) su matematicki izvedeni, ne
proizvoljno upisani.

Random Consistency Index (RI) tabela je iz Saaty (1987), za matrice
velicine n=1..10.
"""

import numpy as np

RANDOM_CONSISTENCY_INDEX = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
}


def compute_ahp_weights(comparison_matrix: np.ndarray, labels: list) -> dict:
    """
    Racuna AHP tezine iz pairwise comparison matrice.

    comparison_matrix: kvadratna, pozitivna, reciprocna matrica
        (a_ij = 1 / a_ji, dijagonala = 1), Saaty skala 1-9.
    labels: nazivi kriterijuma/modula, redoslijed mora odgovarati
        redovima/kolonama matrice.

    Vraca dict sa tezinama, konzistentnostima i detaljima proracuna,
    tako da se cijeli postupak moze prikazati transparentno (npr. kroz
    /api/v1/methodology endpoint).
    """
    n = comparison_matrix.shape[0]
    if comparison_matrix.shape != (n, n):
        raise ValueError("Comparison matrix mora biti kvadratna.")
    if len(labels) != n:
        raise ValueError("Broj labela mora odgovarati dimenziji matrice.")

    # Glavni sopstveni vektor (Saaty, 1987, Sec. 4) - koristimo najveci
    # sopstveni eigenvalue/eigenvector matrice.
    eigenvalues, eigenvectors = np.linalg.eig(comparison_matrix)
    max_index = np.argmax(eigenvalues.real)
    lambda_max = eigenvalues[max_index].real
    principal_eigenvector = eigenvectors[:, max_index].real

    # Normalizacija na sumu 1 -> ovo su konacne tezine
    weights = principal_eigenvector / principal_eigenvector.sum()
    weights = np.abs(weights)  # eigenvector moze doci sa negativnim predznakom u cjelini

    # Consistency Index i Consistency Ratio (Saaty, 1987)
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


# ---------------------------------------------------------------------
# Pairwise comparison matrica za 4 modula risk-scoring modela.
#
# NAPOMENA (vazno za rad): ovo su procjene autora rada, koje bi u
# potpunoj metodologiji trebalo prikupiti anketom vise eksperata
# (marketing profesionalci / mentor), po uzoru na Sidanski (2025) koji
# je anketirao 65+ marketing zaposlenih za slican AHP model u
# digitalnom marketingu, ili Cabrita & Frade (2016) koji su koristili
# internu ekspertizu kompanije. Trenutne vrijednosti treba tretirati
# kao POCETNU/PLACEHOLDER prosudbu koju rad treba validirati ili
# zamijeniti stvarnom anketom prije finalne odbrane.
#
# Obrazlozenje trenutnih procjena:
# - Autenticnost (A) je ocijenjena kao najvaznija (naspram Q, S, B),
#   jer laznu/kupljenu publiku smatramo najozbiljnijim, potencijalno
#   diskvalifikujucim rizikom (u skladu sa "conjunctive"/"aspiration
#   level" logikom iz ssrn5468566.pdf - investitori odbacuju opciju
#   ako kljucni atribut padne ispod praga, bez obzira na ostale).
# - Sentiment (S) je umjereno vazniji od kvantitativnih metrika (Q),
#   jer direktno odrazava reputacioni rizik.
# - Brand-fit (B) je ocijenjen kao najmanje vazan OD OVA CETIRI u
#   direktnom poredjenju, ne zato sto je nebitan, nego zato sto (za
#   razliku od ostala tri) njegov uticaj vec eksplicitno ulazi i kroz
#   risk cap mehanizam (vidi risk_aggregation.py) - pa njegova tezina
#   u ponderisanom zbiru namjerno ne duplira taj uticaj.
#
# Redoslijed: Q (kvantitativni), A (autenticnost), S (sentiment), B (brand-fit)
# ---------------------------------------------------------------------

MODULE_LABELS = ["quantitative", "authenticity", "sentiment", "brand_fit"]

# a_ij = koliko puta je red i vazniji od kolone j (Saaty skala 1-9,
# recipro ne vrijednosti 1/n za obrnut smjer)
MODULE_COMPARISON_MATRIX = np.array([
    #        Q       A      S      B
    [1.0,    1/2,   1.0,   2.0],   # Q
    [2.0,    1.0,   2.0,   3.0],   # A
    [1.0,    1/2,   1.0,   2.0],   # S
    [1/2,    1/3,   1/2,   1.0],   # B
])


def get_module_weights() -> dict:
    """
    Vraca izracunate tezine za 4 modula risk-scoring modela, zajedno
    sa CR i ostalim detaljima, racunato iz MODULE_COMPARISON_MATRIX.
    """
    return compute_ahp_weights(MODULE_COMPARISON_MATRIX, MODULE_LABELS)


# Brzi test
if __name__ == "__main__":
    result = get_module_weights()
    print("=== AHP TEZINE MODULA ===")
    for label, w in result["weights"].items():
        print(f"  {label}: {w}")
    print(f"Lambda max: {result['lambda_max']}")
    print(f"Consistency Index (CI): {result['consistency_index']}")
    print(f"Consistency Ratio (CR): {result['consistency_ratio']}")
    print(f"Konzistentno (CR < 0.10)? {result['is_consistent']}")