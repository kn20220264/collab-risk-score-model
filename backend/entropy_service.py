"""
Shannon entropijski modul za objektivno odredjivanje tezina kriterijuma.

Izvor: Hwang, C.L. & Yoon, K. (1981). Multiple Attribute Decision
Making: Methods and Applications. Springer-Verlag. (temeljni,
najcesce citirani rad za entropijsku metodu tezina)

Kljucna razlika u odnosu na AHP: entropija ne trazi subjektivnu
prosudbu eksperta o vaznosti kriterijuma, nego IZVODI tezine direktno
iz varijabilnosti podataka - kriterijum koji vise "razlikuje"
kreatore u datom uzorku dobija vecu tezinu. Zbog toga entropijska
metoda ima smisla samo kada se poredi VISE kreatora odjednom (decision
matrix: redovi = kreatori, kolone = kriterijumi/moduli) - za analizu
jednog jedinog kreatora nema varijabilnosti iz koje bi se tezine
izracunale, pa se u tom slucaju koristi AHP (vidi ahp_service.py).

Ovo je razlog zasto u ovoj aplikaciji:
- GET /api/v1/creators/youtube/{handle} (jedan kreator) koristi
  fiksne AHP tezine.
- POST /api/v1/creators/bulk (vise kreatora) moze da koristi
  entropijske tezine izracunate iz tog konkretnog batch-a, kao
  objektivnu alternativu/provjeru AHP tezinama.

Napomena o ogranicenju metode (za rad, poglavlje 4.1): Banihabib et
al. (2020, SN Applied Sciences) pokazuju da Shannon-ova entropija,
kao cisto objektivna metoda, moze dati tezine koje ne odrazavaju
stvarnu vaznost kriterijuma za odluku, jer se oslanja iskljucivo na
statisticku varijabilnost, a ne na ekspertsku prosudbu. Zbog toga se
u ovom radu entropijske tezine koriste kao DOPUNSKA, uporedna
perspektiva uz AHP, a ne kao zamjena za njega.
"""

import numpy as np


def compute_entropy_weights(decision_matrix: np.ndarray, labels: list) -> dict:
    """
    Racuna Shannon entropijske tezine iz decision matrice.

    decision_matrix: 2D niz oblika (broj_kreatora, broj_kriterijuma).
        Sve vrijednosti moraju biti >= 0 (koristimo module_scores
        skalirane 0-100, pa je ovo uvijek zadovoljeno).
    labels: nazivi kriterijuma/modula, redoslijed mora odgovarati
        kolonama matrice.
    """
    m, n = decision_matrix.shape
    if len(labels) != n:
        raise ValueError("Broj labela mora odgovarati broju kolona matrice.")
    if m < 2:
        raise ValueError(
            "Entropijska metoda zahtijeva najmanje 2 kreatora u uzorku "
            "da bi se izracunala varijabilnost. Za jednog kreatora "
            "koristiti AHP tezine (ahp_service.py)."
        )

    # Korak 1: normalizacija (svaka kolona sabira na 1)
    # Dodajemo mali epsilon da izbjegnemo dijeljenje sa nulom ako je
    # cijela kolona 0.
    col_sums = decision_matrix.sum(axis=0)
    col_sums = np.where(col_sums == 0, 1e-9, col_sums)
    p = decision_matrix / col_sums

    # Korak 2: entropija po kriterijumu (Hwang & Yoon, 1981)
    k = 1 / np.log(m)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_ln_p = np.where(p > 0, p * np.log(p), 0.0)
    e = -k * p_ln_p.sum(axis=0)

    # Korak 3: stepen diverzifikacije (d_j = 1 - e_j) i normalizacija u tezine
    d = 1 - e
    if d.sum() == 0:
        # Svi kriterijumi identicno "informativni" -> jednake tezine
        weights = np.ones(n) / n
    else:
        weights = d / d.sum()

    return {
        "weights": {label: round(float(w), 4) for label, w in zip(labels, weights)},
        "entropy_per_criterion": {label: round(float(v), 4) for label, v in zip(labels, e)},
        "sample_size": m,
    }


def build_decision_matrix(module_scores_list: list, labels: list) -> np.ndarray:
    """
    Pomocna funkcija: pretvara listu module_scores dict-ova (jedan po
    kreatoru, npr. [{"quantitative": 62.1, "authenticity": 80.0, ...}, ...])
    u numpy decision matricu spremnu za compute_entropy_weights.
    """
    return np.array([[creator_scores[label] for label in labels] for creator_scores in module_scores_list])


# Brzi test
if __name__ == "__main__":
    # Simulacija: 3 kreatora, 4 modula (kao ilustracija - u pravoj
    # upotrebi ovo dolazi iz stvarno izracunatih module_scores rezultata)
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