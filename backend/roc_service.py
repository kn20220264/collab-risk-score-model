"""ROC (Rank Order Centroid) modul za odredjivanje tezina modula iz rangiranja kriterijuma."""

MODULE_LABELS = ["quantitative", "authenticity", "sentiment", "brand_fit"]

# Rangiranje od najvaznijeg ka najmanje vaznom.
MODULE_RANK_ORDER = ["brand_fit", "authenticity", "quantitative", "sentiment"]

def compute_roc_weights(rank_order: list) -> dict:
    """Racuna ROC tezine: w_k = (1/n) * sum(1/i) za i=k..n, k=rang kriterijuma."""
    n = len(rank_order)
    weights = {}

    for k, label in enumerate(rank_order, start=1):
        w_k = sum(1 / i for i in range(k, n + 1)) / n
        weights[label] = round(w_k, 4)

    return {
        "weights": weights,
        "rank_order": rank_order,
        "n": n,
    }


def get_module_weights() -> dict:
    return compute_roc_weights(MODULE_RANK_ORDER)


if __name__ == "__main__":
    result = get_module_weights()
    print("=== ROC TEZINE MODULA ===")
    print(f"Rang (najvazniji -> najmanje vazan): {result['rank_order']}")
    for label, w in result["weights"].items():
        print(f"  {label}: {w}")
    print(f"Suma tezina: {sum(result['weights'].values())}")
