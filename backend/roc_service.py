"""
ROC (Rank Order Centroid) modul za odredjivanje tezina modula.

Izvor: formula i metod su standardni u MCDM literaturi za slucajeve
kada postoji samo RANG kriterijuma (redoslijed po vaznosti), bez
potrebe za punim parnim poredjenjima kao kod AHP-a (Saaty, 1980).
ROC je narocito pogodan kada je broj dostupnih donosilaca odluka
(eksperata) mali - za razliku od AHP ankete koja idealno zahtijeva
vise ispitanika (npr. Sidanski, 2025, koji je anketirao 65+ ljudi),
ROC zahtijeva samo JEDNO, jasno rangiranje kriterijuma.

Zamjenjuje raniju AHP-baziranu verziju (ahp_service.py) kao primarni
metod za odredjivanje tezina modula (quantitative/authenticity/
sentiment/brand_fit) u ovom radu, jer autor nije imao pristup
dovoljnom broju eksperata za pouzdanu AHP anketu. AHP modul
(ahp_service.py) je zadrzan u kodu kao alternativna, dokumentovana
opcija - ne obrisan, samo vise nije podrazumijevan izbor.

Formula (za n kriterijuma, kriterijum na rangu k, k=1 je najvazniji):
    w_k = (1/n) * sum(i=k do n) 1/i

Ova formula predstavlja "centroid" (teziste) svih mogucih tezinskih
vektora konzistentnih sa datim rangom - otud naziv "Rank Order
Centroid". Rezultat je uvijek normalizovan (suma tezina = 1) po
konstrukciji.
"""

MODULE_LABELS = ["quantitative", "authenticity", "sentiment", "brand_fit"]

# ---------------------------------------------------------------------
# Rangiranje modula od najvaznijeg (rang 1) do najmanje vaznog (rang 4).
#
# OBRAZLOZENJE RANGA (procjena autora, revidirano nakon testiranja -
# poglavlje 4.3):
# 1. Brand-fit - najvazniji; tematska irelevantnost sadrzaja u odnosu
#    na kategoriju brenda predstavlja fundamentalan strateski rizik
#    koji ne moze biti nadoknadjen ni najboljom moguicom autenticnoscu
#    ili angazovanjem publike - cak i savrseno prava, angazovana
#    publika nece kupiti proizvod koji nije relevantan za njene
#    interese (empirijski ilustrovano testiranjem - vidi MKBHD +
#    NYX Professional Makeup primjer, gdje je visoka autenticnost
#    [78.76] bez odgovarajuceg tematskog poklapanja i dalje
#    proizvodila neopravdano nizak rizik po ranijem rangiranju).
# 2. Autenticnost - drugi po vaznosti; lazna/kupljena publika ostaje
#    ozbiljan, potencijalno diskvalifikujuci rizik (conjunctive/
#    aspiration level logika, ssrn5468566.pdf), ali se sada smatra
#    sekundarnim u odnosu na temeljnu relevantnost sadrzaja za brend.
# 3. Kvantitativne metrike - vazne za doseg/vidljivost kampanje, ali
#    manje kriticne od tematske relevantnosti i autenticnosti publike.
# 4. Sentiment - rangiran najnize; raspolozenje publike je promjenljivo
#    i manje pouzdan dugorocni prediktor rizika saradnje.
#
# NAPOMENA (transparentno navedeno): ovo rangiranje je procjena autora
# rada, ne rezultat formalne ankete vise eksperata. Prvobitna verzija
# rangiranja (autenticnost #1) je revidirana nakon opservacije da
# generise neintuitivno niske rizik-procjene za slucajeve ekstremne
# tematske neusaglasenosti (npr. tehnoloski kreator x kozmeticki
# brend) uprkos visokoj autenticnosti kanala - ovo je dokumentovano
# kao dio iterativnog procesa kalibracije modela.
# ---------------------------------------------------------------------

MODULE_RANK_ORDER = ["brand_fit", "authenticity", "quantitative", "sentiment"]

def compute_roc_weights(rank_order: list) -> dict:
    """
    Racuna ROC tezine iz liste kriterijuma poredjanih od najvaznijeg
    ka najmanje vaznom.

    rank_order: lista naziva kriterijuma, redoslijed = rang
        (prvi element = rang 1 = najvazniji).
    """
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
    """
    Vraca ROC tezine za 4 modula risk-scoring modela, racunato iz
    MODULE_RANK_ORDER.
    """
    return compute_roc_weights(MODULE_RANK_ORDER)


# Brzi test
if __name__ == "__main__":
    result = get_module_weights()
    print("=== ROC TEZINE MODULA ===")
    print(f"Rang (najvazniji -> najmanje vazan): {result['rank_order']}")
    for label, w in result["weights"].items():
        print(f"  {label}: {w}")
    print(f"Suma tezina: {sum(result['weights'].values())}")