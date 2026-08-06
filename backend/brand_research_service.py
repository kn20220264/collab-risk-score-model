"""Automatsko istrazivanje brenda preko web pretrage; opis se kesira po nazivu brenda radi reproducibilnosti brand-fit skora."""

import os
import re
import json
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-sonnet-4-6"

_CACHE_PATH = os.path.join(os.path.dirname(__file__), "brand_profile_cache.json")

def _load_cache() -> dict:
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    """Atomic write (tmp fajl + rename)."""
    tmp_path = _CACHE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, _CACHE_PATH)


def _normalize_brand_key(brand_name: str) -> str:
    return brand_name.strip().lower()

def _clean_description(text: str) -> str:
    """Uklanja markdown i visak razmaka iz opisa (ulazi u embedding)."""
    text = re.sub(r"[*_#`]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def research_brand(brand_name: str, force_refresh: bool = False) -> dict:
    """Istrazuje brend preko web pretrage; rezultat se kesira po nazivu brenda. force_refresh zaobilazi kes."""
    cache_key = _normalize_brand_key(brand_name)
    cache = _load_cache()

    if not force_refresh and cache_key in cache:
        cached_result = dict(cache[cache_key])
        cached_result["from_cache"] = True
        return cached_result

    prompt = f"""Istrazi brend "{brand_name}" koristeci web pretragu.

Sastavi KRATAK opis (30-50 rijeci, STROGO) koji sadrzi SAMO:
- Industriju/kategoriju (1-3 rijeci, npr. "potrosacka elektronika")
- Konkretne proizvode/usluge koje brend prodaje, nabrojane kao
  imenice/imeničke fraze (npr. "pametni telefoni, laptopovi, tableti,
  pametni satovi, bezicne slusalice")
- Kljucne teme/oblasti povezane sa brendom (npr. "tehnologija,
  inovacija, dizajn proizvoda")

VAZNO ZA USLUGE/PLATFORME (za razliku od fizickih proizvoda): opisi
uslugu kroz AKTIVNOST/ISKUSTVO KORISNIKA, ne kroz interne poslovne ili
tehnicke termine kompanije. NE koristi rijeci poput "API", "metasearch",
"platforma za...", "algoritam", "tehnologija za..." - umjesto toga,
opisi sta korisnik STVARNO RADI kad koristi tu uslugu, istim jezikom
kojim bi to opisao neko ko je stvarno koristi (npr. za Skyscanner:
"pretrazivanje i poredjenje cijena avio karata, planiranje putovanja,
pronalazenje jeftinih letova" umjesto "metasearch, Search API, price
alerts").


NE UKLJUCUJ (ovo je namjerno iskljuceno, ne propust):
- Ciljnu demografiju ili psihografske segmente publike
- Filozofiju brendiranja, misiju, vrijednosti kompanije
- Ton komunikacije ili stil brenda
- Marketinske fraze, superlative ili prodajni jezik
- Istoriju ili nagrade brenda
- Markdown formatiranje (zvjezdice, crtice, naslove)
- Naziv brenda na pocetku odgovora

Piši GUSTO, kao listu kljucnih pojmova u recenici, ne kao marketinski
tekst. Cilj je da ovaj opis embedding model moze direktno da poredi sa
naslovima YouTube videa - konkretni, prepoznatljivi termini su
vazniji od elegantnog stila pisanja.

Primjer dobrog formata (za drugi brend, ilustrativno): "Prehrambena
industrija, konditorski proizvodi. Prodaje: keksi, cokoladni premazi,
sladoled, cokoladne tablice. Teme: pecenje, deserti, porodicni obroci,
doruccak."

Ako je brend nepoznat ili ne mozes naci dovoljno informacija, jasno to
navedi na pocetku odgovora, ali i dalje pokusaj da daš sto konkretniji
opis na osnovu dostupnih podataka."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
        messages=[{"role": "user", "content": prompt}],
    )

    text_parts = [block.text for block in response.content if block.type == "text"]
    description = _clean_description(" ".join(text_parts))
    
    result = {
        "brand_name": brand_name,
        "generated_description": description,
        "from_cache": False,
    }

    cache[cache_key] = {
        "brand_name": brand_name,
        "generated_description": description,
    }
    _save_cache(cache)

    return result


def clear_cached_brand(brand_name: str) -> bool:
    """Brise keširani zapis za brend, ako postoji."""
    cache_key = _normalize_brand_key(brand_name)
    cache = _load_cache()

    if cache_key in cache:
        del cache[cache_key]
        _save_cache(cache)
        return True
    return False


if __name__ == "__main__":
    result = research_brand("Apple")
    print("=== GENERISANI OPIS BRENDA (novi, konkretan format) ===")
    print(result["generated_description"])
    print(f"\nBroj rijeci: {len(result['generated_description'].split())}")
    print(f"Iz keša: {result['from_cache']}")

    print("\n=== DRUGI POZIV ZA ISTI BREND (treba biti iz keša) ===")
    result2 = research_brand("Apple")
    print(f"Isti opis: {result['generated_description'] == result2['generated_description']}")
    print(f"Iz keša: {result2['from_cache']}")