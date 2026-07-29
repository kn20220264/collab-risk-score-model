"""
Modul za automatsko istrazivanje brenda na osnovu samo naziva.

Problem koji rjesava: brand-fit modul (brand_fit.py) racuna cosine
similarity izmedju embeddinga OPISA BRENDA i sadrzaja kanala. Ako
korisnik unese samo kratak/nepotpun opis (ili samo naziv), embedding
tog teksta nosi premalo informacija da bi poredjenje bilo smisleno.

Rjesenje: koristimo Claude API sa ugradjenim web_search alatom da
AUTOMATSKI sastavimo opis brenda - koji se zatim koristi kao ulaz za
brand_fit.calculate_brand_fit_score, umjesto korisnikovog sirovog teksta.

VAZNA ISPRAVKA (otkrivena testiranjem, dokumentovano u poglavlju 4.3):
Prva verzija ovog modula generisala je opise optimizovane za CITLJIVOST
COVJEKU - ukljucivali su ciljnu demografiju, psihografske segmente,
filozofiju brendiranja i ton komunikacije (npr. "brend kao zivotni
stil", "cist, elegantan i nenametljiv ton"). Testiranjem na parovima
MKBHD+Apple (teorijski gotovo idealan brand-fit slucaj: vodeci tech
reviewer + najveci tech brend) pokazalo se da ovakav apstraktan,
marketinski jezik daje NIZU cosine similarity nego kraci, konkretniji
opis - jer se sadrzaj YouTube kanala (naslovi videa, opis kanala)
sastoji od KONKRETNIH, PROIZVODNO-ORIJENTISANIH termina (npr. "iPhone
15 Review"), koji se semanticki bolje poklapaju sa isto tako konkretnim
opisom proizvoda nego sa apstraktnim opisom brend-filozofije.

Ovo je operacionalizacija poznatog principa iz IR/NLP literature: veca
duzina teksta sa vise raznorodnih tema (demografija + ton + filozofija)
"razblazuje" (dilutes) semanticki signal usrednjavanjem, cineci
embedding manje fokusiranim na kljucne, prepoznatljive koncepte.

Zbog toga je prompt ispod izmijenjen da generise KRACI, PROIZVODNO/
TEMATSKI fokusiran opis, optimizovan za embedding poredjenje, a ne za
citljivost coyjeku. AI obrazlozenje (explanation_service.py) i dalje
moze da koristi bogatiji, opisniji jezik - ali SAMO opis koji ulazi u
embedding proracun je sada sveden na jezgro: kategorija, konkretni
proizvodi/usluge, kljucne teme.

VAZNA ISPRAVKA #2 - REPRODUCIBILNOST (otkrivena testiranjem VOSTCAST +
Red Bull, poglavlje 4.3 - poznati bug prenesen iz ranijih sesija):
Ranija verzija ovog modula pozivala je Claude API SVAKI PUT IZNOVA za
isti naziv brenda, bez ikakvog cuvanja rezultata. Buduci da generisanje
teksta preko LLM-a nije 100% deterministicko (cak i uz istu instrukciju,
formulacija/redoslijed recenica moze blago varirati izmedju poziva),
ovo je proizvodilo RAZLICITE opise brenda za IDENTICAN naziv, sto je
dalje davalo razlicite embedding vektore i, posljedicno, razlicite
brand-fit skorove za identican kanal+brend par u uzastopnim pozivima
(dokumentovano: VOSTCAST+Red Bull je u tri uzastopna poziva dao brand-fit
skorove 75.18, 99.71 i 89.38 - razlika od skoro 25 poena na identicnom
ulazu).

Ovo je ozbiljan problem test-retest pouzdanosti (reliability) za alat
koji sluzi za podrsku odlucivanju - model koji daje razlicite rezultate
za identican upit tesko je braniti kao pouzdan.

Rjesenje: generisan opis brenda se sad KESIRA (cache) po normalizovanom
nazivu brenda (lowercase, trim razmaka), u jednostavan JSON fajl na
disku. Prvi poziv za dati brend generise i cuva opis; svaki naredni
poziv za ISTI naziv brenda vraca vec sacuvan opis, umjesto da ga
generise iznova. Ovo garantuje da isti naziv brenda uvijek proizvodi
identican brand-fit skor za dati kanal (potpuna reproducibilnost),
dok se generisanje iznova moze eksplicitno zatraziti (force_refresh=True)
ako korisnik zeli da osvjezi/ispravi opis (npr. nakon rucne provjere ili
promjene u brendu tokom vremena).

Napomena: keširanje na nivou fajla je namjerno jednostavno rjesenje
(MVP nivo) - dovoljno za potrebe ovog rada i test-seta kanala. Za
produkcionu upotrebu sa vise konkurentnih korisnika/procesa, bila bi
potrebna prava baza podataka (npr. SQLite/Postgres) sa transakcionim
zakljucavanjem, sto je van obima ovog rada.
"""

import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-sonnet-4-6"

_CACHE_PATH = os.path.join(os.path.dirname(__file__), "brand_profile_cache.json")


def _load_cache() -> dict:
    """
    Ucitava keš brand profila sa diska. Ako fajl ne postoji (prvo
    pokretanje aplikacije), vraca prazan rjecnik - bezbjedan fallback.
    """
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            # Osteceni/nekompletan fajl (npr. prekid pisanja usred
            # procesa) - tretiramo kao prazan kes umjesto da srusimo
            # aplikaciju; sledeci uspjesan _save_cache ce ga popraviti.
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    """
    Cuva ažurirani keš na disk. Piše u privremeni fajl pa ga preimenuje
    (atomic write) - izbjegava da eventualni pad procesa nasred pisanja
    ostavi napola upisan, neispravan JSON fajl.
    """
    tmp_path = _CACHE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, _CACHE_PATH)


def _normalize_brand_key(brand_name: str) -> str:
    """
    Normalizuje naziv brenda u kljuc za keš (lowercase, trim), tako da
    "Red Bull", "red bull" i " Red Bull " pogode isti keširani zapis.
    """
    return brand_name.strip().lower()


def research_brand(brand_name: str, force_refresh: bool = False) -> dict:
    """
    Istrazuje brend preko web pretrage i sastavlja KRATAK,
    PROIZVODNO/TEMATSKI fokusiran opis pogodan za embedding-based
    poredjenje sa sadrzajem kanala.

    Rezultat se KEŠIRA po normalizovanom nazivu brenda - prvi poziv za
    dati brend generise i cuva opis; svaki naredni poziv sa istim
    nazivom vraca sacuvani opis umjesto da ga generise iznova, cime se
    garantuje reproducibilnost brand-fit skora za isti kanal+brend par
    (vidi napomenu o reproducibilnosti na vrhu fajla).

    Parametri:
        brand_name: naziv brenda koji se istrazuje.
        force_refresh: ako je True, zaobilazi keš i generise nov opis
            preko API poziva, cak i ako vec postoji keširan zapis za
            ovaj naziv (npr. korisnik zeli da rucno osvjezi/ispravi
            opis, ili je brend promijenio ponudu proizvoda tokom
            vremena).

    Vraca dict sa generisanim opisom i indikatorom da li je rezultat
    dosao iz keša (from_cache), radi transparentnosti u API odgovoru.
    """
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
    description = " ".join(text_parts).strip()

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
    """
    Uklanja keširani zapis za dati naziv brenda, ako postoji. Korisno
    za rucno ciscenje pogresno generisanog opisa bez brisanja citavog
    kesa. Vraca True ako je zapis pronadjen i uklonjen, False inace.
    """
    cache_key = _normalize_brand_key(brand_name)
    cache = _load_cache()

    if cache_key in cache:
        del cache[cache_key]
        _save_cache(cache)
        return True
    return False


# Brzi test
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