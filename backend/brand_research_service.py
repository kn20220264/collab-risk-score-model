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
"""

import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-sonnet-4-6"


def research_brand(brand_name: str) -> dict:
    """
    Istrazuje brend preko web pretrage i sastavlja KRATAK,
    PROIZVODNO/TEMATSKI fokusiran opis pogodan za embedding-based
    poredjenje sa sadrzajem kanala.

    Vraca dict sa generisanim opisom, radi transparentnosti u API
    odgovoru.
    """
    prompt = f"""Istrazi brend "{brand_name}" koristeci web pretragu.

Sastavi KRATAK opis (30-50 rijeci, STROGO) koji sadrzi SAMO:
- Industriju/kategoriju (1-3 rijeci, npr. "potrosacka elektronika")
- Konkretne proizvode/usluge koje brend prodaje, nabrojane kao
  imenice/imeničke fraze (npr. "pametni telefoni, laptopovi, tableti,
  pametni satovi, bezicne slusalice")
- Kljucne teme/oblasti povezane sa brendom (npr. "tehnologija,
  inovacija, dizajn proizvoda")

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

    return {
        "brand_name": brand_name,
        "generated_description": description,
    }


# Brzi test
if __name__ == "__main__":
    result = research_brand("Apple")
    print("=== GENERISANI OPIS BRENDA (novi, konkretan format) ===")
    print(result["generated_description"])
    print(f"\nBroj rijeci: {len(result['generated_description'].split())}")