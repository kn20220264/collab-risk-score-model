import { useEffect, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

const MODULE_INFO = {
  quantitative: {
    title: "Kvantitativne metrike",
    description:
      "Engagement rate i stabilnost pregleda kroz posljednje objave — signal koliko je publika dosljedno aktivna.",
  },
  authenticity: {
    title: "Autentičnost",
    description:
      "Odnos pretplatnika i prosječnih pregleda — otkriva sumnju na kupljenu ili neaktivnu publiku.",
  },
  sentiment: {
    title: "Sentiment",
    description:
      "AI analiza tona komentara ispod nedavnih videa — direktan pokazatelj reputacionog rizika.",
  },
  brand_fit: {
    title: "Brand-fit",
    description:
      "Semantičko poklapanje sadržaja kanala sa opisom brenda, preko embedding sličnosti.",
  },
};

const RISK_CAP_DISPLAY = [
  { name: "Izrazito negativan sentiment", cap: 40 },
  { name: "Sumnjiv odnos pretplatnici/pregledi", cap: 35 },
  { name: "Vrlo slabo brand-fit poklapanje", cap: 50 },
];

function Methodology() {
  const [methodology, setMethodology] = useState(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/api/v1/methodology`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!cancelled && data) setMethodology(data);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  const weights = methodology?.ahp?.weights;
  const cr = methodology?.ahp?.consistency_ratio;

  return (
    <section className="methodology-section" id="how-it-works">
      <div className="section-inner">
        <div className="section-heading">
          <div className="eyebrow">Kako se računa skor</div>
          <h2>Četiri nezavisna modula, ponderisana AHP metodom</h2>
          <p className="section-subtitle">
            Svaki modul se boduje 0–100, a zatim se kombinuje ponderisanim
            zbirom čije težine su matematički izvedene iz pairwise comparison
            matrice (AHP), a ne proizvoljno upisane.
          </p>
        </div>

        <div className="module-grid">
          {Object.entries(MODULE_INFO).map(([key, info]) => (
            <div className="card module-card" key={key}>
              <div className="module-card-top">
                <span className="module-badge">
                  {key === "quantitative" && "Q"}
                  {key === "authenticity" && "A"}
                  {key === "sentiment" && "S"}
                  {key === "brand_fit" && "B"}
                </span>
                {weights && (
                  <span className="module-weight">
                    {Math.round(weights[key] * 100)}%
                  </span>
                )}
              </div>
              <div className="card-title module-card-title">{info.title}</div>
              <p className="module-card-desc">{info.description}</p>
            </div>
          ))}
        </div>

        {typeof cr === "number" && (
          <p className="ahp-note">
            AHP consistency ratio (CR) = {cr.toFixed(3)} —{" "}
            {methodology.ahp.is_consistent
              ? "matrica poređenja je konzistentna (CR < 0.10)."
              : "matrica poređenja je iznad praga konzistentnosti."}
          </p>
        )}

        <div id="methodology" className="card risk-cap-card">
          <div className="card-title">Risk cap mehanizam</div>
          <p className="risk-cap-desc">
            Ponderisani zbir sam po sebi ne dočarava kritične nalaze — zato
            se primjenjuje i nekompenzatorni ("conjunctive") risk cap: ako
            bilo koja pojedinačna metrika upadne u kritičnu zonu, finalni
            skor se ograničava nezavisno od ostalih modula.
          </p>
          <div className="risk-cap-rules">
            {RISK_CAP_DISPLAY.map((rule) => (
              <div className="risk-cap-rule" key={rule.name}>
                <span>{rule.name}</span>
                <span className="risk-cap-value">max {rule.cap}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="transparency-callout">
          <p>
            Za razliku od komercijalnih alata poput CreatorScore ili
            HypeAuditor, čiji algoritmi nisu javno objavljeni, cijela
            metodologija ovog prototipa — AHP matrica, izračunate težine i
            risk cap pravila — dostupna je otvoreno preko API-ja.
          </p>
          <a
            href={`${API_BASE}/api/v1/methodology`}
            target="_blank"
            rel="noreferrer"
            className="btn btn-ghost btn-small"
          >
            Pogledaj punu metodologiju (JSON)
          </a>
        </div>
      </div>
    </section>
  );
}

export default Methodology;
