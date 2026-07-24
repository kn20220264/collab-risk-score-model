import { useState } from "react";

const API_BASE = "http://127.0.0.1:8000/api/v1/creators/youtube";

const RISK_COLORS = {
  "Nizak rizik": "#4fae7c",
  "Srednji rizik": "#e8a33d",
  "Visok rizik": "#e2554c",
};

function ScoreGauge({ score, category }) {
  const r = 80;
  const circumference = Math.PI * r;
  const offset = circumference * (1 - score / 100);
  const color = RISK_COLORS[category] || "#8b93a1";

  return (
    <div className="gauge-wrap">
      <svg viewBox="0 0 200 110" className="gauge">
        <path
          d="M20,100 A80,80 0 0 1 180,100"
          fill="none"
          stroke="#2a303c"
          strokeWidth="14"
          strokeLinecap="round"
        />
        <path
          d="M20,100 A80,80 0 0 1 180,100"
          fill="none"
          stroke={color}
          strokeWidth="14"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="gauge-fill"
        />
      </svg>
      <div className="gauge-score">
        <span className="gauge-number">{score}</span>
        <span className="gauge-max">/100</span>
      </div>
      <div className="gauge-category" style={{ color }}>
        {category}
      </div>
    </div>
  );
}

function ModuleBar({ label, value }) {
  return (
    <div className="module-row">
      <div className="module-label">{label}</div>
      <div className="module-track">
        <div className="module-fill" style={{ width: `${value}%` }} />
      </div>
      <div className="module-value">{value}</div>
    </div>
  );
}

function Analyzer() {
  const [handle, setHandle] = useState("");
  const [brand, setBrand] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  async function handleAnalyze(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const params = new URLSearchParams({
        brand_name: brand,
        include_explanation: "true",
      });
      const cleanHandle = handle.replace(/^@/, "");

      const response = await fetch(`${API_BASE}/${cleanHandle}?${params}`);

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Analiza nije uspjela.");
      }

      const data = await response.json();
      setResult(data.creator);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="analyzer-section" id="analyzer">
      <div className="section-inner">
        <div className="section-heading">
          <div className="eyebrow">Isprobaj uživo</div>
          <h2>Pokreni analizu u realnom vremenu</h2>
          <p className="section-subtitle">
            Unesi YouTube handle i naziv brenda — sistem automatski
            istražuje brend, prikuplja podatke sa YouTube-a, analizira
            sentiment komentara i brand-fit, te računa finalni skor rizika.
          </p>
        </div>

        <form className="input-card" onSubmit={handleAnalyze}>
          <div className="field">
            <label htmlFor="handle">YouTube handle</label>
            <input
              id="handle"
              type="text"
              placeholder="@mkbhd"
              value={handle}
              onChange={(e) => setHandle(e.target.value)}
              required
            />
          </div>
          <div className="field">
            <label htmlFor="brand">Naziv brenda</label>
            <input
              id="brand"
              type="text"
              placeholder="Npr. Nike"
              value={brand}
              onChange={(e) => setBrand(e.target.value)}
              required
            />
          </div>
          <button type="submit" className="btn btn-primary" disabled={loading}>
            {loading ? "Analiziram..." : "Pokreni analizu"}
          </button>
        </form>

        {loading && (
          <div className="status-note">
            Prikupljam podatke sa YouTube-a, analiziram sentiment i brand-fit
            — ovo moze potrajati do 30 sekundi.
          </div>
        )}

        {error && <div className="error-note">Greska: {error}</div>}

        {result && (
          <div className="results">
            <div className="channel-line">
              <strong>{result.display_name}</strong>
              <span>
                {result.subscriber_count.toLocaleString()}{" "}
                pretplatnika · {result.video_count.toLocaleString()}{" "}
                videa
              </span>
            </div>

            {result.brand_description_auto_generated && (
              <div className="status-note">
                Opis brenda automatski generisan istraživanjem naziva "
                {brand}": {result.brand_description_used}
              </div>
            )}

            <div className="results-grid">
              <div className="card gauge-card">
                <ScoreGauge score={result.score} category={result.risk_category} />
                {result.triggered_risk_flags.length > 0 && (
                  <div className="caps-warning">
                    <div className="caps-title">Kriticni nalazi</div>
                    {result.triggered_risk_flags.map((cap, i) => (
                      <div key={i} className="cap-item">
                        {cap}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="card modules-card">
                <div className="card-title">Skorovi po modulima</div>
                <ModuleBar
                  label="Kvantitativne metrike"
                  value={result.modules.quantitative}
                />
                <ModuleBar
                  label="Autenticnost"
                  value={result.modules.authenticity}
                />
                <ModuleBar label="Sentiment" value={result.modules.sentiment} />
                <ModuleBar label="Brand-fit" value={result.modules.brand_fit} />
              </div>
            </div>

            {result.ai_explanation && (
              <div className="card explanation-card">
                <div className="card-title">AI obrazlozenje</div>
                <p className="explanation-text">{result.ai_explanation}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}

export default Analyzer;
