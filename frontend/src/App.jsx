import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000/analyze";

const RISK_COLORS = {
  "Nizak rizik": "#4FAE7C",
  "Srednji rizik": "#E8A33D",
  "Visok rizik": "#E2554C",
};

function ScoreGauge({ score, category }) {
  const r = 80;
  const circumference = Math.PI * r;
  const offset = circumference * (1 - score / 100);
  const color = RISK_COLORS[category] || "#8B93A1";

  return (
    <div className="gauge-wrap">
      <svg viewBox="0 0 200 110" className="gauge">
        <path
          d="M20,100 A80,80 0 0 1 180,100"
          fill="none"
          stroke="#2A303C"
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

function App() {
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
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          channel_handle: handle,
          brand_description: brand,
        }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Analiza nije uspjela.");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="eyebrow">Procjena rizika poslovne saradnje</div>
        <h1>Collab Risk Score</h1>
        <p className="subtitle">
          Unesi YouTube kanal i opis brenda za analizu rizika saradnje
        </p>
      </header>

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
          <label htmlFor="brand">Opis brenda</label>
          <textarea
            id="brand"
            placeholder="Npr. Tehnoloski brend koji prodaje pametne telefone i dodatnu opremu..."
            value={brand}
            onChange={(e) => setBrand(e.target.value)}
            rows={3}
            required
          />
        </div>
        <button type="submit" disabled={loading}>
          {loading ? "Analiziram..." : "Pokreni analizu"}
        </button>
      </form>

      {loading && (
        <div className="status-note">
          Prikupljam podatke sa YouTube-a, analiziram sentiment i brand-fit —
          ovo moze potrajati do 30 sekundi.
        </div>
      )}

      {error && <div className="error-note">Greska: {error}</div>}

      {result && (
        <div className="results">
          <div className="channel-line">
            <strong>{result.channel.title}</strong>
            <span>
              {result.channel.subscriber_count.toLocaleString()} pretplatnika ·{" "}
              {result.channel.video_count.toLocaleString()} videa
            </span>
          </div>

          <div className="results-grid">
            <div className="card gauge-card">
              <ScoreGauge
                score={result.risk_assessment.final_score}
                category={result.risk_assessment.risk_category}
              />
              {result.risk_assessment.triggered_caps.length > 0 && (
                <div className="caps-warning">
                  <div className="caps-title">Kriticni nalazi</div>
                  {result.risk_assessment.triggered_caps.map((cap, i) => (
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
                value={result.risk_assessment.module_scores.quantitative}
              />
              <ModuleBar
                label="Autenticnost"
                value={result.risk_assessment.module_scores.authenticity}
              />
              <ModuleBar
                label="Sentiment"
                value={result.risk_assessment.module_scores.sentiment}
              />
              <ModuleBar
                label="Brand-fit"
                value={result.risk_assessment.module_scores.brand_fit}
              />
            </div>
          </div>

          <div className="card explanation-card">
            <div className="card-title">AI obrazlozenje</div>
            <p className="explanation-text">{result.ai_explanation}</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;