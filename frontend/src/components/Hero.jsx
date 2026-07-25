import { useState } from "react";
import CreatorProfileCard from "./CreatorProfileCard";

const API_BASE = "http://127.0.0.1:8000/api/v1/creators/youtube";

function PreviewCard() {
  return (
    <div className="preview-card" aria-hidden="true">
      <div className="preview-card-header">
        <div className="preview-avatar" />
        <div>
          <div className="preview-name">@primjer-kreator</div>
          <div className="preview-sub">2.4M pretplatnika · 312 videa</div>
        </div>
        <span className="pill pill-low">Nizak rizik</span>
      </div>

      <div className="preview-score-row">
        <div className="preview-score">
          <span className="preview-score-number">82</span>
          <span className="preview-score-max">/100</span>
        </div>
        <div className="preview-modules">
          <div className="preview-module">
            <span>Kvantitativne metrike</span>
            <div className="preview-track">
              <div className="preview-fill" style={{ width: "78%" }} />
            </div>
          </div>
          <div className="preview-module">
            <span>Autentičnost</span>
            <div className="preview-track">
              <div className="preview-fill" style={{ width: "91%" }} />
            </div>
          </div>
          <div className="preview-module">
            <span>Sentiment</span>
            <div className="preview-track">
              <div className="preview-fill" style={{ width: "74%" }} />
            </div>
          </div>
          <div className="preview-module">
            <span>Brand-fit</span>
            <div className="preview-track">
              <div className="preview-fill" style={{ width: "68%" }} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Hero() {
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
    <section className={`hero${result ? "" : " hero-centered"}`} id="top">
      <div className="hero-inner">
        <div className="hero-copy">
          <div className="eyebrow">Procjena rizika brend—kreator saradnje</div>
          <h1>
            Znaj tačno s kim sarađuješ, <span className="accent-text">prije potpisa</span>.
          </h1>
          <p className="hero-subtitle">
            Unesi YouTube kreatora i brend — dobijaš skor rizika od 0 do 100,
            sa AI obrazloženjem zašto.
          </p>
        </div>

        <div className="hero-visual">
          <PreviewCard />
        </div>
      </div>

      <div className="hero-form-row">
        <form className="hero-form" onSubmit={handleAnalyze}>
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
          <div className="status-note status-note-left">
            Prikupljam podatke sa YouTube-a, analiziram sentiment i brand-fit
            — ovo moze potrajati do 30 sekundi.
          </div>
        )}

        {error && <div className="error-note error-note-left">Greska: {error}</div>}
      </div>

      {result && (
        <div className="results section-inner">
          {result.brand_description_auto_generated && (
            <div className="brand-desc-card">
              <div className="brand-desc-label">
                Opis brenda automatski generisan istraživanjem naziva "{brand}"
              </div>
              <p className="brand-desc-text">{result.brand_description_used}</p>
            </div>
          )}

          {result.brand_fit_warning && (
            <div className="warning-note">{result.brand_fit_warning}</div>
          )}

          <CreatorProfileCard creator={result} />
        </div>
      )}
    </section>
  );
}

export default Hero;
