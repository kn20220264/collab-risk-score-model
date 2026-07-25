import { useState } from "react";
import CreatorProfileCard from "./CreatorProfileCard";

const API_BASE = "http://127.0.0.1:8000/api/v1/creators/youtube";

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
            {result.brand_description_auto_generated && (
              <div className="status-note">
                Opis brenda automatski generisan istraživanjem naziva "
                {brand}": {result.brand_description_used}
              </div>
            )}

            {result.brand_fit_warning && (
              <div className="warning-note">{result.brand_fit_warning}</div>
            )}

            <CreatorProfileCard creator={result} />
          </div>
        )}
      </div>
    </section>
  );
}

export default Analyzer;
