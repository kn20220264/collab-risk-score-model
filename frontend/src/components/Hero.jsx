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
  return (
    <section className="hero" id="top">
      <div className="hero-inner">
        <div className="hero-copy">
          <div className="eyebrow">Procjena rizika brend—kreator saradnje</div>
          <h1>
            Znaj tačno s kim sarađuješ, <span className="accent-text">prije potpisa</span>.
          </h1>
          <p className="hero-subtitle">
            Collab Risk Score analizira YouTube kreatora kroz četiri nezavisna
            modula — kvantitativne metrike, autentičnost, sentiment i
            brand-fit — i vraća transparentan skor rizika od 0 do 100, sa
            AI obrazloženjem.
          </p>
          <div className="hero-actions">
            <a href="#analyzer" className="btn btn-primary">
              Analiziraj kreatora
            </a>
            <a href="#methodology" className="btn btn-ghost">
              Pogledaj metodologiju
            </a>
          </div>
          <div className="hero-chips">
            <span className="chip">4 nezavisna modula</span>
            <span className="chip">AHP + entropijske težine</span>
            <span className="chip">Otvorena metodologija</span>
          </div>
        </div>

        <div className="hero-visual">
          <PreviewCard />
        </div>
      </div>
    </section>
  );
}

export default Hero;
