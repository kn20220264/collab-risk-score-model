function Header() {
  return (
    <header className="site-header">
      <div className="site-header-inner">
        <a href="#top" className="logo">
          <span className="logo-mark">CRS</span>
          <span className="logo-word">Collab Risk Score</span>
        </a>

        <nav className="site-nav">
          <a href="#how-it-works">Kako radi</a>
          <a href="#methodology">Metodologija</a>
          <a href="#analyzer">Analiza</a>
        </nav>

        <a href="#analyzer" className="btn btn-primary btn-small">
          Pokreni analizu
        </a>
      </div>
    </header>
  );
}

export default Header;
