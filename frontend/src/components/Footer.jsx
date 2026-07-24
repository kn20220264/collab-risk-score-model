function Footer() {
  return (
    <footer className="site-footer">
      <div className="section-inner footer-inner">
        <div className="footer-brand">
          <span className="logo-mark">CRS</span>
          <p>
            Collab Risk Score je akademski prototip izrađen u okviru
            diplomskog rada, sa strukturom API-ja inspirisanom alatom
            CreatorScore radi poređenja metodologija.
          </p>
        </div>

        <div className="footer-links">
          <div className="footer-col">
            <span className="footer-col-title">Proizvod</span>
            <a href="#how-it-works">Kako radi</a>
            <a href="#methodology">Metodologija</a>
            <a href="#analyzer">Analiza uživo</a>
          </div>
          <div className="footer-col">
            <span className="footer-col-title">API</span>
            <a
              href="http://127.0.0.1:8000/docs"
              target="_blank"
              rel="noreferrer"
            >
              API dokumentacija
            </a>
            <a
              href="http://127.0.0.1:8000/api/v1/methodology"
              target="_blank"
              rel="noreferrer"
            >
              Metodologija (JSON)
            </a>
          </div>
        </div>
      </div>
      <div className="footer-bottom">
        © {new Date().getFullYear()} Collab Risk Score — akademski prototip.
      </div>
    </footer>
  );
}

export default Footer;
