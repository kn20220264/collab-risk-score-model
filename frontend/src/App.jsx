import "./App.css";
import Header from "./components/Header";
import Hero from "./components/Hero";
import Methodology from "./components/Methodology";
import Analyzer from "./components/Analyzer";
import Footer from "./components/Footer";

function App() {
  return (
    <div className="app">
      <Header />
      <Hero />
      <Methodology />
      <Analyzer />
      <Footer />
    </div>
  );
}

export default App;
