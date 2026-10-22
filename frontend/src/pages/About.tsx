export default function About() {
  return (
    <>
      <h1>About this release</h1>
      <section className="card">
        <h2>Included</h2>
        <ul>
          <li>
            A backend that forecasts drought for Rajasthan and Barmer, reports observed heat, and assesses
            crop impact, with each forecast month labelled by its own measured test result.
          </li>
          <li>A mechanical check of every number in a written report against its sources.</li>
          <li>
            Demo mode: six saved examples that run with no key and no network. Live mode, when you run the
            backend with your own Gemini key.
          </li>
          <li>This website, with an Evidence page whose numbers are read from saved result files.</li>
          <li>
            The small language model's code, configs and tests (in <code>slm/</code>), its evaluation results
            (see the SLM page), and a link to the one published adapter on Hugging Face. The adapter is
            downloaded when the service loads it; no weights are stored in this repository.
          </li>
        </ul>
      </section>
      <section className="card">
        <h2>Not included yet</h2>
        <ul>
          <li>
            A small-language-model service that starts by default. It is an optional container profile whose
            image is untested. The real 1.5B SFT adapter was served on the author's Windows laptop with the hf
            backend in a two-request smoke test, not a benchmark. The 0.5B and DPO adapters are not published.
          </li>
          <li>
            A proven benefit of the small language model for the analyst. An optional passage check exists
            (off by default, shown separately, not used to write the report); its effect has not been measured.
          </li>
          <li>One-command container setup (planned, not part of this site's first version).</li>
          <li>A 3D view.</li>
        </ul>
        <p className="muted small">
          The site does not claim that the small language model improves the analyst, and the saved
          examples are snapshots: asking again in live mode can give different text.
        </p>
      </section>
      <section className="card">
        <h2>Data and credit</h2>
        <p>
          Weather data by <a href="https://open-meteo.com/">Open-Meteo.com</a>, licensed{" "}
          <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. The files used here were
          changed: daily values were aggregated to monthly. ENSO and Indian Ocean Dipole series are from
          the NOAA Physical Sciences Laboratory. Full details, with file hashes, are in the data snapshot
          manifest in the repository.
        </p>
      </section>
    </>
  );
}
