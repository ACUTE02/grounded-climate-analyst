import { N } from "../components/Num";
import { ADAPTER_URL, REPO_URL } from "../components/Layout";
import aboutJson from "../data/about.json";
import type { About as AboutData } from "../data/types";

const about = aboutJson as unknown as AboutData;

export default function About() {
  return (
    <>
      <div className="page-head">
        <p className="eyebrow">About</p>
        <h1>What this release is, and what it is not.</h1>
      </div>

      <section className="card">
        <h2>Included</h2>
        <ul>
          <li>
            A backend that forecasts drought for Rajasthan and Barmer, reports observed heat, and assesses crop
            impact, with each forecast month labelled by its own measured test result.
          </li>
          <li>A mechanical check of every number in a written report against its sources.</li>
          <li>
            Demo mode: six saved examples that run with no key and no network. Live mode, when you run the backend
            with your own Gemini key.
          </li>
          <li>This website, with an Evidence page whose numbers are read from saved result files.</li>
          <li>
            The small language model's code, configs and tests, its evaluation results, and a link to the one
            published adapter. The adapter is downloaded when the service loads it; no weights are stored here.
          </li>
        </ul>
      </section>

      <section className="card" id="limits">
        <h2>Limits</h2>
        <ul>
          <li>
            The retrieval index in this release has <N n={about.corpus_chunks} /> chunks. The retrieval scores on the
            Evidence page were measured on an earlier, larger corpus and are not reproduced by this one.
          </li>
          <li>
            One live report was run on the author's machine for the first saved question. Its number check came back
            clean; one writing attempt hit a temporary "high demand" error from the model service and the pipeline's
            second attempt produced the report. One report is an example, not a measurement.
          </li>
          <li>
            The small language model service is optional and not started by default. The real 1.5B SFT adapter was
            served on the author's Windows laptop with the hf backend in a two-request smoke test: it answered one
            answerable question and refused one unanswerable one. That is not a benchmark. The 0.5B and DPO adapters
            are not published.
          </li>
          <li>
            A benefit of the small language model for the analyst is not shown. An optional passage check exists (off
            by default, shown separately, not used to write the report); its effect has not been measured.
          </li>
          <li>Saved examples are snapshots: asking again in live mode can give different text. There is no 3D view.</li>
        </ul>
      </section>

      <section className="card">
        <h2>Data and attribution</h2>
        <p>
          Weather data by <a href="https://open-meteo.com/">Open-Meteo.com</a>, licensed{" "}
          <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>; daily values were aggregated to
          monthly. Open-Meteo's terms limit the free API to non-commercial use. Contains modified Copernicus Climate
          Change Service information 2026.
        </p>
        <p>
          ENSO (ONI) and Indian Ocean Dipole (DMI) series were obtained from the NOAA Physical Sciences Laboratory.
          The DMI is computed from HadISST1.1: HadISST data were obtained from the Met Office and are © British Crown
          Copyright, Met Office, 2003, provided under an Open Government Licence. Citation: Rayner et al. (2003),
          J. Geophys. Res. 108(D14), 4407, doi:10.1029/2002JD002670.
        </p>
        <p className="small muted">Full wording and file hashes: NOTICE and the data snapshot manifest in the repository.</p>
      </section>

      <section className="card">
        <h2>Licence, credit and links</h2>
        <p>
          The code is licensed under Apache-2.0. If you reuse this work, please credit Ayushmaan Gupta and link back
          to the repository.
        </p>
        <ul>
          <li>
            Repository: <a href={REPO_URL}>{REPO_URL}</a>
          </li>
          <li>
            Hugging Face adapter: <a href={ADAPTER_URL}>GHOSTBHAIYA/climate-slm-1.5b-sft</a> (Apache-2.0, base model
            Qwen/Qwen2.5-1.5B-Instruct)
          </li>
        </ul>
      </section>
    </>
  );
}
