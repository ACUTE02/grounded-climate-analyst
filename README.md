# Climate Suite

A climate risk analyst for Rajasthan and Barmer (India). It forecasts drought, reports observed
heat, and assesses crop impact, and it shows **how far each number can be trusted**: every
forecast month carries its own measured test result, every number in a written report is
checked mechanically against a source, and what the system does not know is stated, not hidden.

This repository holds the backend (Python, FastAPI), the website (React) and a demo mode that
runs with no key and no network.

## Run the demo (no key, no network after the build)

With Docker:

```
git clone https://github.com/ACUTE02/grounded-climate-analyst
cd <folder where you cloned>
docker compose up --build
```

Then open http://localhost:8080. Only the website's port is published. A cold build downloads base images and
packages and can take several minutes: on the author's machine the first build took more than 10 minutes and a
second build that had to run pip again took 310 seconds; later builds reuse cached layers and take seconds.

Without Docker (the Docker image uses Python 3.11.9 and builds the website with Node 24.16.0; the non-Docker install
also worked once on Python 3.14.5 with the lock file, in about 6 minutes; nothing else was tried and nothing is guaranteed):

```
cd backend
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt -c requirements.lock.txt
uvicorn api.app:app --port 8000
```

and in a second terminal:

```
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173. Demo mode answers only the six saved examples; any other
question gets a plain refusal (HTTP 409).

## Run live (your own key)

Live mode runs the real pipeline: tool routing and the written report use the Gemini API, and the retrieval
index uses its embedding API. You need your own free key (https://aistudio.google.com/apikey) and a network
connection. The app counts its own generation calls against a budget of 20 a day and 5 a minute (the free tier when
this was built); one report costs 2 to 4 of them. Live mode inside Docker is not supported: the container has no models
or index. Demo mode needs none of this.

Windows PowerShell (from the folder where you cloned):

```
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -c requirements.lock.txt
copy .env.example .env
```

macOS / Linux (bash):

```
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -c requirements.lock.txt
cp .env.example .env
```

Then, in both:

1. Open `backend/.env` in an editor (it is gitignored), put your key in `GEMINI_API_KEY` and set
   `CLIMATE_SUITE_MODE=live`.
2. `python -m scripts.setup` (still in `backend/`). It restores the input data from the checked-in snapshot (no network)
   and builds the forecast models; without a key it skips the retrieval index, says so, and ends with "Setup incomplete".
   With a key it also builds the index: that step downloads the corpus pages and calls the embedding API, so it needs the
   network and the key. Some source pages (the NDMA ones) have returned 404 or 503, so the index you get can be smaller
   than the author's, which had 101 chunks. The time and the number of embedding calls were not measured.
   `--refetch` fetches fresh weather data instead of the snapshot; the numbers can then differ slightly from the
   published evidence.
3. `uvicorn api.app:app --port 8000`
4. In a second terminal: `cd frontend`, `npm install`, `npm run dev`, then open http://localhost:5173 and type a
   question on the Ask page.

If the key is missing the backend says so in its start-up log, and a question returns a report marked "REPORT NOT
GENERATED". A daily-quota error comes back as HTTP 429 with an explanation. The saved examples are snapshots: asking the
same question live can give different text. One live report was run on the author's machine (its calls and grounding
result are in the author's run report, not in this repository).

## What the numbers say

These are on the website's Evidence page; each is read from a saved file by a script, and the file
is named next to it.

- **Drought forecast skill** (1 minus the model's error divided by the error of always predicting
  the seasonal normal), `backend/models/horizon_manifest.json`:
  Rajasthan 0.2622 (t+1, validated), 0.0766 (t+2, weak), -0.0145 (t+3, no skill);
  Barmer 0.2053, 0.0438, -0.0489 (same order). Running `python -m scripts.setup` from the
  snapshot reproduces these exactly.
- **Heat stress:** forecasting was tested and has no usable skill at any horizon in either
  region, so only observed heat is reported (`backend/models/heat_manifest.json`).
- **The number checker** (`backend/evaluation/checker_eval_results.json`): 14 test cases, precision
  1.0 and recall 0.9286 on a hand-labelled set, with a documented known defect.
- **Retrieval** (Track A): measured on the original 224-chunk corpus (`backend/retrieval/eval_results.json`
  and `eval_results_hybrid.json`); plain dense search stays the default. The corpus built on a
  fresh clone differs (some source pages are not always reachable), so these numbers are not
  reproduced by it.

## About this release

Included: the backend, demo and live modes, the website, the data snapshot, the saved examples, and
the evaluation results of a separate small language model (shown on the website, quoted from that
project's own README).

The small language model (SLM): its code, configs, tests and the one published adapter are part of
this release under `slm/`. The adapter is public on Hugging Face
([GHOSTBHAIYA/climate-slm-1.5b-sft](https://huggingface.co/GHOSTBHAIYA/climate-slm-1.5b-sft),
Apache-2.0, on the base model Qwen/Qwen2.5-1.5B-Instruct, LoRA rank 16, alpha 32) and is downloaded
when the service loads it; no weights are stored here. Only this 1.5B adapter is published: the 0.5B
and DPO adapters whose results appear in the SLM README are not part of this release. The service is
optional (compose profile `slm`). The real 1.5B SFT adapter was downloaded from Hugging Face and served on
the author's Windows laptop with the hf backend. It answered one answerable question from its passage and
refused one unanswerable question. That was a two-request smoke test, not a benchmark.
The training data is not included in this release (the tests that need it skip when it is missing).
The Colab and benchmark scripts and the notebook are kept for reference only: the 0.5B and DPO
adapters they refer to were not published.

The analyst has an optional step that asks the small language model about the first retrieved passages
(`CLIMATE_SUITE_SLM=on` and `SLM_URL`; **off by default**). The SLM service and the backend both default to port 8000,
so start the SLM on another port (`PHASE5_PORT=8001`) and set `SLM_URL=http://127.0.0.1:8001` for the backend. Its answers are shown next to the report
and are not used to write it or to check its numbers. Its effect has not been measured, and no claim is
made that the small language model improves (or worsens) the analyst.

The website's SLM page has a "Try it" card: you paste a passage and a question and the backend forwards them to the
SLM service (`POST /slm/answer`, passage at most 4000 characters, question at most 500; nothing is stored or logged).
It works in demo mode too, because it uses no Gemini call, but only when `SLM_URL` is set and the service is running;
otherwise the card is off and says how to start it. In Docker, set `SLM_URL=http://host.docker.internal:8001` for an
SLM running on the host (empty by default). The model is small and experimental, and no claim is made about how
fast or how accurate it is.

Not included yet: a measurement of that step, and a 3D view.

Known limits: the saved examples are snapshots (asking again in live mode can give different text);
three of the four live-written examples carry a number-check "warning" caused by numbers in the
system's own notes (a count of characters, a date), which the checker flags on purpose; one example
(out of scope) has no written report; saved forecasts use data through December 2024 because the
data snapshot is fixed.

## Tests

```
cd backend && python -m pytest          # live tests are opt-in (RUN_LIVE_* variables)
cd frontend && npm test
```

## Data and credit

The data files in `backend/data_snapshot/` keep the terms of their sources, listed here and in `NOTICE`;
they are not under the Apache licence of the code. Each file's source, fetch date and SHA256 are in
`backend/data_snapshot/manifest.json`.

- **Weather:** Weather data by [Open-Meteo.com](https://open-meteo.com/), licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: the daily values were aggregated
  to monthly (and columns renamed). Open-Meteo describes its Historical Weather API as based on reanalysis
  datasets, with ERA5 as the main one; which dataset served each value was not recorded here.
  Contains modified Copernicus Climate Change Service information 2026. Neither the European Commission
  nor ECMWF is responsible for any use that may be made of the Copernicus information.
  Open-Meteo's terms say the free API may only be used for non-commercial purposes
  (https://open-meteo.com/en/terms); `--refetch` uses that free API.
- **ENSO (ONI):** obtained from NOAA PSL (NOAA's three-month running mean of ERSST.V5 anomalies in the
  Nino 3.4 region). Data provided by the NOAA/OAR/PSL, Boulder, Colorado, USA, from their Web site at
  https://psl.noaa.gov/. PSL says information on its servers is in the public domain unless annotated
  otherwise (https://psl.noaa.gov/disclaimer/).
- **Indian Ocean Dipole (DMI):** created by NOAA PSL from HadISST1.1 (its file header says so). HadISST data
  were obtained from https://www.metoffice.gov.uk/hadobs/hadisst/ and are (c) British Crown Copyright,
  Met Office, provided under an Open Government Licence. Citation: Rayner et al. (2003), J. Geophys. Res.
  108(D14), 4407, doi:10.1029/2002JD002670.
- **Small language model:** the adapter is on Hugging Face
  ([GHOSTBHAIYA/climate-slm-1.5b-sft](https://huggingface.co/GHOSTBHAIYA/climate-slm-1.5b-sft), Apache-2.0)
  on the base model Qwen/Qwen2.5-1.5B-Instruct (Apache-2.0).

If you reuse this work, please credit Ayushmaan Gupta and link back to the repository:
https://github.com/ACUTE02/grounded-climate-analyst

## Licence

Apache License 2.0: see `LICENSE`, and `NOTICE` for the credit request and third-party material.
