# 🌾 Climate Suite

**A climate risk analyst for Rajasthan and Barmer (India) that forecasts drought, reports observed heat and assesses crop impact — and shows how far each number can be trusted.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-Website-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Gemini](https://img.shields.io/badge/Google-Gemini%20API-4285F4?logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![Docker](https://img.shields.io/badge/Docker-Demo%20mode-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-pytest%20%2B%20frontend-informational)](#-testing)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](#-license)

> "Validated, weak / directional, no skill" — every forecast month carries its own measured test result, every number in a written report is checked mechanically against a source, and what the system does not know is stated, not hidden.

---

## 📖 Overview

**Climate Suite** is an end-to-end climate-risk analyst for India's drought-prone Rajasthan region and the Barmer district. It answers questions an agricultural extension officer would ask with a written report that is:

- **Honest about skill**, because each forecast horizon is labelled with its own measured result
- **Grounded**, meaning every number in a written report is checked by code that does not use an LLM
- **Honest about gaps**, because unavailable sources and missing data are shown, not filled in

This repository holds the backend (Python, FastAPI), the website (React), and a demo mode that runs with no key and no network. A small language model (SLM) service is included as an optional part.

## ✨ Features

- 🌧️ **Drought forecast with per-horizon labels** — t+1 validated, t+2 weak, t+3 no skill, each with its measured skill score
- 🌡️ **Observed heat only** — heat forecasting was tested and has no usable skill at any horizon in either region, so it is reported as a null result
- 🌾 **Crop-impact assessment** — part of the same report
- ✅ **Mechanical number checker** — every number in a written report is checked against a source, with its own measured precision and recall
- 📚 **Retrieval over a document corpus (Track A)** — plain dense search is the default; hybrid search and re-ranking were built, measured, and left off
- 🤖 **Optional small language model step** — asks a fine-tuned 1.5B model about the first retrieved passages; **off by default** and not used to write the report
- 🧪 **Demo mode** — runs with no key and no network, on six saved examples
- 🖥️ **Website** — Home, Analyst (saved examples, and free-text questions in live mode), SLM (a chat that answers only from the text it is given), Evidence (each number is read from a saved file) and About

## 🏗️ Architecture

```
                       User question (natural language)
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Backend (FastAPI)            │
                    │   tool routing via Gemini      │
                    └───────────────┬────────────────┘
                                    │
              ┌──────────┬──────────┼───────────┬────────────┐
              ▼          ▼          ▼           ▼            ▼
        ┌──────────┐┌─────────┐┌──────────┐┌──────────┐┌───────────┐
        │ Drought  ││  Heat   ││ Retrieval││   Crop   ││ Data      │
        │ forecast ││(observed││ (plain   ││  impact  ││ snapshot  │
        │          ││  only)  ││  dense)  ││          ││           │
        └────┬─────┘└────┬────┘└────┬─────┘└────┬─────┘└─────┬─────┘
             └───────────┴──────────┴───────────┴────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  Report writer (Gemini)        │
                    └───────────────┬────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │  Mechanical number checker     │
                    │  (no LLM — checks every number │
                    │   against a source)            │
                    └───────────────┬────────────────┘
                                    ▼
                          Report shown on the website

        Optional, off by default:  first retrieved passages ──► SLM service
        (its answer is shown next to the report; it does not write the
         report and does not check its numbers)
```

In **demo mode** there is no live routing or writing: the website shows six saved examples, and any other question gets a plain refusal (HTTP 409).

## 📁 Project Structure

```
grounded-climate-analyst/
├── backend/
│   ├── api/                # FastAPI app
│   ├── demo/               # The six saved examples (demo mode)
│   ├── retrieval/          # Retrieval and its evaluation results
│   ├── evaluation/         # Number-checker evaluation
│   ├── models/             # Forecast manifests (the evidence files)
│   ├── data_snapshot/      # Fixed input data, with manifest.json (source, date, SHA256)
│   └── scripts/            # setup (python -m scripts.setup)
├── frontend/               # React website
├── slm/                    # Small language model service: code, configs, tests
├── docker-compose.yml      # Demo mode (profile "slm" for the optional SLM service)
├── LICENSE
└── NOTICE
```

## 🛠️ Tech Stack

`Python 3.11` · `FastAPI` · `React` · `Google Gemini API` · `Docker` · `Hugging Face (SLM adapter)` · `Open-Meteo` · `NOAA PSL`

## 📊 Measured Results (honest, as tested)

Each number below is on the website's Evidence page, read from a saved file by a script.

| What | Result |
|---|---|
| Drought forecast skill, Rajasthan | t+1 **0.2622** (validated), t+2 0.0766 (weak), t+3 −0.0145 (no skill) |
| Drought forecast skill, Barmer | t+1 **0.2053** (validated), t+2 0.0438 (weak), t+3 −0.0489 (no skill) |
| Heat-stress forecasting | No usable skill at any horizon in either region; only observed heat is reported |
| Number checker | 14 test cases, precision **1.0**, recall **0.9286** on a hand-labelled set, with a documented known defect |
| Retrieval | Plain dense search stays the default; hybrid and re-ranking measured and kept off |

Skill is 1 minus the model's error divided by the error of always predicting the seasonal normal. Files: `backend/models/horizon_manifest.json`, `backend/models/heat_manifest.json`, `backend/evaluation/checker_eval_results.json`, `backend/retrieval/eval_results.json` and `eval_results_hybrid.json`. Running `python -m scripts.setup` from the snapshot reproduces the forecast numbers exactly.

### 🔎 Retrieval: a negative result, kept on purpose

On an easier document-level test (12 queries, top 5) plain search finds the right document every time. On a stricter paragraph-level test against hand-labelled paragraphs (25 questions: 12 regular and 13 hard) top-5 precision was lower, and hybrid search and re-ranking did not help:

| Setup | Top-5 precision (easier / harder questions) |
|---|---|
| Plain dense search (default) | 53% / 37% |
| Plain + re-ranking | 48% / 29% (worse) |
| Hybrid (BM25 + dense) | 50% / 39% (about the same) |
| Hybrid + re-ranking | 48% / 29% (worse) |

These were measured on the original 224-chunk corpus. The corpus built on a fresh clone differs (some source pages are not always reachable), so a fresh clone does not reproduce them.

### 🤖 Small language model (separate project, quoted from its own README)

A Qwen2.5-1.5B-Instruct model with a LoRA adapter, tuned to answer from a passage or reply exactly "I cannot answer this from the provided context."

| Model (test split, 43 items) | Token F1 (95% interval) | False refusals | Correct refusals |
|---|---|---|---|
| 1.5B untuned | 0.656 | 0 / 32 | 6 / 11 |
| 1.5B + LoRA SFT (the published adapter) | **0.791** [0.688, 0.882] | 0 / 32 | **11 / 11** |

The test set is small (only 11 refusal items), so the margins are wide. The effect of the SLM step on this analyst has **not been measured**, and no claim is made that it improves or worsens the analyst.

## 🚀 Run the Demo (no key, no network after the build)

With Docker:

```bash
git clone https://github.com/ACUTE02/grounded-climate-analyst
cd <folder where you cloned>
docker compose up --build
```

Then open http://localhost:8080. Only the website's port is published. A cold build downloads base images and packages and can take several minutes: on the author's machine the first build took more than 10 minutes and a second build that had to run pip again took 310 seconds; later builds reuse cached layers and take seconds.

Without Docker (the Docker image uses Python 3.11.9 and builds the website with Node 24.16.0; the non-Docker install also worked once on Python 3.14.5 with the lock file, in about 6 minutes; nothing else was tried and nothing is guaranteed):

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt -c requirements.lock.txt
uvicorn api.app:app --port 8000
```

and in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173. Demo mode answers only the six saved examples; any other question gets a plain refusal (HTTP 409).

## 🔑 Run Live (your own key)

Live mode runs the real pipeline: tool routing and the written report use the Gemini API, and the retrieval index uses its embedding API. You need your own free key (https://aistudio.google.com/apikey) and a network connection. The app counts its own generation calls against a budget of 20 a day and 5 a minute (the free tier when this was built); one report costs 2 to 4 of them. Live mode inside Docker is not supported: the container has no models or index. Demo mode needs none of this.

Windows PowerShell (from the folder where you cloned):

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -c requirements.lock.txt
copy .env.example .env
```

macOS / Linux (bash):

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -c requirements.lock.txt
cp .env.example .env
```

Then, in both:

1. Open `backend/.env` in an editor (it is gitignored), put your key in `GEMINI_API_KEY` and set `CLIMATE_SUITE_MODE=live`.
2. `python -m scripts.setup` (still in `backend/`). It restores the input data from the checked-in snapshot (no network) and builds the forecast models; without a key it skips the retrieval index, says so, and ends with "Setup incomplete". With a key it also builds the index: that step downloads the corpus pages and calls the embedding API, so it needs the network and the key. Some source pages (the NDMA ones) have returned 404 or 503, so the index you get can be smaller than the author's. The time and the number of embedding calls were not measured. `--refetch` fetches fresh weather data instead of the snapshot; the numbers can then differ slightly from the published evidence.
3. `uvicorn api.app:app --port 8000`
4. In a second terminal: `cd frontend`, `npm install`, `npm run dev`, then open http://localhost:5173 and type a question on the Ask page.

If the key is missing the backend says so in its start-up log, and a question returns HTTP 503 with "Live mode needs a Gemini key in backend/.env (GEMINI_API_KEY). See the README." A daily-quota error comes back as HTTP 429 with an explanation. The saved examples are snapshots: asking the same question live can give different text. One live report was run on the author's machine (its calls and grounding result are in the author's run report, not in this repository).

## 🤖 The Small Language Model (optional)

Its code, configs, tests and the one published adapter are part of this release under `slm/`. The adapter is public on Hugging Face ([GHOSTBHAIYA/climate-slm-1.5b-sft](https://huggingface.co/GHOSTBHAIYA/climate-slm-1.5b-sft), Apache-2.0, on the base model Qwen/Qwen2.5-1.5B-Instruct, LoRA rank 16, alpha 32) and is downloaded when the service loads it; no weights are stored here. Only this 1.5B adapter is published: the 0.5B and DPO adapters whose results appear in the SLM README are not part of this release. The service is optional (compose profile `slm`).

The real 1.5B SFT adapter was downloaded from Hugging Face and served on the author's Windows laptop with the hf backend. It answered one answerable question from its passage and refused one unanswerable question. That was a two-request smoke test, not a benchmark.

The analyst has an optional step that asks the small language model about the first retrieved passages (`CLIMATE_SUITE_SLM=on` and `SLM_URL`; **off by default**). The SLM service and the backend both default to port 8000, so start the SLM on another port (`PHASE5_PORT=8001`) and set `SLM_URL=http://127.0.0.1:8001` for the backend. Its answers are shown next to the report and are not used to write it or to check its numbers.

The website's SLM page is a chat. Each question is answered from a text the SLM service is given: either a passage you paste (`POST /slm/answer`) or the project's own material (`POST /slm/ask`): a saved report with its forecast labels plus the best-matching passages of the source corpus (keyword search over `retrieval/chunks.jsonl`, no key), assembled by the backend. The raw weather tables are not given to the model. At most 4000 characters of text and 500 of question go to the service, so long material is cut and the page says so; nothing is stored or logged. It works in demo mode too, because it uses no Gemini call, but only when `SLM_URL` is set and the service is running; otherwise the chat is off and says how to start it. In Docker the corpus is not built, so only the report part is available there; set `SLM_URL=http://host.docker.internal:8001` for an SLM running on the host (empty by default). The model was trained on passage questions: how well it answers from this mixed material has not been measured, and no claim is made about how fast or how accurate it is.

The training data is not included in this release (the tests that need it skip when it is missing). The Colab and benchmark scripts and the notebook are kept for reference only: the 0.5B and DPO adapters they refer to were not published.

## ⚠️ Known Limits

- The saved examples are snapshots: asking again in live mode can give different text.
- Three of the four live-written examples carry a number-check "warning" caused by numbers in the system's own notes (a count of characters, a date), which the checker flags on purpose.
- One example (out of scope) has no written report.
- Saved forecasts use data through December 2024, because the data snapshot is fixed.
- Not included yet: a measurement of the SLM step, and a 3D view.

## 🧪 Testing

```bash
cd backend && python -m pytest          # live tests are opt-in (RUN_LIVE_* variables)
cd frontend && npm test
cd slm && python -m pytest tests        # some tests skip when the training data is not present
```

## 🗂️ Data and Credit

The data files in `backend/data_snapshot/` keep the terms of their sources, listed here and in `NOTICE`; they are not under the Apache licence of the code. Each file's source, fetch date and SHA256 are in `backend/data_snapshot/manifest.json`.

- **Weather:** Weather data by [Open-Meteo.com](https://open-meteo.com/), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: the daily values were aggregated to monthly (and columns renamed). Open-Meteo describes its Historical Weather API as based on reanalysis datasets, with ERA5 as the main one; which dataset served each value was not recorded here. Contains modified Copernicus Climate Change Service information 2026. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information. Open-Meteo's terms say the free API may only be used for non-commercial purposes (https://open-meteo.com/en/terms); `--refetch` uses that free API.
- **ENSO (ONI):** obtained from NOAA PSL (NOAA's three-month running mean of ERSST.V5 anomalies in the Nino 3.4 region). Data provided by the NOAA/OAR/PSL, Boulder, Colorado, USA, from their Web site at https://psl.noaa.gov/. PSL says information on its servers is in the public domain unless annotated otherwise (https://psl.noaa.gov/disclaimer/).
- **Indian Ocean Dipole (DMI):** created by NOAA PSL from HadISST1.1 (its file header says so). HadISST data were obtained from https://www.metoffice.gov.uk/hadobs/hadisst/ and are (c) British Crown Copyright, Met Office, provided under an Open Government Licence. Citation: Rayner et al. (2003), J. Geophys. Res. 108(D14), 4407, doi:10.1029/2002JD002670.
- **Small language model:** the adapter is on Hugging Face ([GHOSTBHAIYA/climate-slm-1.5b-sft](https://huggingface.co/GHOSTBHAIYA/climate-slm-1.5b-sft), Apache-2.0) on the base model Qwen/Qwen2.5-1.5B-Instruct (Apache-2.0).

If you reuse this work, please credit Ayushmaan Gupta and link back to the repository: https://github.com/ACUTE02/grounded-climate-analyst

## 📄 License

Apache License 2.0: see `LICENSE`, and `NOTICE` for the credit request and third-party material.

## ⭐ Support

If you found this project useful, star the repository, and share it with others.

---

<div align="center">

**🌾 Climate Suite**

*Climate-risk analysis that says how far to trust each number.*

</div>
