# Models and datasets

Every model and dataset used by the pipeline is listed here with its source and licence.
Open-source only; if a licence is unclear, it is not used until clarified.

| model or dataset | source URL | licence | used in |
|------------------|------------|---------|---------|
| football-player-detection.pt (YOLOv8, classes ball / goalkeeper / player / referee) | https://github.com/roboflow/sports (`examples/soccer/setup.sh`, Google Drive id `17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q`) | **Non-commercial use only, licence review pending.** See caveats below. | detection |
| football-players-detection (Roboflow Universe dataset the detector was trained on) | https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc | CC BY 4.0 according to search results; project page not verified (HTTP 403). Footage derives from the DFL Bundesliga Data Shootout. | detection (indirectly, via the weights) |
| DFL - Bundesliga Data Shootout (source footage of the dataset above) | https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout | Unclear: competition rules could not be read. | detection (indirectly) |
| DFL sample clips (5 x 30 s, 1080p, 25 fps), stored in `data/raw/` | https://github.com/roboflow/sports (`examples/soccer/setup.sh`) | Unclear, same as the DFL competition data. Local development and evaluation input only. **Nothing derived from them is published**: no frames, minimaps, clips, positions, tagged events (the golden set lives in `evals/private/`), per-clip results or clip ids. | detection, pitch (test input); golden set `dfl` (private) |
| google/siglip-base-patch16-224 | https://huggingface.co/google/siglip-base-patch16-224 | Apache-2.0 | tracking (team assignment) |
| football-pitch-detection.pt (YOLOv8-pose, 32 pitch keypoints) | https://github.com/roboflow/sports (`examples/soccer/setup.sh`, Google Drive id `1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf`) | **Non-commercial use only, licence review pending.** Same caveats as the detector weights, see below. | pitch |
| football-field-detection (Roboflow Universe dataset the keypoint model was trained on) | https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi | CC BY 4.0 according to search results; project page not opened. Image source not verified. | pitch (indirectly, via the weights) |
| Qwen/Qwen3-8B-GGUF, quantisation Q4_K_M (5.03 GB), run by Ollama as `hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M` | https://huggingface.co/Qwen/Qwen3-8B-GGUF | Apache-2.0 (model card, checked 2026-10-28) | report (default LLM backend, local) |
| StatsBomb Open Data, FIFA World Cup 2018 final (competition 43, season 3, match 8658), stored in `data/raw/statsbomb/` | https://github.com/statsbomb/open-data | **Not an open licence:** StatsBomb Public Data User Agreement (LICENSE.pdf, last updated 8 September 2023). See caveats below. | metrics (reference data), report (demo on a full match) |
| Claude (Anthropic API), model `claude-haiku-5-5` (low-cost; changed from `claude-opus-5-5` on 2026-10-28) | https://platform.claude.com | Proprietary API, paid per token | report (optional backend, `TAKTIKAI_LLM_BACKEND=anthropic`; not run so far) |

## Caveats for StatsBomb Open Data

Checked 2026-10-28 against the User Agreement in the repository (LICENSE.pdf):

- Allowed: use "for analysis, research and to facilitate the shared ideas & understanding of the
  data"; analysis "may be shared publicly".
- Not allowed: to "distribute, reproduce, sell or in any way provide the data to any external or
  third party" or to "commercially exploit the data or any analysis derived from" it.
- Required: "accredit any publication of analysis formed from StatsBomb Data with the StatsBomb
  brand logo"; StatsBomb also asks users to register at statsbomb.com/resource-centre.

Decision (2026-10-28): used only as reference data for checking the metrics and demonstrating the
report agent on a full match. The raw JSON and everything derived from it stay under `data/`
(never committed); generated reports carry a text credit, and the logo must be added before any
StatsBomb-based analysis is published (docs/TODO.md). The test fixtures are hand-written, not
StatsBomb data. `statsbombpy` is not used: its repository and PyPI page state no software
licence, so the JSON files are downloaded with the standard library instead.

Publication (2026-10-29): StatsBomb-derived analysis is published only in the static Hugging
Face Space (`space/`): aggregates of one match (team counts, ball losses per third, the pressing
proxy, shot and xG totals, the strongest pass-network links) and one accepted report, with the
logo in the page header and footer and stamped into every chart. No raw data, `events.json` or
per-event lists are published. Player names in the pass network are StatsBomb nicknames from
`lineups/<match>.json` where present.

| asset | source | terms | used in |
|---|---|---|---|
| `space/assets/hudl-statsbomb-logo-black.png`, `hudl-statsbomb-logo-white.png` | StatsBomb media pack (statsbomb.com/media-pack/, redirects to the Hudl brand portal), downloaded by the project owner on 2026-10-29 | Required by clause 1.4 of the User Agreement for any published analysis; the portal shows no separate brand guidelines without an interactive session. Used unmodified, scaled only. | Space header, footer and charts |

## Caveats for the detector weights

Found while planning the detection stage (2026-10-27):

- The weights are a YOLOv8 model trained with ultralytics. The roboflow/sports README states that
  YOLOv8 "is distributed under the AGPL-3.0 license"; Ultralytics treats trained weights as covered
  by AGPL-3.0 too. The `ultralytics` package used to run them is AGPL-3.0. If TaktikAI is ever
  offered as a network service, its source must then be released under AGPL-3.0.
- The `.pt` file is hosted on Google Drive with no licence statement of its own. The roboflow/sports
  repository code is MIT, which covers the code, not necessarily the weights.
- The training dataset is listed as CC BY 4.0 on Roboflow Universe, but its images come from the
  DFL Bundesliga Data Shootout Kaggle competition, whose data terms may limit use to the
  competition or to non-commercial purposes. Not verified.

Decision: accepted for development and portfolio use only. Before any commercial use, review the
licence chain or replace the detector (for example, an Apache-2.0 detector such as RF-DETR trained
on data with a clear licence). The detector sits behind `backend` in `configs/detection.yaml`, so
it can be swapped without touching the rest of the stage.

The pitch keypoint model (`football-pitch-detection.pt`, added 2026-10-27) has the same chain:
YOLOv8-pose trained with ultralytics (AGPL-3.0), weights on Google Drive with no licence of their
own, training dataset listed as CC BY 4.0 on Roboflow Universe. Same decision: non-commercial use
only, licence review pending; swappable via `backend` in `configs/pitch.yaml`.

## Libraries with notable licences

| library | licence | note |
|---------|---------|------|
| ultralytics | AGPL-3.0 | runs the detector and the pitch keypoint model; copyleft, see above |
| mplsoccer | MIT | minimap pitch drawing |
| streamlit | Apache-2.0 | tagger and review apps (`apps` extra) |
| supervision | MIT | ByteTrack, annotators; pinned `<0.31` because `sv.ByteTrack` was removed in 0.31.0 |
| transformers, torch, umap-learn, scikit-learn, opencv-python, gdown | Apache-2.0 / BSD / MIT | team assignment, video I/O, weight download |
| anthropic (Python SDK, 2.x) | Apache-2.0 | optional report backend (`anthropic` extra); imported only when that backend is selected |

## External tools (called as binaries, not bundled)

| tool | source | licence | used in |
|------|--------|---------|---------|
| FFmpeg / ffprobe 9.0.2 ("full" build by gyan.dev, installed with `winget install Gyan.FFmpeg`; on Kaggle/Colab from the system packages) | https://ffmpeg.org, build: https://www.gyan.dev/ffmpeg/builds/ | GPL-3.0 for this build (`--enable-gpl --enable-version3`, includes libx264); FFmpeg itself is LGPL-2.1+ without those options | clips stage: cutting (stream copy) and re-encoding event clips; reading keyframe times |
| `hf` CLI (huggingface_hub, installed with `pip install hf` by whoever publishes) | https://github.com/huggingface/huggingface_hub | Apache-2.0 | `scripts/publish_space.sh --push`: uploads `space/` to the Hugging Face Space |
| Ollama 0.40.12 (installed with `winget install Ollama.Ollama`) | https://github.com/ollama/ollama | MIT | report stage: serves the local LLM on 127.0.0.1:11434 (HTTP, standard library client) |

TaktikAI only runs the `ffmpeg` and `ffprobe` executables that the user installed as separate
programs (`subprocess`) and talks to a locally running Ollama server over HTTP; it does not ship,
link or redistribute them.
