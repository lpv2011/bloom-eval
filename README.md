# bloom-eval: Brand CI for Bloom Brand Skills

Does a brand edit make generated assets **better or worse**? `bloom-eval` answers that with
numbers. It generates a fixed set of tasks against a Bloom brand, scores each image with
deterministic checks (palette, headline copy, logo integrity), and compares two runs with a
significance test so a CI job can fail on a real regression.

Built **only on Bloom's public API**. It assumes nothing about Bloom's internals.

> Independent project. Not affiliated with or endorsed by Bloom.

## Quick start (no Bloom account needed)

```bash
pip install -e ".[vision,dev]"        # also: apt install tesseract-ocr  (for the copy check)
pytest                                 # 13 tests, incl. an offline end-to-end run vs a mock Bloom API
bloom-eval validate                    # how good are the checkers themselves? (synthetic faults)

python scripts/make_demo.py            # good set vs "bad brand edit" set
bloom-eval score demo/suite.yaml demo/good --out demo/runs
bloom-eval score demo/suite.yaml demo/bad  --out demo/runs
bloom-eval compare demo/runs/<good> demo/runs/<bad>    # prints a table, exit code 1 = regression
```

## With a real Bloom brand

```bash
export BLOOM_API_KEY=...
bloom-eval doctor                      # compare endpoint paths/fields with Bloom's live OpenAPI spec
cp suite.example.yaml suite.yaml       # set brand.id, palette, logo, tasks
bloom-eval run suite.yaml              # asks before spending credits; writes runs/<id>/report.html
# ...make a Brand Edit / change the brand, then run again...
bloom-eval compare runs/<before> runs/<after>
```

**Do this first:** run `bloom-eval doctor`. The public docs do not confirm the exact generate
and Skill-read paths or all request field names, so those are isolated in `config.py` and
marked `VERIFY`. If `doctor` reports a mismatch, write an `endpoints.yaml` and pass
`--endpoints endpoints.yaml`; no code changes needed.

## What is checked

| Check | Method | Notes |
|---|---|---|
| `color` | CIEDE2000 distance of pixels to the palette: share of off-palette pixels, plus how many palette colours actually appear. Near-neutral pixels ignored. | Palette from `suite.brand.palette`, else hex colours found in the Skill snapshot |
| `text` | Tesseract OCR, fuzzy window match against `expected_text` (default >= 0.95) | Needs the `tesseract` binary |
| `logo` | ORB feature matching + affine fit: found, and not stretched (singular-value ratio) | Heuristic; flat wordmarks may return `uncertain` |
| `review:*` | Bloom Review verdicts (`--with-review`) | Private beta; result parsing is tolerant, raw JSON is kept |
| judge (optional) | Vision-LLM with a pinned rubric (`checks/judge.py`) | Off by default; validate against human labels before trusting |

`uncertain` is a first-class outcome: it is excluded from pass rates and reported separately.

## How trustworthy are the checks?

`bloom-eval validate` injects known faults into synthetic brand banners and reports false
positives and recall. Current result (n=12 per condition):

```
clean images wrongly failed:   color 0/12   logo 0/12   text 0/12
hue_shift       -> color caught 12/12
logo_stretched  -> logo  caught 12/12
logo_missing    -> logo  caught 12/12
headline_typo   -> text  caught 12/12
```

**Caveat:** synthetic banners are far easier than real generations (photos, textured
backgrounds, stylised type). Treat these as a sanity floor, not accuracy claims. The real
test is labelling ~50 real Bloom outputs by hand and measuring agreement; that is the first
thing to do with account access.

## Statistics

Per check: pass rate = pass / (pass + fail), with a 95% Wilson interval. `compare` uses a
two-proportion z-test and flags `REGRESSION` only if the drop is >= 5 points **and** p < 0.05
(both configurable). Generation is stochastic, so use `samples >= 3` per task and enough
tasks (aim for 30+ decided images per check) or small suites will under-detect.

## Cost and limits (from Bloom's public docs)

- 2K image = 1 credit, 4K = 2 credits. A 20-task x 3-sample suite is about 60 credits.
- 120 requests / 60 s per API key. The client uses a 100/60 s sliding window, honours
  `Retry-After` on 429, and never retries 402 (credits).
- `run` prints the estimate and asks before generating (`--yes` for CI).

## Layout

```
src/bloom_eval/
  config.py     all Bloom paths/fields (VERIFY-marked), YAML-overridable
  client.py     REST client: auth, rate limit, retries, tolerant response parsing
  runner.py     snapshot Skill -> generate N samples/task -> score -> results.json
  checks/       color.py  text.py  logo.py  judge.py
  report.py     aggregate, Wilson CI, z-test compare, HTML report
  synth.py      synthetic banners + fault injection
  validate.py   measures the checkers (offline)
  doctor.py     endpoints vs live OpenAPI spec
tests/          colour math vs published CIEDE2000 reference values, stats, mocked end-to-end
```

## Known limitations

- Generation always uses the brand's **active** Skill. To test a Brand Edit *candidate*
  before Apply you need a sandbox brand (duplicate sources) or Bloom-side support.
  That gap is the point of the feature proposal below.
- No determinism/seed control is documented, so variance is measured, not removed.
- Typography and "voice" need the optional judge or Bloom Review; code can't decide them.
- Skill JSON layout is unknown, so palette/skillId discovery is key-name based (`find_palette`, `find_key`).

## The product idea this prototypes: "Test before Apply"

When a Brand Edit candidate is prepared, Bloom already shows a diff of the Skill files. This
harness shows what a second panel could add: generate a small fixed suite against the
candidate and show deltas ("color 91% -> 94%, logo 88% -> 79%") before the user applies it,
with Review verdicts as one signal and user Apply/Discard decisions as labels for improving
the checks over time. Only Bloom can run generations against an unapplied candidate; this
repo proves the metrics, statistics and CI workflow around it.

## Roadmap

1. Hand-label 50 real outputs; report agreement per check.
2. Typography via Review/judge with human-agreement numbers.
3. Webhook/SSE instead of polling if Bloom adds it.
4. `suite init --brand <id>`: auto-draft tasks from the Brand Skill's guidance files.
