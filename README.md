# Is InceptionV1's "V neuron" a V detector? Two pre-registered tests

The essay [*Neural Networks Are Already Clean*](https://already-clean.goodfire.com/) reinterprets InceptionV1
unit **mixed4e:55**, long described as a polysemantic "cat faces and car fronts" neuron, as a single detector
of V shapes: pointed things drive it, round things push it negative. The essay's evidence is co-occurrence:
which images activate the unit most, and which units feed it and read from it. Its largest input,
**mixed4d:493**, is described as a "fox snout" channel.

In natural photos, shape and category go together, so top images alone cannot tell a V detector from a unit
that responds to cat faces and car fronts. This repository contains two small pre-registered studies that
vary shape and content independently, on 4e:55 (the essay's "V neuron") and on 4d:493 (its "fox snout"
input), together with all code, results and the full write-up.

Blog post: (link to be added)

- Full report: [REPORT.md](REPORT.md)
- Results page with charts, methods and decision tables: [report/index.html](report/index.html). This is a
  static page; open it in a browser from a local copy of the repository.

## Results

Scores are relative to the unit's natural maximum: 1.0 is the mean response of its top 0.2% of ImageNetV2
images. The pre-registered threshold for "strong" is 0.30.

| Study | Pre-registration | Pre-registered verdict |
|---|---|---|
| 1. 4e:55: shapes, silhouettes, input routes | [PREREGISTRATION.md](PREREGISTRATION.md) | **Mixed.** None of the "supported" or "refuted" rules fired. |
| 2. 4d:493, the "fox snout" input | [PREREGISTRATION_4d493.md](PREREGISTRATION_4d493.md) | **Pointed-snout account refuted; content account supported.** The elongation account was not supported. |

**Study 1 (4e:55).** Verdicts are in `results/stage2_tests.json` under `decisions`.
- Synthetic shapes with neutral texture and no eyes: V 0.12, rounded V (U) 0.11, disk 0.02. The essay's
  account needed V ≥ 0.30.
- Ranked by V minus disk, 4e:55 is #65 of 832 mixed4e units. The essay's account needed the top 5%.
- Small apertures: shape share 0.70 for cat faces (passes the essay's ≥ 0.67), 0.42 for vehicles (fails).
- Input routes: the cat and vehicle routes share a single mixed4d input, 4d:493. The rest of each route is
  different.
- A pair of eyes on a disk adds +0.001, so the face-layout account gets no support.
- Secondary: keeping an object's outline while flattening or scrambling its interior leaves 0.15 and 0.11 of
  the drive (the essay's account predicted ≥ 0.5). Flattening or scrambling the background leaves 0.99 and
  0.97.

**Study 2 (4d:493).** Verdicts are in `results_4d493/stage2_tests.json` under `decisions`.
- Shapes alone peak at 0.10 (V). The V does beat the rounded V by +0.05 [0.03, 0.07]: a real but small
  apex preference.
- With the disk window sized to the face circle, cat faces score disk 0.52 vs V 0.42 (shape share 0.00).
  Dog faces have a shape share of 0.18.
- Outline-only silhouettes keep 0.10 (flat interior) and 0.08 (scrambled interior).
- The V-silhouette route test was underpowered (26 V stimuli reached 0.1), so it is N/A, as pre-registered.

**Exploratory and post hoc analyses (not part of any verdict).**
- `results/exploratory_E1_E2_E3.json` was designed and run after study 1's pre-registered results
  had been seen. It contains E1 (an equal-area ellipse control), E2 (apertures at 4× area) and E3 (the V
  with its apex flipped).
- The report also marks a few post hoc observations, such as breed-level comparisons and eyes inside wedge
  shapes.
- The example images on the results page are picked by a fixed rule and are illustrations, not evidence.

## Repository layout

```
README.md
REPORT.md                           full written report
PREREGISTRATION.md                  study 1 (4e:55); deviations D1–D3; post-run notes
PREREGISTRATION_4d493.md            study 2 (4d:493); deviations D1–D2; post-run note
requirements.txt
src/                                experiment code
  common.py                         model, hooks, data loading, stimuli, statistics, paths
  unitprobe.py                      readout for a mixed4d pool-branch unit (study 2)
  s0_scan.py … s4_figure.py         study 1
  s5_stage01_4d.py, s6_tests_4d.py  study 2
results/                            study 1: JSON results, figures, prereg.sha256, exploratory checks
results_4d493/                      study 2
report/
  index.html, img/                  results page (static; its build scripts are not included)
```

- Files ending in `_debug54` or `_debug492` come from pipeline-debug runs on stand-in units (4e:54 and
  4d:492), done before the single run on each real unit. They are kept for transparency but are not part of
  the reported results.
- `report/index.html` is a static page generated from the results. The scripts that build it are not part of
  this repository. It loads web fonts from Google Fonts, and everything else is local.
- The pre-registrations are kept exactly as hashed. Their relative links (for example `REPORT.md` or
  `results/prereg.sha256`) refer to this layout.

## Pre-registration and hashing

Each study followed the same protocol:

1. The pre-registration was written before the study's test stimuli were shown to the unit.
2. Its SHA-256 was appended, with a UTC timestamp, to that study's `prereg.sha256`.
3. Deviations (for example D1–D3 in study 1) were appended to the pre-registration, with reasons, before the
   test they affect. The file was re-hashed at each stage.
4. Immediately before the test run on the real unit, the pre-registration and the test code (`s2_tests.py`
   or `s6_tests_4d.py`, plus `common.py` and, for study 2, `unitprobe.py`) were hashed again.
5. The pipeline was debugged on a stand-in unit first. The reported run on the real unit is the first and
   only one.
6. After each run, a short "Post-run note" paragraph was appended to the pre-registration. It was not
   hashed, so the current pre-registration files do not match their last recorded hash.

### Verification (checked against the files in this repository)

| `prereg.sha256` entry | Recorded (UTC) | What was hashed | Check | Result |
|---|---|---|---|---|
| `411def45…` | 2026-10-16 23:12:47 | `PREREGISTRATION.md`, first version | Current file cut at "## Deviations", with that section holding its original placeholder "(Filled in after the run.)" | Match |
| `0a170d94…` | 23:20:12 | `PREREGISTRATION.md` after D1 | Current file cut just before D2 | Match |
| `4408bd29…` | 23:28:07 | `PREREGISTRATION.md` after D2–D3, before the unit-55 test run | Current file with the post-run notes removed | Match |
| `f13c78cc…` | 23:28:07 | `src/s2_tests.py` | Current file | Match |
| `e1275d58…` | 23:28:07 and 01:47:59 | `src/common.py` (both logs) | Current file | No match; see below |
| `008a7783…` | 2026-10-17 01:35:32 | `PREREGISTRATION_4d493.md`, first version | Current file with the post-run note removed and D1–D2 replaced by "(None yet.)" | Match |
| `47cecedd…` | 01:47:59 | `PREREGISTRATION_4d493.md` after D1–D2, before the 4d:493 test run | Current file with the post-run note removed | Match |
| `a329ebf7…` | 01:47:59 | `src/s6_tests_4d.py` | Current file | Match |
| `15e8995b…` | 01:47:59 | `src/unitprobe.py` | Current file | Match |

In summary:
- For both studies, the version hashed immediately before the test run is reproduced exactly by removing
  the appended post-run paragraph.
- Every recorded entry for the pre-registrations is reproduced, including each one's first version. For
  those, cut the file at its "Deviations" heading and restore the original placeholder: "(Filled in after
  the run.)" for study 1, "(None yet.)" for study 2.
- The full current pre-registration files (`86d64bfe…`, `f0f1c877…`) match no recorded hash. That is
  expected, because the post-run notes were added after the runs.
- `s2_tests.py`, `s6_tests_4d.py` and `unitprobe.py` match their recorded hashes.
- `common.py` does not match `e1275d58…`. For publication, the directory settings at the top of the file
  (`SCR`, `PROJ` and the comment on `WORK`) were changed to the repository-relative defaults described under
  Reproducing. They had pointed to directories on the machine that ran the studies. No other line changed.

Other scripts (stage 0–1, exploratory and figures) were never hashed. The pre-registrations only commit to
hashing the test-stage code.

Two caveats apply. The hash logs were written on the same machine by the same analyst who ran the tests.
They show that the files are unchanged since the logged points (apart from the noted appends), but they
cannot prove when those points were. On the original machine, file modification times were consistent with
the protocol: each hashed code file was last modified before its hash entry, and each reported result file
was written after it. Git does not preserve those times.

To check the hashes yourself, run this from the repository root (`sha256sum` on Linux):

```bash
shasum -a 256 src/s2_tests.py src/s6_tests_4d.py src/unitprobe.py src/common.py
python3 - <<'EOF'
import hashlib
for name in ["PREREGISTRATION.md", "PREREGISTRATION_4d493.md"]:
    b = open(name, "rb").read()
    pre_run = b[:b.index(b"\n**Post-run note")].rstrip(b"\n") + b"\n"   # drop the appended post-run paragraph
    print(hashlib.sha256(pre_run).hexdigest(), name, "(post-run note removed)")
EOF
cat results/prereg.sha256 results_4d493/prereg.sha256
```

## Reproducing

**Environment.** The runs used Python 3.12.11 on an Apple M1 Max with PyTorch's MPS backend. `common.py`
falls back to CPU when MPS is unavailable; CUDA is not used. `requirements.txt` pins the versions used.
`torch-lucent` was installed with `--no-deps`, because its declared dependencies include test and notebook
tooling that this code does not need:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install $(grep -v -e '^#' -e '^torch-lucent' requirements.txt)
pip install --no-deps torch-lucent==0.1.8
```

**Datasets** (not included). Place them under `scratch/data/` (see Paths below):

| Dataset | Source | Expected location |
|---|---|---|
| ImageNetV2, matched-frequency (10,000 images, 10 per class) | Hugging Face dataset `vaishaal/ImageNetV2`, file `imagenetv2-matched-frequency.tar.gz` | `scratch/data/imagenetv2/imagenetv2-matched-frequency-format-val/<class 0–999>/` |
| Oxford-IIIT Pets (images and annotations) | https://www.robots.ox.ac.uk/~vgg/data/ (`pets/`) | `scratch/data/pets/images/`, `scratch/data/pets/annotations/` (with `list.txt`, `trimaps/`, `xmls/`) |
| DTD (Describable Textures Dataset) | https://www.robots.ox.ac.uk/~vgg/data/ (`dtd/`) | `scratch/data/dtd/dtd/images/<category>/` |

ImageNetV2 stands in for the ImageNet validation set, which requires an account.

**Model weights** download automatically on first use into `scratch/cache/`, which is set as `TORCH_HOME`
unless it is already set. InceptionV1 comes through Lucent with the original Lucid "inception5h" weights.
Mask R-CNN (used for object masks in the silhouette tests) comes through torchvision.

**Paths.** All paths are relative to the repository root and are set at the top of `src/common.py`:

- `scratch/` holds `data/`, `work/` (scan caches, about 0.6 GB) and `cache/` (model weights). Set
  `VNT_SCRATCH` to use another directory.
- Results are written to `results/` (study 1) and `results_4d493/` (study 2). Re-running a script overwrites
  the published result files there. To keep them for comparison, set `VNT_PROJECT` to another directory
  that contains an empty `results/` folder; the scripts then read and write `results/` and `results_4d493/`
  inside it.

`scratch/` is ignored by git. Expect small numerical differences on other hardware or library versions.

**Script order.** Run the scripts from `src/` with `python -I`. Each script adds its own directory to
`sys.path`. The scripts' docstrings describe their inputs, outputs and details.

```bash
# Study 1: 4e:55
python -I s0_scan.py                                  # scan ImageNetV2 + Pets, A_ref, validity check V0
python -I s1_stage1.py                                # stage-1 shape tuning, validity check V1, frozen parameters
python -I s2_tests.py --unit 54 --tag _debug54        # optional: pipeline-debug run on a stand-in unit
python -I s2_tests.py --unit 55                       # pre-registered tests (the reported run)
python -I s3_exploratory.py                           # exploratory E1–E3 (not pre-registered)
python -I s4_figure.py                                # summary figure

# Study 2: 4d:493
python -I s5_stage01_4d.py --unit 493                 # rescan, V0, stage-1 tuning, V1
python -I s6_tests_4d.py --unit 492 --tag _debug492   # optional: pipeline-debug run
python -I s6_tests_4d.py --unit 493                   # pre-registered tests (the reported run)
```

On the M1 Max, study 1 took about 35 minutes of wall-clock time (about 10 of them compute) and study 2
about 4 minutes of compute.

## Authorship

The study design and the experiments were orchestrated by a human (Francisco Girbal Eiras). The
implementation (stimuli, measurements and analysis) was written with Claude Opus 5.5 and checked by a human.

## Images

Images in the figures (`results/`, `results_4d493/`, `report/img/`) come from ImageNetV2, Oxford-IIIT Pets
and DTD, and remain under those datasets' original terms.
