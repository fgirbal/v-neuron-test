# Pre-registration 2: is InceptionV1 mixed4d:493 a "pointed snout" detector?

Written 2026-10-17, before any stimulus was shown to this unit as part of this study. A SHA-256 of this file
goes to `results_4d493/prereg.sha256` before the test stage runs. Later changes are logged under
**Deviations**. This is the follow-up to the mixed4e:55 study ([REPORT.md](REPORT.md)). It reuses that
study's code and conventions except where stated.

## Why this unit

The essay calls 4d:493 "fox snout" / "pointed snouts", 32% of 4e:55's drive. In study 1, 4d:493 was the one
mixed4d input shared by every route into 4e:55 (cats, dogs, wild canids, vehicles, synthetic wedges). If any
unit in this circuit is the "pointed" one, it should be this one.

## What I already know about this unit (disclosed)

1. It provides 34% of 4e:55's positive drive over 4e:55's top 200 ImageNetV2 images.
2. Among its own top 30 ImageNetV2 images (by spatial max), the commonest classes are lynx, red fox and
   lesser panda.
3. On synthetic V silhouettes, it was the largest mixed4d contributor to 4e:55's (weak) response. So it
   responds to them somewhat, at an unknown fraction of its own maximum.

Nothing else about its responses has been looked at.

## Changes from study 1, and why

- **Apertures sized to the face (P3).** In study 1, apertures covered ~18% of the face, so P3 mostly measured
  the aperture outline. Here the disk aperture *is* the face circle (diameter = head size), and the other
  apertures have equal area.
- **The ellipse control is pre-registered (P1, P2, P3).** Study 1's exploratory checks found that an
  equal-area ellipse drove 4e:55 as well as the V, so V vs disk confounds pointedness with elongation. The
  pointedness contrasts are now V vs ellipse (same area, same width, no corners) and V vs U (rounded apex).
- **The P4 route test asks the shape question directly:** do category-free V silhouettes reach 493 through
  the same inputs as natural cat and fox faces?
- **Silhouette vs interior (F3) is promoted to primary (P6).**
- **Stage-2 synthetics use the gray background only.** Study 1's scene filter left only 2 backgrounds.

## Unit and readouts

- Unit: mixed4d channel 493 = channel 29 of the pool-projection branch (`mixed4d_pool_reduce_pre_relu_conv`).
  Its input is a 3×3 / stride-1 max-pool of mixed4c (512 channels). Pre-ReLU at location p is exactly
  b + Σ_c W[29,c]·pool(mixed4c)_c(p).
- **Drive** = post-ReLU spatial max of mixed4d channel 493 (14×14 map).
- **rel** = drive / A_ref, where A_ref = mean drive of the unit's top 0.2% of ImageNetV2 images (the
  `aref4d[493]` already computed in study 1).
- Everything else as in study 1: thresholds ("strong" ≥ 0.30, "≈" = 90% CI inside ±0.10), 95% bootstrap CIs,
  class groups, preprocessing, datasets.

## Validity checks

- **V0:** ≥ 50% of the top 50 ImageNetV2 images come from mammal classes (ImageNet indices 151–299 and
  330–388). If it fails: stop and report.
- **V1:** the stage-1 battery drives ≥ 10% of mixed4d channels to ≥ 0.30 of their own A_ref. If it fails:
  P1/P2 are uninterpretable.

## Stimuli

**Stage 1 (tuning only).** The same grid as study 1, same seed: V/U with α ∈ {30, 60, 90, 120}°, O, bar;
24 orientations; sizes {64, 112, 160}; outlines and textured fills. θ\*, α\*, s\* = argmax of 493's mean rel
over filled V silhouettes. Frozen before stage 2.

**Stage 2 (P1, P2, P5).** At θ\* (±15°), α\*, s\* (and 0.8·s\*), position jitter ±8 px, gray background:
- shape: V; U; E (ellipse with the V's area and width); O (disk, equal area);
- fill: neutral DTD texture (study 1's held-out categories); pet fur; phase-scrambled fur;
- eyes: absent / present.

60 stimuli per cell (4 × 3 × 2 × 60 = 1,440).

**P3, face-sized apertures.**
- Content:
  - Oxford Pets cat faces (300) and dog faces (300), head box ≥ 40 px;
  - the unit's top 100 ImageNetV2 images in mammal classes, translated so its argmax is at the center;
  - 200 neutral textures.
- Pets faces are scaled so the head box's longer side = h, with h centered.
- **h** = the largest value in {160, 144, 128, 112, 96, 80, 64} for which all four apertures (V at θ\*/α\*,
  U, E, and a disk of diameter h), each of area π(h/2)², fit inside a 216 px bounding box. If none fits at
  α\*, use α = 90° for the V/U/E apertures (logged).
- Top images get the same apertures. Conditions: full (no aperture), V, U, E, O. Gray outside the aperture.

**P4, routes.** Contribution vector at the argmax location over the 512 pooled mixed4c inputs. Groups as in
study 1 (cats = ImageNetV2 cats + Pets cats; dogs; wild canids; vehicles), plus stage-2 V silhouettes
(neutral, no eyes). Threshold rel ≥ 0.2 (falling back to 0.1 if n < 30; flagged underpowered if still < 30).
Same A/B split, route definition (50% of positive drive), Jaccard and cross-carriage X.
Secondary: the same with baseline subtraction.

**P6, silhouette vs interior.** As study 1's F3: the unit's top 100 ImageNetV2 images (Mask R-CNN mask
covering the argmax) plus top 100 Pets images (trimaps). Edits: interior flat / interior scrambled (outline
kept); background flat / background scrambled (object kept).

**Secondary.** F6 within-category pointed vs round breeds and F7 pointed vs round objects, with the same
class and breed lists as study 1.

## Hypotheses

- **H_P (the essay):** detects a pointed apex / converging contour, independent of category.
- **H_E (elongation):** prefers elongated outlines, regardless of apex sharpness.
- **H_C (content):** driven mainly by interior content (fur, eyes, snout and face parts); outline matters little.

## Primary predictions

| Test | H_P | H_E | H_C |
|---|---|---|---|
| **P1** stage 2, neutral, no eyes | V ≥ 0.30; V−E ≥ 0.10 and V−U ≥ 0.10 (CI lower > 0) | V≈U≈E (pairwise ≈); E−O ≥ 0.10 | every shape < 0.30 |
| **P2** rank of 493 among 528 mixed4d channels by V−E | top 5% (≤ 27) | not top 5% | not top 5% |
| **P3** face-sized apertures, cat and dog faces | V−O ≥ 0.10 **and** V−E ≥ 0.10 on both | V≈E, and E−O ≥ 0.10 | σ ≤ 0.33 on both, and full face ≥ 2× the best aperture's drive on neutral texture |
| **P4** V silhouettes vs cats | X_{V←cat} ≥ 0.5 and X_{cat←V} ≥ 0.5 | (no prediction) | X_{V←cat} ≤ 0.25 |
| **P5** eye effect, disk, neutral | (none) | (none) | (none); H_face if ≥ 0.15 |
| **P6** outline kept, interior removed | R ≥ 0.5 for both edits | (none) | R ≤ 0.25 for both edits |

σ uses the corrected formula from study 1 (D3): σ = S⁺/(S⁺ + C⁺), with neutral texture as the comparison
content. P4's V half is N/A if the V-silhouette group is underpowered.

## Decision rules

- **H_P supported:** P1, P2 and P3 pass under H_P, and P6 R ≥ 0.5 (and P4 shared, if powered).
- **H_P refuted:** V1 passes; P1 fails under H_P; on both cat and dog faces V−O < 0.10 or V−E < 0.10; and
  P6 R ≤ 0.25.
- **H_C supported:** P1 shapes all < 0.30, P3 σ ≤ 0.33 on both, P6 R ≤ 0.25 (and P4 X_{V←cat} ≤ 0.25, if powered).
- **H_E supported:** the P1 H_E pattern and the P3 H_E pattern on both face sets.
- **Otherwise:** mixed, reported as graded quantities.

## Procedure

1. Rescan ImageNetV2 and Pets with 493's readouts, then V0.
2. Stage 1 → V1 and θ\*/α\*/s\*.
3. Hash this file and the test code.
4. Debug the test pipeline on another mixed4d pool-branch unit (4d:492). Then run on 493 once. That run is
   the one reported; any rerun after a bug fix is logged.

## Deviations

**D1 (after stage 1, before any stage-2/test stimulus). Ellipse definition.** Stage 1 froze α\* = 30°
(a narrow V), θ\* = 90°, s\* = 112. The text above says E has "the V's area and width". For study 1's wide
120° V, width (x-extent) was its long axis, so E was equally elongated. For a narrow 30° V, x-extent is the
*short* side, which would make E nearly a circle and defeat its purpose as an elongation control.
E is therefore defined as: same area as the V, **long axis equal to the V's longest extent and aligned with
it**. This was decided from the stage-1 tuning parameters only.

**D2 (found on the debug unit 4d:492, before the 4d:493 test run). Malformed U at narrow angles.**
Study 1's U construction (arc radius 0.35 × arm length) puts the tangent points beyond the arm ends when the
apex angle is narrow. At α\* = 30° it renders as a crescent, not a rounded V. For study 2, U is redefined:
the arc is tangent to the arms at 50% of their length (radius = 0.5·L·tan(α/2)), then scaled to equal area.
Study 1's tests are unaffected, since they ran at α\* = 120°, where the old construction is valid. The only
study-1 output that touches the bad shape is the exploratory stage-1 "by shape" U summary (α = 30° cells).
Stage-1 θ\*/α\*/s\* selection uses V only, so it's unaffected in both studies.

**Post-run note.** P4's V-silhouette half was underpowered: only 26 stage-2 V silhouettes reached rel ≥ 0.1.
As pre-registered, it is N/A, and the decision rules ran without it. There were no reruns of the 4d:493
test stage.
