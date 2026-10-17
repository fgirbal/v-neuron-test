# Pre-registration: is InceptionV1 mixed4e:55 a "V-shape" neuron?

Written 2026-10-17, before any test stimulus was shown to the unit. A SHA-256 of this file is saved to
`results/prereg.sha256` before the test stage runs. Anything changed after that is logged under
**Deviations** at the bottom, with the reason.

## Background

*Neural Networks Are Already Clean* (Cammarata, Goodfire, 2026) reinterprets the most famous polysemantic
neuron, InceptionV1 4e:55 (cat faces + car fronts), as a single "V-shape" / pointed-vs-round detector. The
evidence offered is top dataset images, the unit's strongest inputs and readers, and a handful of photos where
pointed objects drive it and round ones push it negative. In natural photos, shape is confounded with
category (the positives shown are dogs, foxes, cats and cars; the negatives golf balls, clocks and snails).
This study manipulates shape and content independently.

## Hypotheses

- **H_V (the essay):** the unit detects converging contours that meet at an apex (a V / pointed shape),
  largely independent of category. Round shapes push it negative.
- **H_face:** the unit detects a frontal face configuration (two eye-like blobs over a central feature),
  shared by animal faces and car fronts (headlights). This is the alternative Multimodal Neurons raised
  ("cat eyes and car lights").
- **H_OR (the original reading):** the unit is driven by separate, category-specific routes (cat/fox faces;
  car fronts). Shape is incidental.

The hypotheses are not exclusive. Where results are mixed, we report a graded **shape share** (below) rather
than forcing a winner.

## Model, unit, readouts

- InceptionV1 with the original Lucid weights ("inception5h"), loaded through Lucent. Input: 224×224 RGB in
  [0,1], mapped to `x*255 − 117`.
- Unit: channel 55 of `mixed4e` = channel 55 of the 1×1 branch (`mixed4e_1x1_pre_relu_conv` pre-ReLU;
  `mixed4e` concat output post-ReLU). Spatial map 14×14.
- **Drive** = post-ReLU spatial max. **Negative-pole measure** = pre-ReLU spatial mean.
- **A_ref** = mean drive of the top 0.2% of ImageNetV2 images (top 20 of 10,000). Every drive is reported as
  `rel = drive / A_ref`. Each mixed4e channel gets its own A_ref for the population analysis.
- Natural images: shorter side resized to 224, then center crop 224.
- Data: ImageNetV2 matched-frequency (10/class, all 1,000 ImageNet classes; stands in for ImageNet val, which
  needs an account), Oxford-IIIT Pets (images, trimaps, head boxes), DTD textures.
- Thresholds: "strong" = rel ≥ 0.30. "Approximately zero" = 90% CI inside ±0.10 rel. CIs are 95% bootstrap
  (10,000 resamples over stimuli or images).

## Validity checks (must pass before the tests are interpreted)

- **V0, unit identity.** Of the top 50 ImageNetV2 images, ≥ 60% come from cats ∪ wild canids ∪ dogs ∪ motor
  vehicles, with ≥ 5 cats/wild canids and ≥ 3 motor vehicles. If V0 fails, we stop and report. We do **not**
  search for another channel that fits.
- **V1, stimuli aren't uniformly out of distribution.** The stage-1 synthetic battery drives ≥ 10% of mixed4e
  channels to rel ≥ 0.30 of their own A_ref. Also, known curve detectors (mixed3b 379, 385, 343, 342, 388,
  324, 340, 330, 349, 406) respond to drawn arcs at median rel ≥ 0.5. If V1 fails, P1/P2 failures are
  **uninterpretable**, not evidence against H_V.

Class groups (torchvision ImageNet names; any name not found is dropped and logged):
cats = {tabby, tiger cat, Persian cat, Siamese cat, Egyptian cat};
wild canids = {red fox, kit fox, Arctic fox, grey fox, timber wolf, white wolf, red wolf, coyote, dingo, dhole,
African hunting dog}; dogs = ImageNet classes 151–268;
motor vehicles = {sports car, convertible, cab, racer, limousine, minivan, beach wagon, Model T, jeep,
police van, ambulance, fire engine, garbage truck, pickup, tow truck, trailer truck, moving van, minibus,
school bus, trolleybus, grille}.

## Stimulus families

**F1, synthetic shapes (no category content).**
- *Stage 1 (tuning only).* Shapes: V (two straight arms meeting at an apex, apex angle α ∈ {30, 60, 90,
  120}°), U (the same arms with a rounded apex, radius 0.35 × arm length), O (circle), straight bar.
  Orientation θ ∈ 0–345° in 15° steps (θ = 0: apex pointing down, like a chin). Object size s ∈ {64, 112, 160}
  px on the 224 canvas. Renderings: outlines (dark-on-light and light-on-dark) and filled silhouettes using
  DTD textures {marbled, stratified, grooved, wrinkled}, on uniform gray.
  **θ\*, α\*, s\*** = the argmax of mean rel over filled V silhouettes. Frozen before stage 2.
- *Stage 2 (test).* At θ\* (±15° jitter), α\*, s\* (and 0.8·s\*), position jitter ±8 px:
  shape {V, U, O; equal area, same centroid} × fill {neutral DTD texture from held-out categories {blotchy,
  cracked, crystalline, flecked, pleated, smeared, stained, veined}; pet fur (Oxford Pets body patches away
  from the head); phase-scrambled fur} × eyes {absent; present: two dark disks with highlights in the upper
  interior} × background {uniform gray; ImageNetV2 scene images (valley, lakeside, seashore, sandbar, cliff,
  promontory) whose own rel < 0.05}. 60 stimuli per cell.

**F2, aperture swap on natural content (outline varies, content held fixed).**
Content is seen only through an aperture: V (θ\*, α\*), U or O, all of equal area with the same centroid, on
gray, plus "full" (no aperture) as a reference. Content sources:
(a) Oxford Pets cat faces (head-box center, head scaled to s\*); (b) Oxford Pets dog faces (same);
(c) the top 100 ImageNetV2 images for the unit, translated so their argmax location is at the canvas center,
split into animal vs vehicle by class; (d) neutral DTD textures (stage-2 categories); (e) pet fur without a face.

*Shape share* for content k (2×2 against neutral texture n):
S = ½[(k_V − k_O) + (n_V − n_O)], C = ½[(k_V − n_V) + (k_O − n_O)], **σ_k = S / (S + C)**, clipped to [0,1].

**F3, silhouette vs interior (outline kept, interior removed).**
Top 100 ImageNetV2 images plus top 100 Oxford Pets images. Object masks: Pets trimaps (foreground ∪ boundary);
ImageNetV2 via torchvision Mask R-CNN (COCO, score ≥ 0.5, the instance covering the unit's argmax, otherwise
excluded). Edits: silhouette flat-filled with the object's mean color; silhouette with a phase-scrambled
interior; controls with the background flat-filled or scrambled (object intact).
Retained fraction R = rel(edit) / rel(original).

**F4, route decomposition (exact).** 4e:55 is a 1×1 conv over mixed4d, so its pre-ReLU value at location p
is exactly b + Σ_c W[55,c]·mixed4d_c(p). Per image, take the contribution vector at the argmax location.
Groups (natural images with rel ≥ 0.2; if a group has < 30 images, the threshold drops to 0.1; if still
< 30, the group is reported as underpowered): cats (ImageNetV2 cats + Pets cats), vehicles, dogs, wild
canids, plus stage-2 neutral V silhouettes. Random 50/50 split into A (define) and B (test).
Route_g = the smallest set of 4d channels whose mean positive contribution on A reaches 50% of the total.
Overlap J = Jaccard(Route_cat, Route_vehicle), relative to the within-group split-half ceiling: J_rel = J /
mean(J_cat,A↔B, J_veh,A↔B).
Cross-carriage X: the share of group h's positive drive (on B) carried by Route_g, divided by the share
carried by Route_h. Computed in both directions.
Secondary: the same analysis after subtracting the mean contribution vector of random ImageNetV2 images.

**F5, population specificity.** For all 832 mixed4e channels: Δ_ch = mean rel(V) − mean rel(O) on stage-2
neutral-texture, no-eye stimuli (both backgrounds), each channel normalized by its own A_ref. Rank channel 55.
(θ\* was tuned for unit 55, which biases this test **toward** H_V.)

**F6, within-category natural contrasts (secondary, observational).** Head types follow breed standards,
assigned before measurement.
Pets cats: pointed {Siamese, Abyssinian, Sphynx} vs round {Persian, British_Shorthair, Bombay}.
Pets dogs: pointed {shiba_inu, miniature_pinscher, keeshond, samoyed} vs round/flat {pug, japanese_chin,
boxer, american_bulldog}.
ImageNetV2: pointed {borzoi, collie, Shetland sheepdog, Afghan hound, Saluki, whippet, Ibizan hound, basenji,
Siamese cat, red fox, kit fox, Arctic fox, grey fox} vs round {pug, Pekinese, Shih-Tzu, boxer, French
bulldog, bull mastiff, Japanese spaniel, Persian cat}.
Pets images use head-box crops (square, 1.5× head size, resized to 224).

**F7, objects outside the hypothesized categories (secondary).** Classes not mentioned in the essay.
Pointed: {pick, missile, projectile, warplane, space shuttle, speedboat, canoe, bell cote, volcano, alp}.
Round: {golf ball, tennis ball, soccer ball, basketball, volleyball, ping-pong ball, orange, wall clock,
analog clock, plate, frying pan, manhole cover, airship, bubble, balloon}.
The essay's own examples (bullet train, pool table, fireboat) are reported descriptively only.

## Primary predictions

| Test | H_V predicts | H_face predicts | H_OR predicts |
|---|---|---|---|
| **P1** F1 stage 2, neutral fill, no eyes, gray | rel(V) ≥ 0.30, V − O ≥ 0.20, V − U ≥ 0.10 | V ≈ O, both < 0.30 | V ≈ O, both < 0.30 |
| **P2** F5 rank of ch. 55 by V − O | top 5% (rank ≤ 42 of 832) | not top 5% | not top 5% |
| **P3** F2 shape share σ (cat faces; top-vehicle content) | σ ≥ 0.67 for both | σ ≤ 0.33 | σ ≤ 0.33 |
| **P4** F4 cat vs vehicle routes | J_rel ≥ 0.5 and X ≥ 0.5 both ways; V silhouettes carried by the same route | J_rel ≥ 0.5 (shared face route); V silhouettes **not** carried by it | J_rel ≤ 0.2 and X ≤ 0.25 both ways |
| **P5** F1 eyes effect, rel(O+eyes) − rel(O), neutral fill | < 0.10 | ≥ 0.15 | < 0.15 |

Secondary predictions under H_V (others predict no difference):
- F3: R ≥ 0.5 for both silhouette edits (H_face/H_OR: ≤ 0.25).
- F6: pointed > round within cats and within dogs by ≥ 0.15 rel.
- F7: pointed objects above the all-class median and above round objects; round classes' pre-ReLU spatial
  mean below the all-image median.

## Decision rules

- **H_V supported:** P1, P2 and P3 pass, and P4 shows a shared route.
- **H_V refuted:** V1 passes, P1 fails **and** P3 gives σ ≤ 0.33 for cat faces and vehicles.
- **H_OR supported:** P1 fails, P3 σ ≤ 0.33, P4 disjoint (J_rel ≤ 0.2, X ≤ 0.25), and P5 < 0.15.
- **H_face supported:** P1 fails, P5 ≥ 0.15, and P4 shows a shared cat/vehicle route that does not carry
  V silhouettes.
- **Anything else:** "mixed". We report σ, J_rel, X and the eye effect as graded quantities.

## Procedure and forking-path controls

1. Scan ImageNetV2 (all mixed4e channels plus mixed3b), compute A_ref, run V0.
2. Stage 1 → V1 and θ\*/α\*/s\*.
3. Hash this file. Then run stage 2, F2–F7 once, on channel 55.
4. Code may be debugged on a different unit (mixed4e:54) or on stage-1 data. The first full run on unit 55
   is the one reported. Any rerun after a bug fix is logged below with the reason.
5. Exclusions only as stated: no mask, head box < 40 px, groups below the minimum n.
6. Every pre-registered outcome is reported, whichever direction it points.

## Known limitations, stated in advance

- Edits are automatic (masks, apertures, scrambles); nothing is hand-annotated.
- "V" is operationalized as converging straight contours. The stage-1 sweep over orientation, angle and size
  gives H_V its best shot, but the essay never defines the V precisely.
- ImageNetV2 is a stand-in for ImageNet val; head boxes exist only for the Pets trainval split.
- An aperture adds an artificial outline. That's intended: it is the manipulated variable.

## Deviations

**D1 (2026-10-17, after stage 0, before stage 1). V0 failed on one sub-criterion; continued anyway.**
Top 50 ImageNetV2 images: 60% in the listed groups (pass), 19 cats/wild canids (pass), **2 motor vehicles
(fail; ≥ 3 required)**. The remaining 20 are mostly small-mammal faces my class groups didn't list (lesser
panda ×4, hamster, weasel, lynx ×4, meerkat, polecat, ferret, mongoose, cheetah, hare). The vehicles are a
car grille (#7) and a jeep (#44).
V0 was meant to catch an indexing error (the wrong unit). Two independent checks against the essay's own
figure show the indexing is right. Over the unit's top 200 ImageNetV2 images, its largest mixed4d input is
4d:493 at 34% of positive drive (the essay reports "4d:493 fox snout, 32% of its drive"), and its strongest
5a 1×1 reader is 5a:94 (the essay's "flames" reader).
We therefore continue on channel 55, without searching other channels. Consequence: the unit's vehicle
response in ImageNetV2 is weaker than its cat/fox response, so the vehicle-based tests (σ_vehicle, the
vehicle route) may run into the pre-registered small-n fallbacks.

**D2 (before the unit-55 test run; decided on the debug unit 4e:54). Vehicle content for F2.** The top 100
ImageNetV2 images for unit 55 include only ~2–4 vehicles, too few to estimate σ_vehicle. Rule: if the top 100
contain < 10 vehicle-class images, the vehicle content set becomes the 30 highest-drive vehicle-class images
in ImageNetV2. The code logs it when this fires.

**D3 (before the unit-55 test run; found on the debug unit). σ formula.** As written, S/(S+C) clipped to
[0,1] goes wrong when the shape effect is negative: with S = −0.11 and C = +0.07, S/(S+C) = 3.2, which
clips to 1, i.e. "all shape", the opposite of the data. Corrected to σ = S⁺/(S⁺ + C⁺) with
x⁺ = max(x, 0), undefined if both are ≤ 0. This matches the intended meaning: the share of positive drive
attributable to shape.

**Post-run notes (2026-10-17, after the unit-55 run).** D2 did not trigger for unit 55: its top 100 contain
16 vehicle-class images, so σ_vehicle used the pre-registered top-100 split. Under the pre-registered scene
filter (scene images with rel < 0.05), only 2 scene backgrounds qualified. That's a consequence of the rule,
not a deviation. The exploratory checks E1–E3 (`src/s3_exploratory.py`) were designed and run after the
pre-registered results were seen, and are reported separately.
