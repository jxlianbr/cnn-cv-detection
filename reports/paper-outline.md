# Paper Outline

**Spatial Context versus Spectral Signatures: A Convolutional Neural Network Benchmark for Solar Photovoltaic Detection in Japan**
Julian Brödnow · Nordakademie · Applied Data Science

Target length: 30–40 pages. Chapter structure is fixed by the exposé (§7) — this outline fills it in, does not redesign it. Page budgets are guidance, not hard limits.

Status check against the exposé's own timeline (today: 2026-08-01): Week 1 (submit exposé, verify Zenodo dataset, set up GEE pipeline) should be done or closing; Week 2–3 (replicate spectral baseline, build CNN patch dataset) is starting now. **Results/Discussion sections below are placeholders until those experiments run** — per the paper-writing discipline, they get a stated evidence plan, not a claimed outcome.

---

## Front matter

- **Title** (fixed, from exposé)
- **Abstract** (~200–250 words) — write last. Must state: motivation (Shimada & Takeuchi's documented failure cases), the two RQs, method (3-stage replication + CNN benchmark), and the headline result once known.
- **Keywords**: solar photovoltaic detection, convolutional neural networks, spectral indices, remote sensing, Japan, Sentinel-2, benchmark
- Target: 0.5 page

---

## 1. Introduction (~3 pages)

- **1.1 Motivation** — Japan's PV growth since the 2012 FIT scheme, 3rd-largest installed capacity globally; land-use stakes (forest fragmentation from megawatt-scale sites, farmland abandonment for smaller sites) per Shimada & Takeuchi 2025 (#5) and Su et al. 2018 (#6, background only per exposé §3 — do not let this become a land-use-drivers paper).
- **1.2 The methodological tension** — spectral/pixel-wise classifiers (cheap to label, ignore spatial context) vs. CNNs (exploit spatial/textural patterns, expensive to label/train). Frame using the same research group's own trajectory: CNN for Vietnam (#3, Shimada & Takeuchi 2023) → non-CNN spectral-index+XGBoost for Japan (#5, 2025), citing labeling/compute burden — not an accuracy verdict. This is the paper's hook; lit review §3 and §4.3 already argue this in detail.
- **1.3 Research questions** (verbatim from exposé §2):
  - **Primary**: does a CNN incorporating spatial/textural context improve detection accuracy over the Shimada & Takeuchi (2025) spectral-index+XGBoost baseline, specifically on the three failure modes that paper documents — sub-50 kW installations, non-crystalline-silicon panel materials, spectrally ambiguous arrangements?
  - **Secondary**: how does CNN label efficiency compare to the spectral baseline?
- **1.4 Contribution statement** — not a new architecture; a controlled, same-country/same-data comparison targeting named failure modes rather than a cross-study accuracy comparison (exposé §6).
- **1.5 Chapter roadmap** — one paragraph mapping to §2–8 below.

---

## 2. Related Work (~6–8 pages)

Organize by the lit review's categories (`reports/literature-review.md`), each subsection ending with a 1–2 sentence bridge to why it matters for this paper specifically — don't let this become an annotated bibliography.

- **2.1 Spectral / pixel-wise PV mapping** — Zhang et al. 2022 China RF (#1), Feng et al. 2024 China RF+texture 10m (#4), Shimada & Takeuchi 2025 Japan XGBoost+SPPI-style index (#5) — this is the baseline this paper replicates, so give it the most detailed treatment of the three: exact feature set (reflectance bands + NDVI, MNDWI, PVSI, BSI), reported metrics, and the three named failure cases verbatim.
- **2.2 Spatial-context CNN PV mapping** — DeepSolar (#14), Kruitwagen et al. 2021 (#16, Nature, global CNN+RNN inventory, 68,661 facilities — the highest-profile venue in the review and the natural opening citation for this subsection), da Costa et al. 2021 Brazil segmentation benchmark (#2), Ishii et al. 2016 Japan CNN + spectral-band ablation (#13), Imamoglu et al. 2017 Japan weakly-supervised feedback-CNN (#7).
- **2.3 Japan-specific precedent** — #5 and #13/#7 together establish that Japan has both a CNN lineage and a current spectral-only SOTA; #3 (Vietnam, same lab) shows the lab's own CNN-to-spectral pivot in miniature.
- **2.4 Benchmark/methodology precedent** — Malof et al. 2016 RF-vs-CNN framing (#8), Li et al. 2025 hybrid CNN+PU-RF global dataset (#11), Clark & Pacifici 2023 VHR benchmark dataset (#10); Droguett & Sanchez (#9, non-peer-reviewed, cite only as illustrative of detector-benchmarking practice).
- **2.5 Research gap** — reuse lit review §4 directly: no controlled, same-country/same-data spatial-vs-spectral comparison exists; the Japan CNN lineage predates modern segmentation and was never benchmarked against a spectral-index baseline this strong; Ishii et al.'s band-ablation question is unresolved with modern architectures. Close with the RQs from §1.3 restated as the gap this paper fills.

---

## 3. Data (~3–4 pages)

- **3.1 Imagery** — Sentinel-2 Level-2A surface reflectance via Google Earth Engine, autumn 2022, matching Shimada & Takeuchi 2025's acquisition window exactly (necessary for a fair baseline replication).
- **3.2 Labels** — Shimada & Takeuchi's PV polygon dataset (Zenodo DOI 10.5281/zenodo.10674681). **State plainly whether the release includes point-level train/verification samples or only aggregated polygons** (exposé §4 flags this as unverified as of the exposé date — resolve before writing this section, since it determines whether §3.3 below is "use their sampling" or "re-derive a stratified sample from polygon boundaries checked against Google Earth imagery").
- **3.3 Patch extraction for CNN input** — patch size/resolution, positive (PV-centered) vs. negative sampling strategy, spatial train/test split (must avoid spatial leakage between adjacent patches — flag if not already handled).
- **3.4 Secondary validation** — METI FIT-registered capacity statistics at prefecture level, used as an independent cross-check the way the original study used it.
- **3.5 Failure-case stratification** — operational definitions for the three subsets (sub-50 kW; non-crystalline-silicon/thin-film; spectrally ambiguous/soil-blurred arrangements) and how each is identified in the label data — this stratification is what makes the Results chapter answer the *specific* RQ rather than a generic accuracy comparison.

---

## 4. Methodology (~7–9 pages)

- **4.1 Spectral/XGBoost baseline replication** — exact feature set (surface reflectance bands, NDVI, MNDWI, PVSI, BSI), hyperparameters, training protocol; state this is a *replication*, not a re-implementation from description alone, and report how closely it reproduces the published metrics as a validity check before any comparison is trusted.
- **4.2 CNN classifier** — architecture choice(s): compact from-scratch CNN vs. pretrained ResNet/EfficientNet fine-tuned on patches (mirrors the Brazil benchmark's #2 finding that label quality matters more than architecture — worth citing here to justify not over-investing in architecture search). Input channels: justify whether all Sentinel-2 bands are used or a subset, referencing Ishii et al.'s (#13) finding that band choice materially affects CNN accuracy.
- **4.3 Optional stretch extension** — frozen-embedding comparison against a self-supervised remote-sensing foundation model, explicitly framed as time-permitting, not core to the RQs.
- **4.4 Evaluation protocol** — overall accuracy, precision, recall, F1, Cohen's Kappa on the full test set (matching the original study's metrics exactly, for comparability) *and* on the three stratified failure-case subsets (the actual test of the primary RQ).
- **4.5 Label-efficiency comparison protocol** (secondary RQ) — learning curves: both models trained on shrinking fractions of the training set, accuracy/F1 vs. training-set size plotted for direct comparison.
- **4.6 Compute/reproducibility** — local RTX 2070 Super, Kaggle free-tier GPU fallback; state batch sizes, training time, and any resulting constraints on patch volume or model size, since this is itself evidence for the secondary RQ (compute cost is part of "label efficiency" broadly construed).

---

## 5. Results (~7–9 pages) — PLACEHOLDER pending Week 4–5 experiments

Structure now so results slot in without restructuring later:

- **5.1** Baseline replication vs. published Shimada & Takeuchi (2025) metrics — sanity-check table.
- **5.2** CNN vs. baseline, aggregate metrics, full test set.
- **5.3** CNN vs. baseline, stratified by the three failure-case subsets — **the figure/table that directly answers the primary RQ**. Plan a grouped bar chart or table: metric × subset × model.
- **5.4** Label-efficiency curves (secondary RQ).
- **5.5** Foundation-model comparison, if the stretch extension (§4.3) was run.

Per the paper-writing discipline: until real numbers exist, this chapter should contain the table/figure *specifications* and axis definitions, not invented numbers.

## 6. Discussion (~4–5 pages) — PLACEHOLDER, structure only

- **6.1** Interpret §5.3 against the primary RQ: does the CNN's gain (if any) concentrate in the named failure cases, or is it diffuse across the whole test set? (Exposé §5 states this exact distinction as the test.)
- **6.2** Interpret §5.4 against the secondary RQ and connect to lit review §3's "efficiency vs. accuracy" tension (SPPI's speed claim, #15, vs. segmentation accuracy gains, #2/#11) — this paper supplies the missing empirical data point in that debate, for Japan specifically.
- **6.3** Revisit Ishii et al.'s (#13) band-ablation finding in light of your own channel choice (§4.2) — did spectral richness matter for your CNN too, independent of the spatial-context question?
- **6.4** Design-choice framing per exposé §6: is the labeling/compute burden of CNN detection justified by measurable gains where it counts, or does the spectral approach's simplicity cost little accuracy in practice?

## 7. Limitations (~2–3 pages)

- Single country, single sensor (Sentinel-2), single time window (autumn 2022) — no claim to generalize beyond Japan or beyond this acquisition period.
- Label provenance: both models trained/evaluated against Shimada & Takeuchi's polygons — any label noise in that dataset propagates identically into both models, so this design can show *relative* not *absolute* accuracy, and cannot detect errors both methods share.
- Possible train/test leakage risk given baseline and CNN share a data source — state the spatial train/test split safeguard from §3.3 explicitly here as the mitigation, and name any residual risk.
- Small-N subsets: the three failure-case strata (esp. sub-50 kW, thin-film) are likely rare in the dataset — flag statistical power limits on §5.3's comparisons.
- Compute-constrained architecture search (§4.6) — results may not reflect CNN performance under larger-scale training.

## 8. Conclusion (~1–2 pages)

- Restate the two RQs and answer each in one paragraph, once §5–6 are final.
- Restate the design-choice recommendation for national-scale Japan PV monitoring (exposé §6).
- Future work: texture-only ablation (a gap Shimada & Takeuchi 2025 themselves name), foundation-model comparison if not completed as a stretch goal, extension beyond Japan.

---

## References

All 16 sources in `reports/literature-review.md`. Citation style: confirm required style (APA/IEEE/etc.) with the Nordakademie program guidelines before final formatting pass.

## Cross-check against exposé timeline

| Week | Dates | Milestone | Paper section(s) unblocked |
|---|---|---|---|
| 1 | Jul 24–30 | Submit exposé, verify Zenodo dataset, set up GEE pipeline | §3.2 |
| 2–3 | Jul 31–Aug 13 *(current)* | Replicate spectral/XGBoost baseline; build CNN patch dataset | §3.3, §4.1 |
| 4 | Aug 14–20 | Train/tune CNN; curate failure-case subset | §3.5, §4.2 |
| 5 | Aug 21–27 | Full evaluation, baseline vs. failure-case comparison | §5 |
| 6–7 | Aug 28–Sep 10 | Write Results, Discussion, Limitations | §5, §6, §7 |
| 8 | Sep 11–17 | Proofread, formatting, buffer | all |

Chapters 1–4 (Introduction through Methodology, ~19–24 pages) can be drafted essentially now, since they don't depend on experimental output. Chapters 5–8 need the Week 4–5 experiments first.
