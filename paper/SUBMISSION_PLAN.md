# JOSS Submission Plan — `linref`

Working notes for preparing the Journal of Open Source Software (JOSS) paper for
the linref 1.x release. This file is scratch/planning only and is **not** part of
the published paper (safe to delete before or after submission).

- Paper source: [paper.md](paper.md)
- Bibliography: [paper.bib](paper.bib)
- Branch: `paper/joss-submission` (created from `main` so it includes the source).

## How JOSS submission works (summary)

- The paper is a short (**750–1750 word**) Markdown file with YAML metadata.
- Required sections: **Summary, Statement of need, State of the field, Software
  design, Research impact statement, AI usage disclosure, Acknowledgements,
  References**.
- The paper must **not** report new research results; it describes the software.
- The paper may live in a branch that is never merged, as long as it was created
  from the default branch (done).
- Submission is via a short web form; review happens publicly on GitHub.

### Pre-review gates to be aware of (can cause desk rejection)

1. **Public development history** — repo public > 6 months with active, iterative
   development (not a single burst). *Confirm the public history satisfies this.*
2. **Demonstrated research impact** — evidence the software is used for research
   (by the author at minimum). *This is the biggest thing to gather; see below.*
3. **Good open-source practices** — tests, CI, changelog, docs, CONTRIBUTING,
   stated support/governance. *Check which are present.*
4. **Iterative development over time** — ongoing refinement, not one dump.

## Suggested workflow

1. Resolve the metadata TODOs (ORCID, affiliation, date).
2. Draft **Summary** and **Statement of need** first (easiest, sets the framing).
3. Draft **Software design** (we have strong internal notes for this).
4. Assemble **State of the field** (needs a little research on alternatives).
5. Assemble **Research impact statement** (needs evidence — gather early).
6. Write **AI usage disclosure** and **Acknowledgements** last.
7. Verify all `.bib` DOIs resolve; add ORCID.
8. Compile locally / via the Open Journals GitHub Action to confirm it renders and
   stays within the word limit.
9. Confirm a tagged release exists (v1.0.0 present; decide 1.0 vs 1.1 target).

> STATUS: All sections DRAFTED (v1) in paper.md — Summary, Statement of need, State of
> the field, Software design, Research impact statement, AI usage disclosure,
> Acknowledgements. Body is ~1420 words (within the 750–1750 limit).
> Remaining: author review of drafts; resolve the confirmation items below; compile.

## Open questions / information to gather, by section

### Metadata
- [x] Author ORCID iD — 0009-0007-1801-5517.
- [x] Author affiliation — Toole Design Group, LLC, United States (https://tooledesign.com/).
- [ ] Optional: ROR ID for Toole Design (not yet added).
- [ ] Additional co-authors? (Anyone with non-trivial non-code contributions.)
- [x] Target release: **linref 1.0** (keep paper prose version-agnostic; submit against
      the latest stable tag at submission time — likely 1.0.1 once the bugfix lands).
      Do not frame the paper around 1.1 features so they don't distract from the v1 release.
- [ ] Target submission date: **31 August 2026** (placeholder) — **update `date:` in
      paper.md to the actual submission day before submitting.**

### Summary
- [ ] One-sentence, jargon-free definition of linear referencing for a general reader.
- [ ] The 3–4 headline capabilities to highlight (dissolve, resegment, relate/overlay
      conflation, geometry↔measure).

### Statement of need
- [ ] A concrete motivating example (e.g., conflating crash points + pavement segments
      + traffic onto a common route/measure framework).
- [ ] Precise target audience (state/local DOT analysts, transportation researchers,
      safety analysts, GIS practitioners).
- [ ] Which pain points in current practice to emphasize (manual GIS steps,
      non-reproducibility, mismatched segmentation).

### State of the field
- [ ] Confirm the proprietary comparison (Esri Roads & Highways / ArcGIS linear
      referencing) and describe its limitations fairly.
- [x] Other open-source LRS packages: the source review found **no** general-purpose
      open-source linear-referencing package to compare against — this absence is itself
      part of the build-vs-contribute justification. (State it plainly.)
- [ ] Articulate the "build vs. contribute" argument (extend pandas/geopandas rather
      than reinvent).
- [x] Method-comparison research is an established genre (cheng2005experimental,
      kwon2013evaluating) — frame linref as reproducible *tooling* for it, not new theory.

### Software design
- [ ] Confirm architecture description: `LRS` schema object, `.lr` accessor,
      `EventsData` core (numpy / scipy.sparse), `LineStringM` M-enabled geometry,
      chaining for non-contiguous segments.
- [ ] Pick 2–3 design trade-offs to discuss (accessor vs. subclass; measure-space vs.
      coordinate-space operations; chunking/caching for scale).
- [ ] Optional: one figure/diagram of the data model or a workflow.

### Research impact statement (DRAFTED — strong evidence)
- [x] Three confirmed, public SS4A engagements now anchor the section:
      Newport RI [newport2025sap], Eastgate OH [eastgate2024sap],
      MACOG IN [macog2025movesafe] (memos co-authored by the developer — strongest tier).
      Full detail in [RESEARCH_IMPACT.md](RESEARCH_IMPACT.md).
- [x] Added statewide-scale engagement: Texas Pedestrian Safety Action Plan
      [txdot2023psap], prepared for TxDOT by Jacobs — the largest test of the library,
      which motivated v1 scalability/performance/parameterization gains. Its "unusual
      segment breaks" issue maps directly onto linref's chaining feature.
- [x] TxDOT PSAP publication date confirmed: September 2023.
- [x] Community signals: repo stars/forks; sibling safety packages (`cpm`, `rangel`).
- [ ] CONFIRM: no client/contract restriction on naming the three plans (all are public
      government documents, so low risk) — quick check before submission.
- [ ] Version note handled in-text: engagements used v0.1.x; capabilities persist in v1.0
      (described by feature category, not version-specific method names).
- [ ] STRATEGIC (strongest possible boost): if the network-screening methods-comparison
      paper (TRR / Accident Analysis & Prevention) can be posted as a preprint before or
      alongside submission, cite it here. Sequence deliberately if timing allows.

### Pre-submission metadata note (from impact review)
- [x] `Development Status` classifier: current pyproject.toml already reads `4 - Beta`;
      the old "1 - Planning" only appears on the published v0.1.x PyPI release and will
      resolve when 1.0 is published. No action needed in-repo.

### AI usage disclosure
- [x] Drafted and expanded: GitHub Copilot with Claude Opus 4.8 and Sonnet 5. In the
      software — algorithm benchmarking/optimization, unit-test development, documentation
      refinement/grammar, and drafting some minor features. In the paper — drafting and
      reference research. Authors validated all output.

### Acknowledgements
- [ ] Now that Lucas Yu and Felipe Dias are co-authors, do NOT list them here.
- [ ] Consider thanking applied-project collaborators who are not co-authors (e.g.,
      Catherine Girves, Rahul Rajbhara, Katie Sieb, Ayden Cohen — confirm which).
- [ ] Consider thanking early users/testers who provided feedback, bug reports, or
      feature requests, and any GitHub issue/PR contributors.
- [ ] Employer support: acknowledge if Toole Design / Jacobs provided time or resources.
- [ ] Funding statement: confirm "no specific grant," or note the SS4A-funded projects
      that exercised the software (funded the analyses, not linref development).
- [ ] JOSS reviewers/editor are typically acknowledged after review — add later.

## References gathered so far

Domain / linear referencing:
- `scarponcini2002generalized` — Generalized Model for Linear Referencing in
  Transportation (GeoInformatica). Foundational peer-reviewed data model
  (linear element / referencing method / distance expression).
- `adams2001nchrp460` — NCHRP Report 460, multimodal location referencing guidelines.
  Cite as the *linear subset* of the broader MDLRS model (see caveats).
- `fhwa2014arnold` — FHWA ARNOLD reference manual (national all-roads LRS dataset);
  strongest terminology match (point event, route event, per-carriageway identity).

Road safety:
- `aashto2010hsm` — Highway Safety Manual, 1st ed. (network screening / sliding-window,
  peak searching, EB/PSI).
- `cheng2005experimental` — foundational hotspot-ID method-comparison paper.
- `kwon2013evaluating` — empirical SMW/PS/CRP comparison; the precedent-and-contrast
  citation (bespoke pipeline vs. reproducible tooling).

Software ecosystem:
- `mckinney2010pandas`, `jordahl2020geopandas`, `gillies2007shapely`,
  `harris2020numpy`, `virtanen2020scipy`.

### Open decisions flagged by the source review (RESOLVED)
- [x] **ISO 19148:2012 taxonomy** — keep framing generic. linref is technically based on
      absolute referencing but is configurable for the other approaches; do not map to
      Absolute/Relative/Interpolative and do not purchase/cite the standard.
- [x] **NCHRP 460 phrasing** — use "linear subset of" and keep the reliance light
      (it is broad data-model guidance); no class literally named "linear event."
- [x] **Vonderohe et al. (1997)** precursor — not needed; omit.
- [x] **ADOT LRS case study (FHWA-SA-22-006, 2022)** — superseded by ARNOLD as the
      practice authority; drop from citations. (Author may still review it separately.)

### Reference TODOs
- [ ] Verify every DOI resolves and metadata (volume/pages/year) is exact.
- [ ] Update `jordahl2020geopandas` / `gillies2007shapely` to the version-specific
      Zenodo DOI matching the release actually used, if desired.
- [ ] Consider adding a citation for the sliding-window / network-screening method
      beyond the HSM if a specific paper is preferred.
- [ ] Decide whether to cite any prior linref release or its documentation.
