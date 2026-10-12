# Source Review — `linref` JOSS Paper
### Briefing for Statement of Need / State of the Field sections

**Purpose of this document:** This is a handoff briefing summarizing the research and vetting done on six sources identified for the `linref` JOSS submission. It covers what each source is, why it's relevant, how it maps onto `linref`'s actual concepts, and where/how it should be used in the paper. Use this to draft the **Statement of need** and **State of the field** sections specifically, though several sources also support **Software design**.

Reminder of what JOSS expects in the two target sections:
- **Statement of need**: clearly illustrates the research purpose, places `linref` in the context of related work, states what problems it solves and who the audience is.
- **State of the field**: how `linref` compares to other commonly-used packages/approaches; if alternatives exist, a "build vs. contribute" justification for why they're insufficient.

Two source groups:
- **Group A — Safety analysis / network screening** (supports the *motivating use case* half of Statement of Need — why network screening matters and is done a specific way)
- **Group B — Linear referencing systems / schema** (supports the *core data model* half of Statement of Need, and most of State of the Field)

---

## GROUP A: Safety Analysis / Network Screening

### A1. AASHTO (2010) — *Highway Safety Manual*, 1st Edition

**Citation:**
> American Association of State Highway and Transportation Officials. (2010). *Highway Safety Manual* (1st ed.). Washington, D.C.: AASHTO.

**What it is:** The field's standard practice reference for quantitative roadway safety analysis. Defines the vocabulary and procedures nearly every downstream paper and agency workflow uses.

**Key facts verified:**
- Recommends **sliding window** and **peak searching** as the two primary methods for identifying roadway segments for network screening, using recommended performance measures; recommends **simple ranking** for intersections/facilities.
- Establishes **Empirical Bayes (EB) adjustment** as the standard way to compute potential-for-safety-improvement (PSI) from observed vs. predicted crash frequency.
- Defines the **coefficient of variation (CV)** precision test used within peak searching (CV limiting value of 0.5 is the HSM-recommended threshold).

**Relevance to `linref`:** This is the citation for *why* sliding-window/peak-searching network screening is a defined, standardized analytical task in the first place — it's the reason a linear-referencing library would build tooling around this workflow. Use it wherever the paper names or defines these methods, and again for EB adjustment if the paper's example touches PSI calculation.

**Suggested use:** Opening sentence(s) of Statement of Need, establishing the analytical context `linref` is designed to support: *"Roadway safety network screening — identifying segments where crash risk exceeds expectation — is a standardized practice built around methods such as sliding window and peak searching analysis (AASHTO, 2010)..."*

---

### A2. Cheng & Washington (2005) — "Experimental evaluation of hotspot identification methods"

**Citation:**
> Cheng, W., & Washington, S. P. (2005). Experimental evaluation of hotspot identification methods. *Accident Analysis & Prevention*, 37(5), 870–881. https://doi.org/10.1016/j.aap.2005.04.007

**What it is:** The foundational peer-reviewed academic paper establishing that *comparing* network-screening/hotspot-ID methods empirically is itself a legitimate, established research question — not something the `linref` paper is inventing as a framing device. Nearly every later comparison study (including A3, below) cites this as its starting point.

**Relevance to `linref`:** Supports the **State of the Field** claim that empirical methods-comparison work is an existing research thread, and that `linref` offers new *tooling* for doing this kind of comparison reproducibly rather than introducing a new kind of research question.

**Suggested use:** State of the Field, first sentence establishing that method-comparison studies are a known genre: *"Comparative evaluation of network screening methods has an established research lineage (Cheng & Washington, 2005; Kwon et al., 2013)..."*

---

### A3. Kwon, Park, Yeo, & Chung (2013) — "Evaluating the performance of network screening methods for detecting high collision concentration locations on highways"

**Citation:**
> Kwon, O. H., Park, M. J., Yeo, H., & Chung, K. (2013). Evaluating the performance of network screening methods for detecting high collision concentration locations on highways. *Accident Analysis & Prevention*, 51, 141–149. https://doi.org/10.1016/j.aap.2012.11.006

**What it is:** A direct empirical comparison of Sliding Moving Window (SMW), Peak Searching (PS), and Continuous Risk Profile (CRP) methods, using EB-adjusted PSI against a confirmed hot-spot list from California freeway data. Structurally, this is almost exactly what a `linref`-based case study would replicate — same three-method comparison, same EB/PSI framework — just done in a bespoke, non-reproducible pipeline (California-specific SPFs, custom GIS/statistics workflow, no shared tooling).

**Key facts verified:**
- All three methods take the same inputs (traffic volume, collision data, SPF) but differ only in how they determine segment/site endpoints.
- CRP had the lowest false-positive rate and best hot-spot detection efficiency across all tested conditions; SMW/PS results were more sensitive to segment-length and SPF choice.
- The paper explicitly notes SMW and PS "can converge to CRP" under certain post-processing choices — a nuance worth being aware of if `linref`'s paper implies the three methods are cleanly distinct.

**Relevance to `linref`:** This is your best "here's how it's been done before, here's what doing it in `linref` adds" citation. It validates that the comparison your library enables is a real, previously-published line of inquiry — and gives you a natural contrast point: this kind of study currently requires a custom one-off pipeline; `linref` is meant to make it a repeatable, general-purpose capability.

**Suggested use:** State of the Field, as the direct precedent-and-contrast citation: *"...for example, Kwon et al. (2013) conducted such a comparison for California freeways using a bespoke analysis pipeline; `linref` is designed to make this kind of comparison reproducible and general-purpose rather than agency- or study-specific."*

**Caveat for the drafting agent:** Don't imply SMW/PS/CRP are `linref`'s only supported methods unless that's actually true of the current release — check against the library's actual feature set before naming specific methods as "supported."

---

## GROUP B: Linear Referencing Systems / Schema

### B1. Adams, Koncz, & Vonderohe (2001) — NCHRP Report 460

**Citation:**
> Adams, T. M., Koncz, N. A., & Vonderohe, A. P. (2001). *Guidelines for the Implementation of Multimodal Transportation Location Referencing Systems* (NCHRP Report 460). Washington, D.C.: Transportation Research Board, National Research Council.

**What it is:** The most detailed, most-cited academic/technical data model for linear (and non-linear) transportation referencing systems. Developed at the University of Wisconsin–Madison under NCHRP Project 20-27(3); defines a full UML object model (the "MDLRS" model) intended for adoption by agencies, standards bodies, and GIS-T vendors.

**Verified concept mapping to `linref`:**

| `linref` concept | NCHRP 460 equivalent | Match quality |
|---|---|---|
| Reference line / route | Traversal, built on anchor points/anchor sections | Conceptual match; NCHRP splits this into two layers `linref` collapses into one |
| Measure | `traversal_measure` attribute | Direct match |
| Point event | `Point_Event` class (has `traversal_measure`, references a `Traversal_Reference_Point`) | Near-exact — same structure, same name |
| Linear event | Modeled as a `from_position`/`to_position` association on an Anchor_Section, not a named class | Concept present, **not a matching name** — see caveat |
| Geometry as optional/separable | Explicit separation of topological objects from geometric objects | Strong — this is the core shared principle |

**Important caveats for the drafting agent:**
1. **Scope mismatch.** NCHRP 460 is a *general multimodal, multidimensional* referencing model — it covers geocentric, horizontal, cadastral, vertical, **and** linear referencing together. `linref` implements only the linear subsystem. Phrase this as "`linref` implements the linear-referencing subset of the broader object model NCHRP 460 describes," not as a scope-equivalent citation.
2. **"Linear event" has no matching class name in NCHRP 460.** Don't write anything implying NCHRP 460 has a class literally called "linear event" — it doesn't. The *concept* (an interval located by two measures) is there, expressed as an association, not a first-class object.
3. **Two-layer vs. one-layer route model.** NCHRP 460 separates the coordinate-independent calibration backbone (anchor points/sections) from the applied measurement system (traversals/LRMs) built on it. `linref`'s "reference line" flattens this into one practical construct. This is a legitimate implementation simplification, not an inconsistency — but be precise rather than claiming structural equivalence.

**Suggested use:** State of the Field, as the primary schema-authority citation, with the "implements the linear subset of" framing above.

---

### B2. Scarponcini (2002) — "Generalized Model for Linear Referencing in Transportation"

**Citation:**
> Scarponcini, P. (2002). Generalized Model for Linear Referencing in Transportation. *GeoInformatica*, 6(1), 35–55. https://doi.org/10.1023/A:1013716130838

**What it is:** A peer-reviewed journal treatment of the same problem NCHRP 460 addresses, but framed more abstractly. Formalizes three primitives: the **linear element** (the thing being measured), the **linear referencing method** (how it's measured), and the **distance expression** (the resulting measurement) — combined into a **location expression**. Later work (Curtin et al., 2007, and others) builds directly on this formalization; it's also cited in ARNOLD's technical appendix (via a 2012 GIS-T presentation by Scarponcini himself) as the basis for the ISO 19148:2012 LRM classification — see B3 caveat below.

**Verified concept mapping to `linref`:**

| `linref` concept | Scarponcini equivalent |
|---|---|
| Reference line / key group | Linear element |
| `LRS` schema object | Linear referencing method |
| Measure value | Distance expression |
| Fully specified event (key + measure) | Location expression |

This is arguably the *cleanest* conceptual match of any source reviewed — the three-part decomposition maps almost one-to-one onto `linref`'s design.

**Note on access:** Full text is paywalled (Springer); this review is based on the verified abstract and multiple independent secondary citations (GIS&T Body of Knowledge, later papers citing Scarponcini directly), which consistently describe the same three-primitive structure. Confident in the mapping, but the drafting agent should not lift specific phrasing beyond what's in the abstract without independently confirming against the full text if precise quotation is needed.

**Suggested use:** State of the Field, as the peer-reviewed (as opposed to agency-report) counterpart to NCHRP 460 — useful for satisfying the "cite venues by full name, not abbreviations" JOSS requirement with an actual journal source. Good sentence anchor: *"...the same decomposition Scarponcini (2002) formalizes as linear element, referencing method, and distance expression."*

---

### B3. Federal Highway Administration / Applied Geographics, Inc. (2014) — ARNOLD Reference Manual

**Citation:**
> Federal Highway Administration. (2014). *All Public Roads Geospatial Representation Study: All Roads Network of Linear Referenced Data (ARNOLD) Reference Manual*. Washington, D.C.: U.S. Department of Transportation.
> *(Note: sponsored by FHWA; prepared under contract by Applied Geographics, Inc., with David R. Fletcher and Michael Baker Jr., Inc. as subcontractors. FHWA-as-author is the conventional citation form used elsewhere in the literature — kept as-is.)*

**What it is:** Current, practically-oriented federal guidance (not a strict standard) for how state DOTs should build and maintain statewide all-roads linear referencing systems to meet HPMS reporting requirements. This is the "current practice" anchor of the three schema sources — closest in register to how practitioners (and `linref`'s own docs) actually talk about this domain.

**Verified concept mapping to `linref`** (this is the strongest match found across all three schema sources):

| `linref` concept | ARNOLD equivalent | Notes |
|---|---|---|
| Point event | "Point event" (verbatim term used) | Exact terminology match |
| Linear event | "Route event" covering an interval (e.g., pavement condition) | Same concept, ARNOLD's own term is "route event" not "linear event" — close but not verbatim |
| Reference line | Route-based network (continuous polyline, indexed by LRM from start to end) | `linref` operates at this layer specifically |
| `key_col` (compound identity) | Dual carriageway handling — same route needs independent identity per direction | Directly validates the design; ARNOLD explicitly recommends independent mileage calibration per carriageway |
| `chain_col` | Disconnected-geometry handling addressed via "match points" | Same underlying problem, different terminology |
| Geometry-optional/tabular events | "Additional features are stored as events that reference the LRS" — explicitly tabular, linked by measure | Direct match |

**Important finding — ISO 19148:2012:** ARNOLD states that per **ISO 19148:2012 ("Geographic information — Linear referencing")**, all LRMs fall into exactly three types: **Absolute** (distance from segment start, e.g., mile point/stationing), **Relative** (distance from a known reference location, e.g., mile post/reference post), and **Interpolative** (proportional distance, e.g., percentage/M-values). `linref`'s own docs describe measures generically without mapping explicitly onto this taxonomy.

- This is **not an inconsistency** — it's an open decision. Full ISO 19148 text was not accessible (paid standard) so I could not independently verify its content beyond ARNOLD's summary.
- **Decision needed from the author:** either (a) leave `linref`'s framing generic as-is, since it already implicitly covers "absolute" measures and, via `geom_m_col`, something like interpolative M-values; or (b) explicitly locate `linref` within this three-way taxonomy in the paper for added rigor. This is a judgment call, not a correctness issue — flag it to the paper's author before the drafting agent commits to language either way.
- If citing ISO 19148:2012 directly is desired, it would need to be sourced/purchased separately — I have not verified its content directly, only ARNOLD's description of it.

**Also verified — route-based vs. segment-based networks:** ARNOLD distinguishes intersection-based/segmented networks (broken at every intersection, used for routing/turn restrictions) from route-based networks (continuous polylines indexed by LRM). `linref` corresponds to the route-based layer; be precise in the paper that this is the layer being modeled, since ARNOLD covers both and a careless citation could imply `linref` addresses routing/topology it doesn't.

**Suggested use:** State of the Field, as the "current federal practice" citation — complements B1 (academic) and B2 (peer-reviewed journal) with actual agency-level guidance. Also usable in Statement of Need to establish that agencies are the real-world target audience: *"State DOTs building all-roads linear referencing systems under federal HPMS reporting requirements (FHWA, 2014) manage exactly this kind of route/measure/event structure..."*

---

## Sources considered but NOT included as primary citations

**ADOT LRS Data Case Study (FHWA-SA-22-006, 2022).** Retrieved during research as a secondary check (originally mistaken for the ARNOLD manual itself — since corrected). Independently uses very similar terminology (route, from/to measure, event, "route dominance" for compound route identity) and could serve as informal corroboration, but it's a single-agency case study, not authoritative guidance — don't cite it as a schema reference. Available if useful for a "real-world adoption" anecdote elsewhere in the paper (e.g., Research Impact Statement), not for State of the Field.

**Vonderohe, Chou, Sun, & Adams (1997) — NCHRP Research Results Digest 218**, "A Generic Data Model for Linear Referencing Systems." This is the direct precursor to NCHRP 460 (B1) — the earlier, shorter digest that NCHRP 460 itself builds on and cites repeatedly. Optional: include only if the paper wants to show historical lineage of the LRS data-model concept; otherwise NCHRP 460 alone supersedes it for citation purposes.

**ISO 19148:2012.** See B3 caveat above — flagged as a candidate authority, not independently verified, pending an author decision.

---

## Synthesis: how these six sources work together

The two groups map onto the two paper sections as follows:

- **Statement of Need** draws primarily on Group A (why network screening is a defined, standardized task worth building tooling for) plus B3/ARNOLD (who the real-world audience is and what data structure they actually manage).
- **State of the Field** draws primarily on Group B (B1 academic model, B2 peer-reviewed formalization, B3 current federal practice) to establish that `linref`'s core data model is a faithful, well-grounded implementation of established LRS theory — not an ad hoc invention — while A2/A3 establish that the *analytical* methods `linref` is built to support are similarly well-established rather than novel.

The through-line for both sections: `linref` isn't proposing new theory in either domain (LRS data modeling or network screening methodology) — it's providing the first general-purpose, reproducible **tooling** for well-established practices that currently live in bespoke agency workflows (ARNOLD's framing) or one-off academic study pipelines (Kwon et al.'s framing).

**Suggested synthesis sentence for State of the Field:**
> "`linref`'s data model — reference lines identified by key columns, measures expressed relative to those keys, and point/linear events located by one or two measures — implements the linear-referencing subsystem of the object model formalized in NCHRP Report 460 (Adams et al., 2001), the linear element/method/distance-expression decomposition proposed by Scarponcini (2002), and the route/measure/event structure that state DOTs are directed to build under current FHWA guidance (FHWA, 2014), adapted to the column-oriented structure of a pandas DataFrame rather than an agency-wide relational schema."

---

## Consolidated `paper.bib`

```bibtex
@techreport{aashto2010hsm,
  title        = {Highway Safety Manual},
  author       = {{American Association of State Highway and Transportation Officials}},
  institution  = {AASHTO},
  address      = {Washington, D.C.},
  year         = {2010},
  edition      = {1st}
}

@article{cheng2005experimental,
  title   = {Experimental evaluation of hotspot identification methods},
  author  = {Cheng, Wen and Washington, Simon P.},
  journal = {Accident Analysis \& Prevention},
  volume  = {37},
  number  = {5},
  pages   = {870--881},
  year    = {2005},
  doi     = {10.1016/j.aap.2005.04.007}
}

@article{kwon2013evaluating,
  title   = {Evaluating the performance of network screening methods for detecting high collision concentration locations on highways},
  author  = {Kwon, Oh Hoon and Park, Min Ju and Yeo, Hwasoo and Chung, Koohong},
  journal = {Accident Analysis \& Prevention},
  volume  = {51},
  pages   = {141--149},
  year    = {2013},
  doi     = {10.1016/j.aap.2012.11.006}
}

@techreport{adams2001nchrp460,
  title        = {Guidelines for the Implementation of Multimodal Transportation Location Referencing Systems},
  author       = {Adams, Teresa M. and Koncz, Nicholas A. and Vonderohe, Alan P.},
  institution  = {Transportation Research Board, National Research Council},
  address      = {Washington, D.C.},
  year         = {2001},
  number       = {NCHRP Report 460}
}

@article{scarponcini2002generalized,
  title   = {Generalized Model for Linear Referencing in Transportation},
  author  = {Scarponcini, Paul},
  journal = {GeoInformatica},
  volume  = {6},
  number  = {1},
  pages   = {35--55},
  year    = {2002},
  doi     = {10.1023/A:1013716130838}
}

@techreport{fhwa2014arnold,
  title        = {All Public Roads Geospatial Representation Study: All Roads Network of Linear Referenced Data (ARNOLD) Reference Manual},
  author       = {{Federal Highway Administration}},
  institution  = {U.S. Department of Transportation},
  address      = {Washington, D.C.},
  year         = {2014}
}
```

---

## Open items for the drafting agent to flag back to the author

1. **ISO 19148:2012 decision** — locate `linref`'s measure model within the Absolute/Relative/Interpolative taxonomy explicitly, or leave generic? (See B3.)
2. **Verify actual supported screening methods** before naming SMW/PS/CRP specifically in connection with `linref`'s current feature set (see A3 caveat).
3. **Precise phrasing required** for NCHRP 460 citation — must say "linear subset of" the broader MDLRS model, and must not imply a class literally named "linear event" exists in that model.
4. Confirm whether including the historical precursor (Vonderohe et al., 1997) adds value or is unnecessary given NCHRP 460 supersedes it.