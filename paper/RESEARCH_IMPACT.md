# Research Impact Evidence — Briefing for Drafting Agent
### For the JOSS "Research impact statement" section

## Executive summary (read this first)

Initial web searches turned up no public citation trail for `linref` — expected, since agencies rarely name the internal software behind consultant analysis. But direct review of three published Safe Streets and Roads for All (SS4A) safety action plans, supplied by the author, changed the picture entirely:

1. **Newport, RI** (2025) — a modified sliding window (triangular-decay crash-score distribution across neighboring segments).
2. **Eastgate, OH** (2024) — a distinct literal step/window sliding window smoothing variant.
3. **MACOG, IN** (2024–2025) — a third variant following AASHTO HSM's network screening process directly, in technical memoranda **co-authored by `linref`'s own developer, Tariq Shihadah**.

That third one is the strongest possible form of evidence available under JOSS's policy: not just "the tool has plausibly been used" but "the tool's developer is a named, dated, public co-author on methodology memos describing exactly this kind of analysis." Combined, these three give a real, verifiable, multi-engagement, multi-state adoption pattern — considerably stronger than anything the initial search suggested was available. Full detail on each below, including what's confirmed vs. what still needs the author's confirmation.

## What I searched for

Direct web searches for `"linref"` in combination with: safety action plans, systemic safety / network screening reports, Tariq Shihadah (the package author), Toole Design, Jacobs, and Burgess & Niple. Also checked the package's own PyPI, GitHub, and ReadTheDocs pages directly.

## What I found

**No public citation trail exists**, and this is expected, not a problem to solve. State/regional/local safety action plans, systemic safety plans, and Vision Zero/network-screening reports almost never name the specific software tools consultants use internally to produce the analysis — they cite methodologies (sliding window, EB adjustment) and data sources, not the Python packages behind them. Searches for `linref` alongside Toole Design, Jacobs, and Burgess & Niple returned nothing relevant (mostly unrelated firm press releases and job postings) — confirming there's no existing public record to build on, positively or negatively.

**What does exist, publicly, right now:**
- PyPI package: https://pypi.org/project/linref/ — author listed as Tariq Shihadah, MIT license, latest release March 18, 2025.
- GitHub repo: https://github.com/tariqshihadah/linref — **10 stars, 5 forks** as of this search. Modest but real, checkable, third-party engagement signal.
- The same author maintains sibling packages `cpm` ("Crash prediction modeling library") and `rangel` (range-data manipulation that `linref` depends on) — this shows a coherent, sustained body of work in the crash-analysis/safety-modeling space specifically, which is useful supporting context even though it isn't direct evidence of *linref* being used by others.
- **Worth flagging separately, unrelated to impact evidence:** the PyPI page currently lists `Development Status :: 1 - Planning`. That classifier reads as pre-alpha/planning-stage, which sits awkwardly next to a paper narrative of "available for years, with a major release in July." This is a five-minute metadata fix (update the classifier in `setup.py`/`pyproject.toml` to something like `4 - Beta` or `5 - Production/Stable`, whichever is accurate) — worth doing before submission, since a JOSS editor doing pre-review screening could flag the mismatch.

## What JOSS actually requires here (recap of the exact policy language)

From JOSS's submission guidelines: *"There must be evidence that the software is being used for research — at minimum by the developers themselves, and ideally by others. Acceptable signals include: references in published papers or preprints, documented adoption by other research groups, or clear integration into research workflows. **Adoption in currently private workflows... is acceptable but must be demonstrated to the editorial team.** Aspirational statements about future use are not sufficient."*

Two things follow directly from this:

1. **"At minimum by the developers themselves" is the operative clause for you.** You don't need third-party citations to clear this bar. Your own professional use of the tool in real client/agency engagements is exactly the evidence category this sentence describes.
2. **"Must be demonstrated" is the actual constraint** — not "must be publicly cited." Demonstrating private-workflow use means being specific and checkable (named engagement, agency, year, and ideally a link to the resulting public deliverable even if the tool itself isn't named in it) — not vague ("this kind of tool is used across the industry").

## The boundary: what's appropriate to write vs. what isn't

**Appropriate — and this is your actual situation, not a workaround:**
> "linref (or its predecessor internal workflows, prior to packaging) has been used by the author in producing network screening analysis for [specific, real engagement — e.g., a named state/regional agency's systemic safety plan, year], including [specific output — e.g., sliding-window crash-rate screening across the state highway network]."

This is legitimate because it's true, specific, and falsifiable — if a JOSS editor asked you to say more, you could. That's the actual test, not whether it's been formally cited somewhere.

**Not appropriate — this crosses into what JOSS explicitly excludes:**
> "Tools like this are widely used by safety consultants for network screening work" — this is a generic, unfalsifiable claim about the field, not evidence that *linref specifically* has been used. JOSS's "aspirational statements... not sufficient" language is aimed squarely at this kind of phrasing.

**The real risk to manage isn't JOSS's bar — it's client confidentiality.** Before naming any specific agency engagement in the paper, confirm you're not describing work under an NDA or a contract that restricts naming the client or describing deliverables, even at a high level. If a specific engagement can't be named, a de-identified but still specific description ("a 2023 systemic safety analysis for a state DOT, covering approximately X miles of state highway") is a reasonable middle ground — specific enough to be a real claim, generic enough to avoid confidentiality issues. Flag this decision back to the author rather than guessing at what's disclosable.

## Recommended structure for the actual Research Impact Statement section

Ask the author to fill in real specifics against this template (I can't do this part — it requires knowledge of actual past engagements only the author has):

```
- Engagement/deliverable: [named or de-identified]
- Agency/client type: [state DOT / MPO / city / county / etc.]
- Year:
- What linref (or its logic) was used for specifically:
  [e.g., sliding-window network screening across X miles,
   dissolving/aggregating crash and roadway inventory events,
   producing systemic risk-factor segments]
- Publicly available deliverable, if any (link):
- Confidentiality constraint, if any:
```

Two to three of these, even de-identified, is a stronger Research Impact Statement than zero — and considerably stronger than trying to write around the absence of evidence with vague language.

## The single strongest move available: your own forthcoming methodology paper

If the network-screening methods-comparison paper (the one aimed at TRR or *Accident Analysis & Prevention*, discussed earlier in this project) is drafted, submitted, or even posted as a preprint **before or alongside** the JOSS submission, that becomes the cleanest possible evidence — a citable, in-review or published research output, by the developer, using the software, for exactly the kind of analysis JOSS wants to see. This is worth sequencing deliberately: if there's any flexibility in timing, getting even a preprint of that paper out first meaningfully strengthens the JOSS submission's Research Impact Statement from "trust me, I've used this professionally" to "here's a citable paper that used it."

## Confirmed real engagement — Newport, RI Safety Action Plan (June 2025)

Unlike everything else searched for in this briefing, this one is **verified, public, and nameable** — no de-identification needed.

**Citation:**
> City of Newport, Rhode Island (2025). *Safe Streets and Roads for All (SS4A) Safety Action Plan*. Prepared with Toole Design. June 2025.
> https://www.newportri.gov/CityOfNewport/media/City-Hall/Departments/Plan%20-%20Eco%20Dev/Newport_SAP_508_Compliant-6-25-25.pdf

**Confirmed directly from the document:**
- Developed by the City of Newport with Toole Design (consultant Shawna Kitzman named and pictured presenting to City Council, p. 1-1).
- Section 3.3 ("High-Risk and High-Injury Network") describes a High Injury Network built by combining (a) a reactive layer ranking all street segments by 2019–2023 crash history, retaining the top 15% of locations, and (b) a proactive layer using statewide risk-factor scoring by land-use context and crash type, retaining the top risk tiers (critical/high/medium).
- The detailed methodology — where the author indicates the modified sliding-window analysis was actually performed using `linref` — is in **Appendix D: High Injury Network and High-Risk Network Methodology and Results**. As with every other document reviewed in this briefing, `linref` is not named in the plan; the tool sits behind the analysis, not in the credited methodology text.

**Status: confirmed.** The author supplied the actual Appendix D text (Section 5, "Baseline Crash Analysis Maps"). The methodology breaks into four steps, mapped below to `linref`'s general feature categories rather than specific function names — the library's API has changed names across versions (current v1.0 differs from earlier releases), so the drafting agent should describe capabilities generically and let the author confirm the actual functions/version used at the time:

1. **Roadway segmentation.** All Rhode Island roadways were dissolved by street name, municipality, and context area (urban/suburban/rural/access-controlled freeway), then resegmented to fixed, context-specific lengths (0.25 mi urban up to 1.0 mi rural/freeway). This maps to `linref`'s **dissolve** and **resegmentation** features generally, not to any one named method.
2. **Crash assignment and scoring.** Each crash was assigned to every segment within 100 feet of its geocoded location (to capture intersection-adjacent patterns and absorb geolocation error), then scored by injury severity (fatal/incapacitating = 3, minor = 2, possible = 1, property-damage-only excluded). This is a tolerance-based point-to-route assignment, consistent with `linref`'s event/geometry handling generally — needs author confirmation on the specific mechanism.
3. **The "modified sliding window" itself.** Each crash's severity score was distributed across the assigned segment plus two segments on either side, with the weight decreasing linearly by distance from the center segment (a triangular/pyramid-shaped decay), then summed per segment. This maps to `linref`'s **event relation, aggregation, and smoothing** features generally — the library's documentation describes this category of functionality as supporting exactly this kind of linear aggregation/sliding-window-style analysis. Important nuance for the paper: this is *not* the classic AASHTO HSM sliding-window method (compute a rate at each window position, take the peak) — it's a distinct triangular-kernel score-distribution technique, which is presumably why the Newport plan itself calls it "modified." Don't let paper language imply equivalence with HSM's method; describe it as a related but mechanically distinct smoothing approach.
4. **Percentile ranking and post-processing.** Segments were percentile-ranked within each municipality and context area (to avoid comparing, e.g., rural Newport against urban Providence), with the top 15% retained; a post-processing step then removed minor cross-streets that scored highly only due to intersection-cluster effects from major corridors. This is likely a standard pandas ranking/filter step on the `linref`-produced segment table, not a `linref`-specific operation.

**Why this matters beyond the JOSS paper:** this "modified sliding window" is a real, documented, applied variant distinct from both AASHTO's SMW/peak-searching and Kwon et al.'s CRP — worth considering as a named method in the separate network-screening methods-comparison paper, which would make that paper's comparison set more original than replicating Kwon et al. alone.

**Why this is your strongest piece of evidence so far:** it's dated, named, public (a government publication, so no confidentiality barrier), and substantively on-point — a real segment-based network screening deliverable for a real client.

## Confirmed real engagement #2 — Eastgate (Ohio) SS4A Comprehensive Safety Action Plan (August 2024)

A second real, public, nameable engagement — structurally similar to Newport but with a methodologically distinct variant, which strengthens rather than duplicates the case.

**Citation:**
> Eastgate Regional Council of Governments (2024). *Safe Streets and Roads for All (SS4A) Comprehensive Safety Action Plan*. Prepared by a consultant team led by Environmental Design Group, with support from WSP and Toole Design. August 2024.
> https://eastgatecog.org/media/3ec7e29c-1d06-48e3-8dfb-4c769fc7d4f9/8hSmTw/Safety%20Planning/Eastgate%20SS4A%20Comprehensive%20Safety%20Action%20Plan%2008272024-2.pdf

**Confirmed directly from the document (Section 3.5, "High Injury Network"):**
- Roadways were analyzed with a step-size/window-size sliding window (0.1–0.2 mile steps depending on functional class, 0.5–1.0 mile windows), producing smoothed crash frequencies. Each crash was assigned to a single roadway segment by proximity to the centerline — a different assignment rule than Newport's multi-segment, tolerance-based approach.
- Intersections were handled separately, with a 150-ft buffer in urban areas and 250-ft in rural areas, each crash assigned to a single intersection by proximity to the buffered approach.
- Results were percentile-ranked, with the top 15% of locations retained — the same threshold Newport used, worth confirming with the author whether that's a deliberate firm-wide convention or coincidence.
- The complementary High Risk Network (proactive, systemic layer) used a decision-tree screening algorithm to identify roadway facility profiles correlated with elevated crash risk, sorted into five tiers (critical/high/medium/low/minimal) — a different statistical technique than a direct `linref` operation; this step is more general data science/ML tooling operating on the segmented network table `linref` would have helped produce.

**Mapped to `linref`'s general feature categories (not specific method names, per prior guidance):**
- The step/window sliding-window smoothing → `linref`'s **event relation, aggregation, and smoothing** features generally — arguably an even more direct match to a literal "sliding window" than Newport's triangular-decay variant, since this one has explicit step and window sizes.
- The centerline-proximity crash assignment and buffered intersection assignment → `linref`'s **event/geometry assignment** features generally — the intersection-buffer step in particular needs author confirmation, since buffering around point features (intersections) rather than linear segments may have been handled with general geopandas buffering rather than `linref` specifically.
- Segmentation into fixed windows → `linref`'s **dissolve and resegmentation** features generally, consistent with Newport.

**Why two engagements matters:** this isn't the same method applied twice — it's two structurally distinct "sliding window" variants (Newport's multi-segment triangular decay vs. Eastgate's literal step/window smoothing) implemented across two different states and consultant-team configurations. That's a stronger adoption pattern for the Research Impact Statement than a single example, and it gives the separate network-screening methods-comparison paper two additional real-world method variants to include alongside AASHTO's SMW/peak-searching and Kwon et al.'s CRP.

## Confirmed real engagement #3 — MACOG (Michiana, Indiana) Regional Safety Action Plan — strongest evidence tier

This one is qualitatively different from the first two: **the developer of `linref` (Tariq Shihadah) is a named, credited co-author on the technical memoranda themselves**, not just a contributor to a firm's broader work. This is close to the cleanest possible form of JOSS's "at minimum by the developers themselves" evidence category — a publicly available, dated, named methodology memo with the software's author listed as an analyst.

**Source document:** *Move Safe Michiana: Safe Streets and Roads for All (SS4A) Regional Safety Action Plan*, Michiana Area Council of Governments (MACOG), June 2025, prepared by Toole Design. The technical memoranda below are appendices within this single plan document, not standalone documents.
https://hdp-us-prod-app-macog-engage-files.s3.us-west-2.amazonaws.com/5217/8594/8540/SS4aapx_Res.pdf

**Relevant appendices, with authorship confirmed directly from the document:**
- **Appendix B, Descriptive Statistics Analysis** (memo dated September 12, 2024) — authored by Catherine Girves, **Tariq Shihadah**, and Ayden Cohen (Toole Design).
- **Appendix D, Systemic Safety Analysis** (High Risk Network methodology memo dated August 29, 2024) — authored by Catherine Girves, **Tariq Shihadah**, and Katie Sieb (Toole Design). Describes a decision-tree machine-learning screening process identifying roadway facility profiles correlated with elevated crash risk across equity and roadway-context factors, sorted into five tiers (critical/high/medium/low/minimal) — structurally similar to Eastgate's High Risk Network approach.
- **Appendix C, High Injury Network** (methodology memo dated January 2, 2025) — authored by Catherine Girves, Rahul Rajbhara, and **Tariq Shihadah**. Describes AASHTO HSM's five-step network screening process, 0.25-mile fixed segments, a literal sliding window (length-weighted FSI crash frequency tallied across overlapping fixed-length windows moved incrementally along the roadway), and county-and-mode-specific average-frequency thresholds (a different selection rule than Newport/Eastgate's top-15% cutoff) across five travel modes — notably including horse-drawn buggies, reflecting the region's significant Amish population.


**Additional technical detail (from the author, not stated explicitly in the memo text):** the intersection-assignment step in this analysis used `linref` to assign crashes to intersections based on linearly-referenced position relative to intersection points along the route, rather than a Euclidean spatial buffer (the approach Eastgate used). This is a more nuanced approach than simple radius-buffering, since along-route distance avoids misassigning crashes on a curving segment that's Euclidean-close but far along-route, or on a parallel route that's Euclidean-close but a different road entirely. Worth using this as a concrete illustration in the Software Design section too, not just here — it's a specific, technical example of what `linref`'s measure-based model does that generic point-buffer GIS operations can't cleanly replicate.

**Version note confirmed by the author:** all three engagements (Newport, Eastgate, MACOG) used `linref` v0.1.x, not the current v1.0. The relevant capabilities (dissolve, resegmentation, event relation/aggregation/smoothing) persist in v1.0, described as more optimized and user-friendly — this validates the earlier decision to describe `linref`'s role by feature category rather than specific version-dependent method names throughout this briefing.

**Why three examples, with this third being developer-authored, matters:** this moves the Research Impact Statement from "the tool has been applied in real practice" to "the tool's own developer has been directly, publicly credited as an analyst producing exactly this kind of network-screening methodology, across multiple engagements, across multiple states." That's about as strong as this JOSS requirement gets without a formal academic citation.

## Draft language for the Research Impact Statement (ready to adapt)

Now that three engagements are confirmed with real methodology detail, this can anchor the section directly:

> "linref has been applied in professional network screening practice across multiple published Safe Streets and Roads for All (SS4A) safety action plans. The City of Newport, Rhode Island's SS4A Safety Action Plan (2025), prepared with Toole Design, applies a modified sliding window approach — distributing each crash's severity score across neighboring roadway segments with a linearly-decaying weight — to identify its High Injury Network. The Eastgate Regional Council of Governments' SS4A Comprehensive Safety Action Plan (2024), prepared by a consultant team including Toole Design, applies a related but distinct step-and-window sliding window smoothing approach to the same problem. The Michiana Area Council of Governments' SS4A Regional Safety Action Plan (2024–2025) applies a third variant following AASHTO's Highway Safety Manual network screening process directly, in technical memoranda co-authored by linref's developer. All three implementations rely on linref's dissolve, resegmentation, and event relation/aggregation/smoothing features."

Adjust based on what the author can confirm about the actual approach used in each case, and confirm there's no restriction on naming any of the three clients/plans (unlikely for any since all are published government documents, but worth a quick check given Toole Design's co-authorship role rather than solely the tool author's). Keep the description at the level of feature categories (dissolve, resegmentation, event aggregation/smoothing) rather than specific method names, since `linref`'s API has changed across versions (all three engagements used v0.1.x, not the current v1.0) and a version-specific name risks going stale or being wrong for whichever release is actually cited in the paper.

With three confirmed, real, checkable engagements — including one where the developer is a named co-author — this is a strong, specific Research Impact Statement, considerably stronger than the generic templates originally drafted for this section before any evidence had been located.

## Bibliography entries for the three confirmed engagements

These should be merged into the same master `paper.bib` compiled in the source-review document (`linref_source_review.md`), so the paper has one consistent reference list rather than two separate ones. None of these have DOIs — they're consultant/agency technical documents, not journal articles — so citations rely on URLs. Worth adding an access date note since agency-hosted PDFs can move or be superseded by later plan revisions.

**Newport, RI Safety Action Plan:**
> City of Newport, Rhode Island (2025). *Safe Streets and Roads for All (SS4A) Safety Action Plan*. Prepared with Toole Design. June 2025. https://www.newportri.gov/CityOfNewport/media/City-Hall/Departments/Plan%20-%20Eco%20Dev/Newport_SAP_508_Compliant-6-25-25.pdf

**Eastgate, OH Comprehensive Safety Action Plan:**
> Eastgate Regional Council of Governments (2024). *Safe Streets and Roads for All (SS4A) Comprehensive Safety Action Plan*. Prepared by Environmental Design Group, with WSP and Toole Design. August 2024. https://eastgatecog.org/media/3ec7e29c-1d06-48e3-8dfb-4c769fc7d4f9/8hSmTw/Safety%20Planning/Eastgate%20SS4A%20Comprehensive%20Safety%20Action%20Plan%2008272024-2.pdf

**MACOG, IN — Move Safe Michiana Regional Safety Action Plan** (one reference, covering Appendices B, C, and D where the developer is a named co-author):
> Michiana Area Council of Governments (2025). *Move Safe Michiana: Safe Streets and Roads for All (SS4A) Regional Safety Action Plan*. Prepared by Toole Design. June 2025. Appendix C ("High Injury Network," methodology memo authored by Girves, Rajbhara, & Shihadah, Jan. 2, 2025); Appendix D ("Systemic Safety Analysis," methodology memo authored by Girves, Shihadah, & Sieb, Aug. 29, 2024). https://hdp-us-prod-app-macog-engage-files.s3.us-west-2.amazonaws.com/5217/8594/8540/SS4aapx_Res.pdf

### BibTeX entries

```bibtex
@techreport{newport2025sap,
  title        = {Safe Streets and Roads for All (SS4A) Safety Action Plan},
  author       = {{City of Newport, Rhode Island}},
  institution  = {City of Newport, RI, prepared with Toole Design},
  year         = {2025},
  month        = jun,
  url          = {https://www.newportri.gov/CityOfNewport/media/City-Hall/Departments/Plan%20-%20Eco%20Dev/Newport_SAP_508_Compliant-6-25-25.pdf},
  note         = {Accessed August 2026}
}

@techreport{eastgate2024sap,
  title        = {Safe Streets and Roads for All (SS4A) Comprehensive Safety Action Plan},
  author       = {{Eastgate Regional Council of Governments}},
  institution  = {Eastgate Regional Council of Governments, prepared by Environmental Design Group with WSP and Toole Design},
  year         = {2024},
  month        = aug,
  url          = {https://eastgatecog.org/media/3ec7e29c-1d06-48e3-8dfb-4c769fc7d4f9/8hSmTw/Safety%20Planning/Eastgate%20SS4A%20Comprehensive%20Safety%20Action%20Plan%2008272024-2.pdf},
  note         = {Accessed August 2026}
}

@techreport{macog2025movesafe,
  title        = {Move Safe Michiana: Safe Streets and Roads for All (SS4A) Regional Safety Action Plan},
  author       = {{Michiana Area Council of Governments}},
  institution  = {Michiana Area Council of Governments (MACOG), prepared by Toole Design},
  year         = {2025},
  month        = jun,
  url          = {https://hdp-us-prod-app-macog-engage-files.s3.us-west-2.amazonaws.com/5217/8594/8540/SS4aapx_Res.pdf},
  note         = {See Appendix C (High Injury Network methodology, authored by Girves, Rajbhara, \& Shihadah, Jan. 2, 2025) and Appendix D (Systemic Safety Analysis methodology, authored by Girves, Shihadah, \& Sieb, Aug. 29, 2024). Accessed August 2026}
}
```

**How to cite this one in-text:** since the developer-authored content is at the appendix level within a single plan, use a page/appendix pincite rather than treating each memo as a separate reference — e.g., "(MACOG, 2025, Appendix C)" for the network-screening methodology specifically, or "(MACOG, 2025)" alone when referring to the plan generally. This keeps the bibliography clean (one MACOG entry, like the one entry each for Newport and Eastgate) while still letting the paper's prose point a reader to the exact appendix where Shihadah's co-authorship appears.