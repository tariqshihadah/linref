---
title: 'linref: An Open-Source Python Library for Linearly Referenced Data Management, Engineering, and Analysis'
tags:
  - Python
  - linear referencing
  - geospatial
  - transportation
  - planning
  - pandas
  - geopandas
  - road safety
  - asset management
  - network screening
authors:
  - name: Tariq Shihadah
    orcid: 0009-0007-1801-5517
    affiliation: 1
  - name: Lucas Yu
    # orcid: TODO — add co-author ORCID
    affiliation: 1
  - name: Felipe Dias
    # orcid: TODO — add co-author ORCID
    affiliation: 2
affiliations:
  - index: 1
    name: Toole Design Group, LLC, United States
  - index: 2
    name: Jacobs, United States  # TODO: confirm Felipe's country/entity name
date: 31 August 2026  # TARGET date — UPDATE to the actual submission date before submitting
bibliography: paper.bib
---

<!--
JOSS paper checklist (target length: 750–1750 words):
  [ ] Summary                    — non-specialist overview
  [ ] Statement of need          — research problem, audience, related work
  [ ] State of the field         — comparison to existing tools, build-vs-contribute
  [ ] Software design            — architecture, trade-offs, why they matter
  [ ] Research impact statement  — realized/near-term impact, evidence
  [ ] AI usage disclosure        — required, even if "none"
  [ ] Acknowledgements
  [ ] References
Full guidance: https://joss.readthedocs.io/en/latest/paper.html
See SUBMISSION_PLAN.md for open questions to resolve per section.
-->

# Summary

<!-- DRAFT v1 — review; target 100–200 words -->

Transportation agencies and researchers describe roads, railways, and other networks
using *linear referencing*: rather than locating features by latitude and longitude,
they record positions as measured distances along a route — for example, a pavement
segment "from mile 4.2 to mile 5.8 on Route 9," or a collision "at milepost 12.4."
Data organized this way are called *events*, and they form the backbone of
transportation asset management and roadway safety analysis.

`linref` is a Python library that brings linear referencing to the widely used
`pandas` and `geopandas` data-analysis ecosystem. Through a lightweight accessor
(`.lr`) attached directly to DataFrames, it lets users define a linear referencing
system, generate and reshape point and segment events, convert between measures and
mapped geometry, and — most importantly — combine datasets that were collected against
different routes and segmentations into a single, consistent framework. These
operations are vectorized for performance and scriptable for reproducibility, so
analyses that once required manual GIS work can be expressed as a few lines of code.

# Statement of need

<!-- DRAFT v1 — review -->

Linearly referenced data is the dominant way transportation networks are described
and managed [@scarponcini2002generalized]. National programs institutionalize it: in
the United States, the Federal Highway Administration directs every state to maintain
an all-roads linearly referenced network — the All Roads Network Of Linear referenced
Data (ARNOLD) — as the basis for federal reporting [@fhwa2014arnold]. On top of these
route-and-measure networks, agencies layer many tabular *event* datasets: pavement
condition, traffic volumes, signs, crashes, and more.

A recurring and labor-intensive problem is that these datasets rarely share the same
segmentation. Crash records are points, pavement is reported in variable-length
segments, and traffic counts follow yet another breakdown. Reconciling them — the
operation practitioners call *conflation* — has traditionally required manual,
click-heavy GIS work that is slow, difficult to reproduce, and hard to audit.

`linref` targets this problem directly. It provides scriptable, vectorized operations
to dissolve, resegment, and overlay events, and to relate datasets to one another with
count- and length-weighted aggregation, so that combining many event layers onto a
common framework becomes a few lines of reproducible code. A concrete motivating
application is roadway safety network screening — locating segments where crash
frequency exceeds expectation — which the Highway Safety Manual formalizes around
sliding-window and peak-searching methods [@aashto2010hsm]. `linref` supplies the
segmentation and conflation building blocks these workflows depend on (illustrated by
the High-Injury Network example in the project documentation) rather than prescribing
a single fixed procedure. Its audience includes state and local transportation
agencies, safety and asset-management analysts, and transportation researchers who
need reproducible, auditable alternatives to bespoke GIS pipelines.

# State of the field

<!-- DRAFT v1 — review -->

`linref`'s data model reflects long-established linear-referencing theory rather than
an ad hoc design. Positions are expressed as measures along keyed reference lines, and
features are represented as point or linear events located by one or two measures —
broadly the same decomposition that Scarponcini formalizes as a linear element, a
linear referencing method, and a distance expression [@scarponcini2002generalized],
that appears as the linear subset of the broader multimodal object model in NCHRP
Report 460 [@adams2001nchrp460], and that agencies build in practice under ARNOLD
[@fhwa2014arnold]. `linref` adapts this model to the column-oriented structure of a
DataFrame.

Existing tools leave a gap. Proprietary GIS platforms (for example, Esri Roads and
Highways) offer capable linear-referencing functionality, but they are closed,
graphical, and difficult to script or reproduce. Within the open-source Python
ecosystem, `geopandas` [@jordahl2020geopandas] and `shapely` [@gillies2007shapely]
provide excellent coordinate geometry but have no native notion of routes, measures,
or events, and no measure-based conflation. To our knowledge, no general-purpose
open-source library fills this role, which is why `linref` builds on — rather than
reinvents — the `pandas`/`geopandas` stack, reusing its data structures and the
vectorized foundations of `numpy` [@harris2020numpy] and `scipy` [@virtanen2020scipy].

The analytical workflows `linref` supports are themselves well-studied: comparative
evaluation of network-screening methods is an established research thread
[@cheng2005experimental; @kwon2013evaluating], but one that has typically relied on
one-off, agency- or study-specific pipelines. `linref` aims to make such analyses
reproducible and general-purpose.

# Software design

<!-- DRAFT v1 — review -->

`linref` is built around a deliberate separation between the *schema* of linearly
referenced data and the data itself. An `LRS` (Linear Referencing System) object
records which DataFrame columns hold route keys, measures (a single location for point
events, or begin and end measures for linear events), and geometry — mirroring the way
`geopandas` attaches a coordinate reference system to a GeoDataFrame. This lets the
same operations apply uniformly to any conforming table and keeps `linref`
interoperable with the surrounding `pandas`/`geopandas` workflow instead of imposing a
bespoke object model. Users interact through a lightweight `.lr` accessor registered
on DataFrames, so linear-referencing methods sit alongside familiar pandas operations
rather than replacing them. This accessor pattern was chosen over subclassing
DataFrames to maximize interoperability and minimize surprise.

Behind the accessor, an internal `EventsData` engine represents events as numeric
intervals and performs the core computations — sorting, overlap, and aggregation —
with vectorized `numpy` [@harris2020numpy] and sparse `scipy` [@virtanen2020scipy]
routines, with optional chunking and caching for large networks. Operating in measure
space rather than repeatedly in coordinate space is what makes conflation both fast
and semantically correct. A concrete example illustrates why this matters: assigning
crashes to intersections is commonly done with a Euclidean buffer around each
intersection point, which can misassign a crash that is close in straight-line
distance but far along the road, or that lies on a nearby parallel route. Because
`linref` works in along-route measure space, crashes can instead be assigned by their
linearly referenced position relative to intersection points — a distinction that
generic point-buffer GIS operations cannot cleanly replicate. This measure-based
approach has been used in production network-screening work (see Research impact).

Three further design elements support real-world data. M-enabled geometries
(`LineStringM`) bind a measure to every vertex, bridging linear measures and mapped
geometry so the library can convert between the two and cut or merge geometry along
measures. A chaining construct handles routes whose geometry is non-contiguous — a
common feature of real inventories — by grouping contiguous runs within a key so that
operations do not bleed across physical gaps. Finally, an event-profile system
distributes a value along an event according to a chosen shape (for example, uniform
or triangular), directly supporting the weighted, sliding-window-style aggregation
used in safety screening.

# Research impact statement

<!--
DRAFT v1 — review. All four cited plans are public government publications, so naming
carries low confidentiality risk; author to confirm no client/contract restriction.
Version note: these engagements used linref v0.1.x; the cited capabilities persist in
v1.0, so the text intentionally describes features by category, not version-specific
method names. See RESEARCH_IMPACT.md for full evidence detail.
-->

`linref` has been applied in professional roadway-safety network-screening practice
across multiple published Safe Streets and Roads for All (SS4A) safety action plans.
The City of Newport, Rhode Island SS4A Safety Action Plan [@newport2025sap], prepared
with Toole Design, uses a modified sliding-window approach — distributing each crash's
severity score across neighboring roadway segments with a linearly decaying weight —
to build its High-Injury Network. The Eastgate Regional Council of Governments SS4A
Comprehensive Safety Action Plan [@eastgate2024sap] applies a related but mechanically
distinct step-and-window smoothing variant to the same problem. The Move Safe Michiana
Regional Safety Action Plan [@macog2025movesafe] follows the Highway Safety Manual
network-screening process directly, in technical memoranda co-authored by `linref`'s
developer. At the largest scale, the Texas Pedestrian Safety Action Plan
[@txdot2023psap], prepared for the Texas Department of Transportation by Jacobs,
applied the same sliding-window diffusion of crashes across neighboring segments to the
entire statewide roadway network of millions of crash records; this statewide analysis
was among the most demanding tests of the library to date and directly motivated the
performance, scalability, and parameterization improvements delivered in version 1.0.
Across these engagements — spanning local, regional, and statewide analyses and
multiple consultant teams — `linref` provided the underlying dissolve, resegmentation,
and event relation and aggregation operations, demonstrating sustained use of the
software in real analyses rather than aspirational or single-use application.

# AI usage disclosure

<!-- DRAFT v1 — review. Models: GitHub Copilot with Claude Opus 4.8 and Sonnet 5. -->

Generative AI tools were used in both the development of `linref` and the preparation
of this paper. The authors used GitHub Copilot with Anthropic's Claude models (Opus 4.8
and Sonnet 5). Within the software, Claude Opus assisted with algorithm benchmarking
and optimization, unit-test development, documentation refinement and copy-editing, and
drafting code for some minor features. In preparing this paper, the models assisted
with drafting and structuring the manuscript text and with researching and vetting
references.

All AI-generated content — code, tests, documentation, and manuscript text — was
reviewed, tested, edited, and validated by the authors, who made all core design and
architectural decisions and retain full responsibility for the accuracy, originality,
and integrity of both the software and this paper.

# Acknowledgements

<!-- DRAFT v1 — replace placeholders: contributors/colleagues to thank, and funding. -->

The author thanks colleagues at Toole Design for collaboration on the applied
safety-analysis work that motivated and exercised `linref`, and the open-source
maintainers of `pandas`, `geopandas`, `shapely`, `numpy`, and `scipy`, on which the
library is built. <!-- TODO: name specific individuals if desired. -->
This work received no specific grant or financial support. <!-- TODO: confirm or amend. -->

# References

