What Is Linear Referencing?
===========================

Linear referencing is a system for storing, locating, and interpreting data
based on relative position along a principal linear feature without needing
explicit geographic coordinates such as latitude and longitude. That
principal linear feature is called a **reference line** -- it is the static
backbone against which all other data is positioned. A reference line may
also be referred to as a route, refering to a real-world linear asset,
such as a roadway, rail corridor, pipeline, trail, and more.

Linear referencing allows users to locate point assets along a reference
line, such as signs, intersections, or crashes, as well as linear assets
like guardrails, fences, or bridge structures. It can also describe
supplemental attributes about the reference line itself that do not
inherently carry a spatial component, such as the diameter of a pipe, the
number of lanes along a roadway segment, or a posted speed limit.

This approach is useful when the primary question is not just "where is it
on a map?" but "where is it along the route?" A crash at milepost 8.2,
a pavement section from milepost 3.0 to 4.5, and a speed zone from station
12+00 to 18+50 are all examples of linearly referenced data.

Linear referencing can also be used to relate multiple datasets that share 
the same linear referencing system, aggregating data between collections
of point and linear events. It can also be used to perform spatial operations,
engineering new geometries and attributes through dissolving, resegmenting,
and modifying. Linear referencing can also support advanced analytical 
workflows such as sliding window analysis, data integration, and asset 
management.

In ``linref``, linearly referenced records are modeled as **events**. An event can
represent either a single measured location along a reference line or an
interval bounded by a begin and end measure.

Core Concepts
-------------

Reference lines and key groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every valid event is tied to a single, specified reference line. Datasets
typically contain multiple reference lines, each of which is uniquely
identified by one or more fields in the dataset. In many systems this is a
single route identifier (e.g., ``'US-101'``), but in practice the reference
can be defined by several fields together (e.g., route, direction,
county, year) that combine to uniquely identify the linear
context.

``linref`` represents these identifiers as one or more **key columns**
(``key_col``). The key columns define the group within which measures are
meaningful and comparable. Two events may both have measure 5.0, but if they
belong to different key groups they do not refer to the same place along the
same reference line.

A common example is a column called ``route`` whose values are route names
like ``'US-101'`` or ``'SR-1'``. For more complex datasets, an additional 
column may be needed to uniquely identify the reference line. Examples include
``direction`` for dual carriageway roads, or ``county`` for cases where the
same route name may be used in multiple counties.

When the same keyed reference line appears in multiple disconnected
geographic pieces, ``linref`` can also use a **chain column**
(``chain_col``) to distinguish contiguous geometry groups within that
reference line.

Measures
~~~~~~~~

A measure expresses position along a reference line. Measures are typically
expressed in mileposts, feet, stationing values, or any other monotonic
linear distance system defined along the reference line.

``linref`` supports two event forms:

* **Point events** use a single location measure stored in ``loc_col``.
* **Linear events** use a begin measure and an end measure stored in
  ``beg_col`` and ``end_col``.

Note that ``linref`` can be used with time-based measures as well, though this
is not the primary use case and has not been as thoroughly tested.


Events
~~~~~~

An event is any record whose location is defined relative to a reference
line.

**Point events** describe a single measured location, such as a crash,
intersection, sign, detector, valve, or crossing.

**Linear events** describe an interval along the reference line, such as
pavement condition, guardrail, lane count, speed zone, pipe diameter, or
rail class.

Attributes and geometry
~~~~~~~~~~~~~~~~~~~~~~~

Most event tables carry descriptive attributes that explain what the event
represents. Those attributes can take any form and may reflect design 
characteristics, operational conditions, or other metadata relevant to the
linear reference. Examples include crash severity, pavement condition, traffic
data, installation date, or asset condition.

Geometry is optional. Because a linearly referenced record's position is
fully defined by its key columns and measures, an event table can be purely
tabular. In many workflows, event attributes and event geometry are stored in
separate datasets and related through the same linear referencing system. For
example, a roadway system may be represented by a single spatial layer with 
geometry for all reference lines, while multiple event tables carry attributes
for supplemental information about those reference lines.

For spatial workflows, ``linref`` can also manage:

* ``geom_col`` for standard point or line geometry
* ``geom_m_col`` for M-enabled line geometry used in measure-aware spatial
  operations such as projection, interpolation, and cutting

How ``linref`` Uses ``LRS``
---------------------------

In general practice, "linear referencing system" can refer to an entire
organizational framework of reference lines, calibration, measures, and
related business rules.

In ``linref``, an ``LRS`` object is narrower and more concrete. It defines
the **schema** that tells a DataFrame which columns contain:

* key fields that identify the reference line for each event
* point or linear measure fields
* optional geometry fields
* the closure convention for linear intervals

That schema-driven definition is what enables the ``.lr`` accessor to treat
an ordinary pandas or geopandas table as linearly referenced data.

Parameters of an ``LRS``
------------------------

The table below summarizes how the concepts described above map to ``LRS``
parameters in ``linref``.

``key_col``
   One or more columns that identify the reference line for each event. Often
   a route name, but can include direction, county, or other scoping fields.

``chain_col``
   An optional column that can be used to distinguish between disconnected sections
   within the same keyed reference line, essentially acting as an additional key 
   column.

``loc_col``
   The location measure for point events. Can also be used to define an additional
   reference measure for linear events.

``beg_col`` / ``end_col``
   The begin and end measures for linear events.

``geom_col``
   Geometry associated with point or linear events. Must be singlepart.

``geom_m_col``
   M-enabled geometry for workflows that need geometry tied directly
   to reference line measures. Currently uses a ``linref``-specific 
   M-enabled LineString geometry type until native support is available in
   shapely. Can be generated and manipulated using methods of the ``.lr`` accessor.

``closed``
   The interval closure convention for linear events. This determines how
   event bounds are interpreted during intersections and related operations.

What Makes Data Usable for Linear Referencing
---------------------------------------------

At a minimum, a workable linear referencing setup needs:

* a consistent set of key columns that identify the correct reference line
* valid measures that use the same measurement system within each key group
* monotonic linear events, where begin measures do not exceed end measures
* a defined endpoint convention for linear events when adjacent intervals
  share boundaries

For geometry-based workflows, you will also need:

* a single reference linear dataset with valid, singplepart geometry
* contiguous reference line geometry within each group, or a chain column when a keyed
  reference line is split into disconnected parts
* M-enabled geometry when you need to project data, interpolate locations,
  cut segments, or recover measures from geometry; note that this can be generated
  as needed from a reference line dataset with valid geometry and measures

Common Applications
-------------------

Linear referencing is commonly used to:

* manage roadway assets and information such as cross-sectional design, 
  operational characteristics, located assets, and crash records
* track railway features such as crossings, signals, operational speeds, and track
  classes
* manage inspection records along long linear assets such as pipelines and power lines
* relate, aggregate, or integrate multiple event tables that share the same 
  reference lines and measures
* segment, dissolve, overlay, and summarize data along reference lines for
  analysis
* perform network screening analyses such as sliding window analysis, crash 
  profile analysis, and more
* identify and analyze intersection points and clusters along reference lines

How This Maps to ``linref``
---------------------------

The ``linref`` library is designed to support linear referencing workflows in
all sorts of applications and with various types and formats of data. It also 
contains several helpful features for generating, validating, and managing
linearly referenced data, making it a useful tool for both new and existing datasets.

Once an LRS is set, the ``.lr`` accessor can validate event structure,
manage point and linear events, generate or use M-enabled geometry, relate
multiple datasets, and build analysis-ready linear event frameworks.

For a hands-on introduction, continue with the examples section, especially
the setup and LRS configuration walkthrough.
