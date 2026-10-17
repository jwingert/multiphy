# The viewer

One window opens any completed project run (see [project_folder.md](project_folder.md)):

```bash
python -m multiphy.viewer --run DIR
```

```
usage: python -m multiphy.viewer [-h] --run DIR
```

`DIR` is a run folder: `PROJECT/runs/<run>` for an across-days run,
`PROJECT/days/<day>/runs/<run>` for a same-day run. The run's `selection.json` names its
mode; the window opens in that mode. The viewer runs no matcher: runs come from
`launch run` or the launcher window (`launch gui`), whose `Curate…` and `Inspect…` open a run
here.

## Package map

`multiphy/viewer/`:

| module | role | Qt |
|---|---|---|
| `app.py` | `App`, the window; `open_run(run_dir)`; `main()` behind `python -m` | yes |
| `state.py` | `PairState`: the shown `RunData`, day pair, the pair on screen and selected units; `Actions`, the only way a table reaches the window | yes |
| `data.py` | `UnitData`, `Words`, `RunData`, `Mode`, and the Qt-free pair preparation | no |
| `display.py` | score labels, header texts, gate outcomes, evidence fields; the theme's colour tokens (`PALETTES`, `colors()`, `unit_colors`, `uses_unit_palette`) | no |
| `unit_palettes.py` | the Unit colours palettes (`IDS`, `LABELS`), the active palette and the Day ramp | no |
| `drift_frame.py` | `drift_frame`, `unit_depth`, `electrode_depths`, `registered_positions`, `unit_distance`: a run's registration as drawn depths and unit distances | no |
| `qt.py` | the one Qt binding (PySide6), pinned before matplotlib's Qt backend | yes |
| `theme.py` | the app style, palette, style sheet, bundled fonts and plot colours; the saved theme and unit palette and their live switches | yes |
| `menus.py` | the `Settings` (`Theme`, `Unit colours`) and `Help` menus, the guide window and the `Keyboard shortcuts` window | yes |
| `keys.py` | the shortcut actions, their saved keys and the key notifier | yes |
| `fonts/` | `Figtree.ttf`, `JetBrainsMono.ttf` and their SIL Open Font License 1.1 texts | no |
| `logo/` | `logo.svg` and `icon_<px>.svg` (written by `make_logo.py`); `icon()` (the window icon), `pixmap()` (the About picture) | yes |
| `registry.py` | `PANELS` and `@register_panel` | no |
| `modes/` | `across_days.py`, `same_day.py`; `MODES` maps a mode id to its `Mode` | no |
| `panels/plots/` | the matplotlib plot panels, each registered by id | yes |
| `panels/tables/` | browse (with the day table), rivals, evidence, chains and tracked-unit tables | yes |

`data.py`, `display.py`, `unit_palettes.py` and `modes/` import no Qt, and `viewer/` imports
no lab module (a plugin's own modules included).
`drift_frame.py` imports no Qt and no matplotlib either.

## Theme

The viewer opens in the saved theme: graphite, unless `Settings` ▸ `Theme` ▸ `Dark` or `Light`
was chosen. The choice is saved per user in `~/.config/multiphy/multiphy.ini` (key `theme`; a
missing or unknown value opens graphite). A switch restyles every open window, redraws every plot
(zoom and pan reset) and keeps the selected pair and unsaved notes. A choice that cannot be saved
still switches, and the status bar shows `theme not saved · <file>`; a switch whose tick and
chevron images cannot be written changes nothing and shows `theme not switched · <reason>`.
Panels read `display.colors()` when they draw, and coloured table cells carry a token through
`theme.tone` or `theme.fill`, never a hex colour. `Help` opens this guide, the launcher guide and
`Keyboard shortcuts`: one key per action, recorded in place; a key another action holds is refused
and named, a cleared key leaves its action unbound, `Reset defaults` restores them; keys are saved
as `shortcuts/<action id>` in `multiphy.ini`, and the viewer rebinds at once.
`Help` ▸ `About multiphy` shows the logo, the version (`multiphy.__version__`, `0.0.0`
until launch) and the logo's source unit. The logo is one unit's peak-channel waveform
(`PRN067a-338-1`) tracked across three days; it is the icon of every viewer and launcher window,
with heavier strokes at 64 px and below. `viewer/logo/make_logo.py` writes its SVGs.

`Settings` ▸ `Unit colours` picks the colours of the selected units in the plots. The choice is
saved in the same file under key `unit_palette`; a missing or unknown value opens
`Graphite pastel`, and ids match exactly, case included. Each fixed palette's item shows its first
eight colours; `Day ramp` shows eight steps across the whole ramp.

| item | id |
|---|---|
| `Graphite pastel` (default) | `pastel` |
| `Tol light +` | `tol` |
| `Max-distance` | `maxdist` |
| `Okabe–Ito + tints` | `okabe` |
| `Tableau 20` | `tab20` |
| `Day ramp` | `ramp` |

With a fixed palette on `Graphite` or `Dark`, units take its colours in selection order, so a pair
takes the first two and units 1–2 keep their colours as units are added; past the 15th the
colours repeat. `Day ramp` instead colours each unit by its site's rank among all of the run's
sites, ordered by `session_date` then `siteid` (sites without a date last; `siteid` alone when the
run has no dates); units of one site share a colour, and a run with one site takes the ramp's
first colour. The palette applies to `Graphite` and `Dark`; `Light` keeps its own ten-colour ramp
in selection order, repeating past the 10th, and the menu then reads `Unit colours · dark themes`.
A palette that cannot be saved still applies, and the status bar shows
`unit colours not saved · <file>`.

The UI font is Figtree; tables and plain-text boxes (the launcher log, notes) are JetBrains Mono.
Both ship in `viewer/fonts/` under the SIL Open Font License 1.1 (`Figtree-OFL.txt`,
`JetBrainsMono-OFL.txt`). A font file that cannot be loaded prints
`<file> not loaded · system font` to the terminal, and the system font is used.

## Data protocol

A mode loads a run into one frozen `RunData`. Panels read its methods, never dicts:

| method | returns |
|---|---|
| `unit(cellid)` | the unit's `UnitData` |
| `units()` | the unit table |
| `sites()` | `{cellid: siteid}` of every unit |
| `pairs()` | the pair table (`cellid_a`, `cellid_b`, `siteid_a`, `siteid_b`, `match_score`, `decision`, `selected`, `score_kind`, plus evidence) |
| `day_pairs()` | the day pairs, pairs of two different sites only, each with its streams when a mixture was fitted |
| `geometry(site)`, `geometry_source(site)` | the site's channel positions (or `None`) and where they came from |
| `freq_axis(site)` | the site's tone frequencies, or `None` |
| `min_repeats()` | the fewest trials a fingerprint condition needs |
| `day_name(site)` | the site's day label |
| `available()` | the optional parts this run has |

`UnitData` fields: `cellid`, `siteid`, `waveform`, `ptp`, `patch`, `peak_chan`,
`fingerprints`, `trial_counts`, `fingerprint_bin_seconds`, `isi`, `acg`, `tuning`. A field
the mode does not have is `None` (or `{}`), and a panel skips it.

`available()` holds a subset of `isi`, `acg`, `tuning`; any other part is refused with
`ValueError`. Both modes have `isi` and `acg`, and `tuning` when a site has a frequency axis.
The probe and footprint panels draw measured coordinates only.

## Modes

| id | runs | curates | layout file |
|---|---|---|---|
| `across_days` | two or more dated days of one project | yes: `curation/runs/<run>/history.csv`, `history.lock`, `launch export` | `viewer/across_days_layout.json` |
| `same_day` | the two temporal halves of one day | no: inspect-only, no curation history | `viewer/same_day_layout.json` |

The layout files sit under the lab root (the folder that holds every project). Closing the
window saves its layout there; **View → Reset layout** restores the default.
The default layout has Browse in front of Tracked units; a saved layout keeps the tab it saved.

A `Mode` names its `Words` (the nouns its tables use), its plot panel ids and three functions:
`load(selection, dataset, result, extras) -> RunData`, `header(data, cellids, row)` and
`merge_refusal(data, members)`.

## The day table

`panels/tables/days.py` holds `DayTablePanel`, docked under Browse's run list. It lists every unit of
each day beside its links.

- **Columns:** one per site of `site_order()`, headed by `day_name(site)` on two lines (date over
  site), with `unit`, `µm` (distance), `sc` (score) and `s` (status). A column lists the units of
  every track on that site; each column scrolls on its own. Opening a run leaves no unit picked.
- **Distance and score cells:** `µm` is the distance to the column's reference (see **String**), in
  whole µm; `sc` is the score to it. Both read `--` with no unit picked, for the reference itself,
  or with no value.
- **Header hovers:** `µm` reads `distance to <reference> · µm` and `sc` reads `score to <reference>`;
  with no unit picked, `distance to the picked unit · µm` and `score to the picked unit`. `s` names
  its letters.
- **Widths:** `unit` 64 px, `µm` 40 px, `sc` 40 px, `s` 28 px (`WIDTHS`), under compact header
  padding. A day column is never narrower than those plus frame and scrollbar (184 px under the
  theme); `spread` shares any extra width among all four columns in proportion to `WIDTHS`, again
  whenever the table's viewport resizes (a vertical scrollbar appearing narrows it). The days scroll
  sideways when they don't fit. Opening a run and **View → Reset layout** size Browse to show the
  first `min(days, FIT_DAYS)` day columns (5) at that width, growing or shrinking it. The window never resizes: Browse stops where the other docks,
  or Tracked units tabbed with it (544 px minimum), reach their minimum widths. Dragging Browse
  narrower scrolls the days; wider stretches them.
- **Status:** one letter, its word on hover: `a` (accepted) for a unit of an accepted track; `m`
  (matched) for a track of two or more; else `u` (unmatched). With no pick every score is `--`.
- **Score:** the pair row of the two units (either order): its finite score, else its finite
  similarity, else missing. Non-numeric or non-finite cells count as missing. `--` for a missing
  score and for the picked unit.
- **Held order:** each day keeps a held order. Opening a run holds each day in status order
  (`u`, `m`, `a`, then cellid). A column shows its held order with no pick, for the
  picked unit's own day, and for a day whose header was clicked since the last pick; a unit missing
  from it follows by cellid, so every unit shows once. Accept, Reject, Merge and Unmatch unit never
  change it, so the own day's rows stay in place.
- **Pick order:** a click picks a unit. Every other day sorts to its reference: the picked unit's
  current track first; then distance to the reference, ascending; then score to it, descending; then
  cellid. A missing distance or score sorts last. Distance is 2D between (x, depth) as the drift view
  draws them: other days shifted by the run's saved registration, a day it left unlinked or a group
  with no usable offsets at its measured depth. No distance across recording groups, without `x_um`
  or `depth`, or when the group's registration cannot be read.
- **Headers:** a click on `unit`, `µm`, `sc` or `s` sorts that column into its held order: `unit`
  by cellid; `µm` by distance then score to the reference; `sc` by score then distance to it (`µm`
  and `sc` by cellid with no unit picked); `s` by status, then cellid. The same header again, while
  the column shows that order, reverses it. The arrow shows on a column showing a held order a header
  set, pointing down for `sc` only; another day's header sort lasts until the next pick.
- **String:** a click picks a unit and empties the string. The string holds one strung unit per
  other day (never the picked unit's own day). Each day has a reference, the unit its `µm` and `sc`
  cells and its header sorts measure against: the picked unit for its own day; for another day,
  walking the days from its neighbour toward the picked unit's day (and stopping before it), the
  first strung day's unit, else the picked unit. The own day keeps its held order (see **Held
  order**); every other day sorts to its reference unless its header was clicked since the last
  pick. With days A, B and C and a unit of A picked, C sorts to B's strung unit when B is strung,
  else to the picked unit.
  Extras sit beside the string: same-day clicks and, after `F`, a track's further units on one day.
  A Ctrl-click or a drift-dot click on another day's unit strings it there, or drops it when it is
  already that day's strung unit; that day becomes active. The plots show the picked unit, the strung
  units in day order, then the extras; the evidence shows the active pair: the active day's reference
  and its strung unit, or the picked unit and the extra last clicked. A curation action (a button,
  undo or redo) keeps the string selected with its active pair.
- **Active day:** the day the string acts on; its header shows in the `accent` colour while a unit is
  picked. A pick makes the picked unit's own day active, as does a click on the picked unit. While the
  same-day column is active (after adding an extra there), its header shows in the accent instead.
- **Same-day column:** pinned left of the days, outside their scroll, so it stays in view. Empty
  (headed `same day`) until a pick; each pick fills it with the picked unit's other units of its own
  day, by the pick order above with the picked unit as reference, headed `same day · <day name>`
  (date over site). Its `µm` and `sc` read against the picked unit; the `unit` hover lists only the
  proposed-link lines. A click there, with or without Ctrl, adds the unit as an extra beside the
  string (the same-day column becomes active) or removes it; the pick, the string and the column
  order stay.
  Its header sorts it as a day column's header does, to the picked unit (the same header again
  reverses it), until the next pick refills it. Its width is fixed at a day column's narrowest.
- **Keys:** one key per action (`viewer/keys.py`), saved per user as `shortcuts/<action id>` in
  `multiphy.ini` and changed in `Help` ▸ `Keyboard shortcuts`. The defaults, in action order:
  - `W` / `S`: the unit above / below in the active day (clamped). In another day it becomes that
    day's strung unit (its top unit when none is strung); in the picked unit's own day it becomes the
    pick (no pair opens); with the same-day column active it replaces the extra last clicked.
  - `A` / `D`: the day to the left / right becomes the active day (clamped); a day other than the
    picked unit's own gets its top unit strung when none is.
  - `F` strings the picked unit's current track; `Esc` clears the string back to the picked unit; `X`
    drops the active day's strung unit, so the days beyond sort to the day before.
  - `E` / `R` / `U` save accept / reject / unsure for the pair on screen, then step down as `S` does.
  - `Q` merges (`Merge`); `Shift+E` / `Shift+R` accept / reject the picked unit's track (`Accept`,
    `Reject`); `Shift+X` takes the active unit out of its track (`Unmatch unit`).
  - `Ctrl+Z` / `Ctrl+Shift+Z` undo / redo the last curation action.

  A saved key that is not one key (or not text) falls back to its default; an empty one leaves its
  action unbound; a key an earlier action already holds leaves the later action unbound. No saved
  value stops the window. The verdict, track, merge, unmatch, undo and redo keys work only while the
  run is writable; the movement keys (`W` `S` `A` `D` `F` `Esc` `X`) always work, inspect-only
  included. Letter keys don't fire while you type in the notes box; `Esc` does. A saved key rebinds
  every window of that viewer at once; another viewer open at the time rebinds on its next start, and
  the last save wins.
- **`⇄`:** a unit the backend saved a `proposed_link` pair for (two units of one native group that
  were not kept together in one track; the partner may be on the same day or another day). Hover: one line `proposed link · <partner> · <score>` per
  partner.
- **Wash:** `accent_tint` for the selected units; else `proposal_tint` for the units the backend grouped
  with the picked unit (its track in the result, and its proposed-link partners). The wash shows the
  backend's grouping, so curation never clears it.
- **Curation text:** bold for the picked unit's current track, toned `accept` when that track is
  accepted; else toned `reject` for a unit that shares a rejected track with the picked unit.
- **Buttons:** `Accept · Reject · Merge · Unmatch unit`. Accept and Reject act on the picked unit's
  track; Merge joins the whole tracks of the selected units (2 or more tracks); Unmatch unit takes the
  active unit (the active day's strung unit, the extra last clicked, else the picked unit) out of its
  track. With no pick: `pick a unit`. A refused action writes no history row and shows its message only.
- **Inspect-only runs** (`same_day`) show no buttons; a read-only run greys them out.
- **Tracked units follows both ways:** a pick shows its track's row and Members table; a Tracked units
  row keeps a pick that is already in that track, else picks the track's earliest member by day. A pick
  whose track the `needing review only` or `hide decided` filter hides clears Tracked units' row and
  Members table; the filters stay. A day-table action whose track a filter hides keeps the pick and the
  selection, and clears Tracked units' row and Members table, as a pick does.
- **Tracked units flag:** a track holding a unit with proposed links shows `proposed link · <units>`,
  and the flag never clears. A row needs review only while its status is `proposed` or `unmatched`:
  accepting or rejecting a track takes it out of the queue for every flag (`N units on one day`,
  `weak internal link`, `proposed link`); a unit a rejected track left unmatched stays out too;
  reopening the track or merging queues it again.


## Adding a plot panel

1. In a new module under `viewer/panels/plots/`, subclass `PlotPanel` (`panels/plots/base.py`),
   set `panel_id`, `panel_title` and, if needed, `nrows`, `ncols`, and implement `draw(r)` from
   `self.state.data` (a `RunData`) and the candidate row `r`. Take every colour from
   `display.colors()` inside `draw`; a colour literal does not follow the theme.
2. Decorate the class with `@register_panel`. An id already registered raises `ValueError`.
3. Import the module from `panels/plots/__init__.py`, so it registers when the package loads.

Every registered panel gets a dock, the mode's `plots` first; one the default layout does not
place is tabbed with the fingerprint panel. The viewer
loads no plugin panels: only the modules `panels/plots/__init__.py` imports register.

The interval histogram and the autocorrelogram are two docks tabbed together left of the fingerprint, the interval histogram in front. A layout file saved before they split lists a `spiking` dock, so the window opens the default layout once; closing it saves the new layout.

## The mixture panel

The panel has one column per stream the run saved a weight for (event response, physiology,
waveform), then the combined one. Per column it draws the observed scores of
the shown day pair, the two fixed references (same unit: a unit against its own other half
within a session; different units: simultaneously recorded neighbours inside the gate) and
the fitted mixing fraction. Only the mixing fraction is fitted. A day pair without a fitted
mixture, or a selection without a candidate row, shows the tag `mixture: not fitted`.

The run's assignment, calibration status and measured inputs live in the provenance of
`result.json` and are not drawn.

## The drift panel

The `drift` panel (`Drift across days`) draws every day of one recording group side by side,
each as a probe column with its units, in one depth frame: the anchor day's. It reads the
registration the run saved (`result.provenance['group_runs'][group]['provenance']['registration']`,
see [contract.md](contract.md) section 4); the viewer computes and writes no offsets.
`drift_frame.py` turns a registration into drawn depths and imports no Qt.

### Which group and which days

- **Group drawn**: the recording group of the first selected unit found in the unit table. With
  nothing selected (or no selected unit in the table) the panel still draws: the group that sorts
  first by name. A table with no units shows `no units`.
- **Days**: the group's sessions (`session_id`, else `siteid`), ordered by `session_date`, then
  by session. Group names match exactly (no stripping, no case folding).
- **Linked groups**: for a day-map registration (no `pair_drifts`), the days the registration
  links by measured offsets (`diagnostics[group]['groups']`), kept only for days that have an
  offset and are drawn. For a pair registration (`pair_drifts`, deepunitmatch), the connected
  components of the drawn pairs with `n_matches > 0`; `diagnostics[group]['groups']` is not
  read. Either way a drawn day in no group is a group of its own.
- **Anchor**: the first day, when its linked group has two or more days. Otherwise the first day
  of the largest linked group with two or more days; a tie goes to the group whose first day
  comes earliest. No such group: the first day. The anchor never holds a per-shank offset (that
  registration is refused).

### The day map, re-referenced to the anchor

A day `d` in the anchor's linked group is drawn at

    drawn depth = (scale_d × depth + offset_d − anchor offset) / anchor scale

with `scale_d` its `depth_scale` (default `1.0`) and `offset_d` its `offsets_um` entry, so the
anchor's electrodes, and its units without a per-unit shift, stay at their measured depths.
Electrodes follow the same map from their `y`.

- **Per-shank days** (`shank_offsets_um`): each unit takes its own shank's offset. The shank is
  the unit's `unit_shanks` entry; a unit with no saved shank takes the shank its `x_um` falls on
  (shank `i` holds `i × shank_width_um < x < (i + 1) × shank_width_um`, anything else is shank
  `0`, UnitMatch's rule); a unit with neither a saved shank nor a finite `x_um` gets no dot.
  Each electrode takes the shank of its own `x`.
- **Per-unit shifts** (`unit_offsets_um`): a unit listed there is drawn at `depth` plus its own
  shift (the shift the backend applied at that unit's location), then re-referenced to the
  anchor; electrodes follow the day map.
- **Units the backend never moved** (no `unit_offsets_um` entry, for example units outside its
  eligible single units) follow their day's map.
- **Day pairs** (`pair_drifts`): the anchor comes from the pair components above (so an
  isolated first day `D1` with a matched `D2`–`D3` pair anchors on `D2`). Only a pair with
  matches between the anchor and a day shifts that day, by the drift saved for that pair. A day
  linked to the anchor only through a third day is drawn raw with `no matches to <anchor>`;
  drift never chains through a third day. When no drawn pair has matches (every `n_matches` is
  `0`), the first day is the anchor and every other day is drawn raw with
  `no matches to <anchor>`; the panel never shows `no offsets` for a pair registration.
- **Unlinked days**: a day outside the anchor's linked group is drawn raw (its measured depths)
  and tagged `no matches to <anchor>`.
- **Unreadable registrations** raise `ValueError: registration of <group>: <problem>` (a non-number
  offset, shank lists of different lengths, a shank list without a `shank_width_um`, a pair with
  no `n_matches`, a pair that ends at the anchor, …); the panel shows that text in place of the
  plot.

### The drawing

| element | drawn as |
|---|---|
| columns | one per day in day order, spaced by the probe's width plus `max(0.5 × width, 40)` µm; each tick names the day |
| electrodes | squares at the drawn electrode depths |
| an `unlinked` column | hollow electrodes; electrodes, dots and tick in the muted tone; the tick reads `<day>` over `unlinked` |
| filled dot | a unit in one of the run's tracks or in a selected pair |
| hollow dot | any other unit with a finite depth |
| selection | a ring in the unit's colour around each selected unit of the group; two or more are joined by one line in day order |
| y label | `depth in <anchor> frame (µm)`, or `depth (µm)` when the group has no offsets |
| title | the group (same-day mode), plus `no offsets` when the registration gives no offsets |

The panel has its own dock, a pane beside the footprint/probe tabs. A layout file saved before
the drift panel existed (before S11) has no `drift` dock, so the window opens the default layout
once; closing it saves the new layout.

### Hover

| over | text |
|---|---|
| a linked column | `D2 · +12.5 µm · matches` (day · shift · method) |
| a column whose shift varies (per shank or a depth scale) | `D2 · -40.0 to +10.0 µm · native` (the range over its electrodes, else its dots) |
| an `unlinked` column | `D3 · unlinked · no matches to D1` |
| a column with no offsets | `D2 · no offsets · backend saves no offsets` |
| a unit (within 6 pixels of its dot) | `b:1 · D2 · +12.5 µm · matches`, its own shift; in an unlinked or no-offsets column, `<cellid> · ` and the column's text |

### Click

| click | does |
|---|---|
| a unit dot (within 6 pixels), left button | once the double-click interval passes, like a Ctrl-click in the day table: sets or drops that day's strung unit; with no unit picked it becomes the pick; nothing when that run no longer draws the unit by then |
| pan or zoom on, the right button, a column or empty space | nothing |
| a double-click | resets the view; no unit is added or removed |

A redraw of the same run that draws the same full-size limits (a selection change within one
recording group) keeps your zoom and pan; Reset view, a double-click, a theme change, another run or
another group draw at full size.

### `no offsets` reasons

Every day is drawn raw (measured depths, no column muted) and the title ends in `no offsets`;
the hover names the reason.

| reason | when |
|---|---|
| `run predates offsets` | the group's provenance has no `registration`, and its backend's spec declares `Provides(..., day_offsets=True)`: the run was made before the backend saved offsets |
| `backend saves no offsets` | no `registration`, and the backend does not declare `day_offsets` (or is not registered) |
| `registration off` | the registration's `method` is `none` (run with registration switched off) |
| `group not registered` | a day-map registration (no `pair_drifts`) links none of the drawn days (no day with an offset in a `diagnostics[group]['groups']` list) and carries no note; a pair registration never gives it |
| `dant kept initial motion` | DANT applied the motion it started from (it fitted no round, or its stop-early check reverted the first fitted round), so the run saved no offsets for the group |
