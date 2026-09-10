# The surgical assistant

BRIDGE's assistant layer turns the projector-and-camera rig into something a
scrub or circulating nurse can talk to with their hands full. It is built around
four capabilities that run at once.

| capability | module | what it does |
|---|---|---|
| scene awareness | `perception/scene_graph.py` | a live model of every object on the surface, so "project the scalpel" needs no AI round-trip |
| conversation | `assistant/conversation.py` | follow-ups: "and the other one", "point to it" |
| case autopilot | `surgical/case.py` | WHO-style checklist, phases, and an auditable case record |
| proactive monitor | `surgical/monitor.py` | BRIDGE speaks first when the numbers stop adding up |

and one deep workflow: **the surgical count** (`surgical/counts.py`).

## What BRIDGE will and will not do

BRIDGE **locates, counts, prompts and documents**. It does not diagnose, dose,
decide, or authorise anything.

* Dosing and diagnosis requests are refused in `healthcare/safety.py` **before**
  the intent grammar sees them and again before the AI is called. Two independent
  gates, on purpose.
* A reconciled count is reported as arithmetic — "43 of 43 items accounted for,
  confirm with your own count sheet" — never as permission to close.
* The camera's tally is advisory and is kept in a separate field (`observed`)
  from the human count (`counted_*`). A camera cannot see inside a wound.
* Wording is always what BRIDGE can perceive: "I cannot see three raytec
  sponges", not "you have a retained sponge".
* Skipping a critical checklist item is **recorded, not blocked**. Refusing to
  proceed mid-case would be unsafe in its own right.

## The count

The rule the engine exists to enforce:

```
final count  ==  initial count  +  everything added during the case
```

Phases follow the case: `initial` at count-in, `added` during the procedure,
`closing` at cavity closure, `final` at skin closure. Every item on the sheet is
a `CountLine` with a category — **sponge**, **sharp**, **instrument**,
**miscellaneous** — because sponges and sharps are the retained-item risk and get
the loudest rules.

```
"start the count"                 -> initial count, sheet from the instrument set
"count as per the sheet"          -> every line recorded at its expected quantity
"four artery clamps"              -> records 4 on that line
"adding two lap pads"             -> baseline for lap pads goes up by 2
"closing count" / "final count"   -> moves the phase
"what's the count"                -> reads back the arithmetic
"how many lap pads"               -> one line, plus what the camera can see
"what's missing"                  -> names them and outlines the empty slots on the tray
"close the case"                  -> final reconciliation + the JSON record
```

A short count at the final phase is spoken as **critical**: it cuts off whatever
BRIDGE was saying, is projected on the surface in red, and goes into the record.

### Naming discipline

Matching a spoken phrase to a count line is stricter than matching it to a scene
object, because a wrong match silently corrupts arithmetic. A partial match only
counts when one name *refines* the other and the head noun agrees:

* "clamp" → `artery clamp` ✓ (a shorter way of saying it)
* "curved artery clamp" → `artery clamp` ✓ (the same thing, said precisely)
* "bulldog clamp" → **no match** (shares a noun, disagrees on the qualifier)
* "suture scissors" → **not** `suture needle` (head noun differs)

Instrument sets live in `surgical/sets.py` and can be extended with JSON in
`sets/` matching `InstrumentSet`. Built-ins: minor/basic, laparotomy, suturing
tray, central line. Your own count sheet is always authoritative.

## The case

```
briefing -> sign_in -> timeout -> count_in -> procedure
         -> count_closing -> count_final -> sign_out -> debrief -> closed
```

Each checklist phase is a list of items read one at a time; "confirmed" advances,
"skip" records a skip. The content follows the shape of the WHO Surgical Safety
Checklist and is data — load your hospital's own with
`case.load_checklist(path)`.

At `close the case`, BRIDGE writes `case_records/<case-id>.json`: every
confirmation with its timestamp, every skip (critical skips listed separately),
all count reconciliations, every alert raised, and a disclaimer naming the
operating team as responsible.

## Proactive alerts

Rules are evaluated about once a second on the scene graph, never per frame.

| rule | level | condition |
|---|---|---|
| `closing_short:<item>` | critical | a sponge or sharp is not visible on the tray at closing/final |
| `tray_short:<item>` | caution | before the field is open, the tray disagrees with the initial count |
| `sharp_loose:<id>` | caution | a sharp outside both the tray and the neutral zone for 20 s |
| `left_out:<id>` | caution | something motionless off the tray for 45 s during the procedure |
| `unlisted:<id>` | caution | a new labelled object that is not on the count sheet |
| `view_blocked` | caution | everything vanished at once — the camera is covered |

Three things stop this becoming an alarm nobody hears: every rule has a key and a
cooldown (announced once, then not again until it clears); `info` is never spoken
unasked; and conditions that depend on absence must persist for a grace period,
because a hand over the tray hides instruments.

"Quiet" mutes proactive speech without affecting answers; "alerts on" restores it.

## Zones and the tray layout

`TrayLayout` snapshots where every counted item sat at count-in. That is what
lets "what's missing" outline the **empty slot** rather than just naming the item.
`Zone` is a named polygon in camera space — `tray`, `neutral` (the hands-free
sharps transfer zone), `back_table`. The tray zone is inferred automatically from
the bounding box of everything on the surface at count-in.

Both are camera-space, so both are cleared when calibration is invalidated.

## Latency: why it feels instant

A spoken sentence takes one of three paths, and only the third touches the
network:

1. **Local grammar** (`assistant/intents.py`) — regex match, microseconds. Counts,
   case control, status, mute, clear.
2. **Scene graph** (`perception/scene_graph.py`) — "project the scalpel" resolves
   out of local memory and the reticle is up before the next camera frame.
3. **Gemini** — genuinely open-ended requests only. Whatever the model finds is
   folded back into the scene graph, so the *second* time the same thing is asked
   it takes path 2.

The model is also asked, in the background every `assistant.ai_label_interval_s`
seconds during a case, to name what it sees. Positions always stay local: Gemini
answers *what*, OpenCV answers *where*.

## Voice

Always-on with **barge-in**: speaking over BRIDGE cuts it off mid-sentence.
Because the microphone also hears BRIDGE's own voice through the speakers, an
utterance captured while speaking is compared against what is being said and
discarded if it is mostly the same words (`voice.assistant.is_echo`).

Replies and alerts share one voice channel through the `Announcer`, which is a
priority queue: `critical` preempts and clears, `reply` jumps ahead (the user is
waiting), `caution` waits its turn and is dropped if it goes stale, duplicates
within 20 s are dropped outright.

## Settings

```yaml
assistant:
  enabled: true
  proactive: true            # BRIDGE may speak first
  scan_hz: 5.0               # scene-graph updates per second (local CV)
  ai_label_interval_s: 12.0  # how often the model is asked to name what it sees
  show_count_board: true     # project the live count board onto the surface
  records_dir: case_records
surgical:
  default_set: minor
  sharp_grace_s: 20.0
  field_item_grace_s: 45.0
  absence_grace_s: 6.0       # how long unseen before it counts as missing
  alert_cooldown_s: 45.0
voice:
  barge_in: true
```

## Trying it without an operating theatre

```bash
bridge --simulation
```

Then, in the command box or by voice: *start a case for a minor set* → *start the
count* → *count as per the sheet* → *adding two lap pads* → *final count* → *close
the case*. `bridge --headless-selftest` runs the same sequence with no window.
