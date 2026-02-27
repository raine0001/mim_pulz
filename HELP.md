# MIM Corpus Help

## Quick Start

1. Start server from project root:

```powershell
python app/corpus/query_api.py
```

2. Open main UI:

`http://127.0.0.1:50001/corpus/mim`

3. Open full help page in browser:

`http://127.0.0.1:50001/corpus/help`

## Main UI Areas

- `View Mode`: switch between `Internal Only`, `Routed (Recommended)`, `ORACC Only`.
- `Structural Filters`: filter visible evidence by fragmentation, template, policy, and origin.
- `Result Pool`: choose how many top-ranked cards are loaded per section.
- `Global Search`: open corpus browser search over all indexed pages.
- `Routing Summary (Session)`: shows route usage and structural distribution.
- `Cohesive Method Header`: active artifact preview, structural profile, routing decision, and decision trace.
- `Evidence Navigation`: move through ranked evidence list (`All`, `Belief`, `Speech`, `Behavior`).
- `Citation Explorer`: inspect citation links for selected source.
- `Source Explorer`: open source context, citations, and primary-text jumps.

## Evidence Navigation Controls

- `Prev` / `Next`: move through ranked evidence.
- `Scope`: All / Belief / Speech / Behavior.
- `Go to #`: jump to item index in the current scope.
- `Search evidence`: find first matching artifact in current scope.
- `Prefer Primary ON/OFF`: bias ranking toward primary text candidates.
- `Add Tag` / `Add Note`: save artifacts for revisit (stored in browser local storage).
- `Tagged` + `Load Tagged`: jump to tagged item.

## Keyboard Shortcuts

- `J`: next evidence
- `K`: previous evidence
- `/`: focus evidence search input
- `Enter`: open active source in Source Explorer
- `E`: toggle Citation Explorer
- `Esc`: close tooltip

## Viewing Artifacts and Visual Sources

For the active artifact:

- `Open source record`: opens the page record route.
- `Show excerpt`: opens Source Explorer excerpt.
- `Open linked primary excerpt`: jumps from commentary/wrapper context to primary linked text when available.
- `Open likely visual source`: opens best matched external visual URL.
- `Open visual candidates`: opens ranked list of visual candidates.
- `Compare source vs visual`: opens side-by-side compare page.

Notes:

- Not every source has a local image scan indexed.
- Context/commentary sources can be valid retrieval evidence but may not contain a translatable primary tablet.

## Full Corpus Browsing

Use:

- `Browse All Sources` from UI header, or
- `http://127.0.0.1:50001/corpus/browse`

Capabilities:

- paginated listing across all indexed artifacts
- text search by source/page id
- `goto_page` and `goto_item` jump support

## Useful Routes

- `/corpus/` - API route index
- `/corpus/help` - full help page
- `/corpus/mim` - main UI
- `/corpus/demo?limit=25` - demo JSON payload
- `/corpus/stats` - corpus stats
- `/corpus/browse` - paginated source browser
- `/corpus/page/<page_id>` - page bundle JSON
- `/corpus/page/<page_id>/citations` - citation list
- `/corpus/page/<page_id>/story` - story/summary payload
- `/corpus/page/<page_id>/visuals` - visual candidate list
- `/corpus/page/<page_id>/compare` - side-by-side source vs visual compare

## Troubleshooting

- If `http://127.0.0.1:50001/` shows `Not Found`, this is expected. Use `/corpus/mim`.
- If `View sources` fetch succeeds but UI does not update, hard-refresh and retest after server restart.
- If no thumbnail appears, use visual candidate links and source record links.
- If server reports syntax error after edits, run:

```powershell
python -m compileall app/corpus/query_api.py
```
