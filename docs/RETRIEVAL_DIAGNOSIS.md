# Retrieval Data-Flow Diagnosis — Graph evidence dropped at the weaver

**Date:** 2026-06-22  **Scope:** agent → weaver prompt assembly (production answer path)
**Risk to deposit:** none unless *you* activate the fix (see below). No canonical file was edited.

---

## TL;DR

The Graph-RAG signal — the relationship label (`BLOCKS`, `CAUSES`, `CHILD_OF`…),
the hop distance, and the direct-ticket link list — **is computed by the retriever,
survives normalisation onto every source, and is then silently dropped by the
weaver** before the prompt is built. So in the agent→weaver path (the one the app
actually runs), the model never sees *why* a graph-expanded ticket was retrieved.
The standalone `rag_generator` path renders it correctly; only the weaver doesn't.

This is the "retrieved data that doesn't get passed properly" issue.

---

## Evidence (file : line)

1. **Retriever produces the signal.** `rag/neo4j_search.py` graph-expansion Cypher
   returns `relationship` and `graph_distance` per hit
   (`_jira_expand_cypher`, lines 160–161; `_conf_expand_cypher`, line 192), and
   `hybrid_search` merges graph hits into the returned list with `from_graph=True`
   (lines 445–458, 482–496).

2. **Normalisation keeps it.** `rag/rag_generator.py::_normalise_hit` (lines 131–151)
   copies `from_graph`, `from_direct`, `relationship`, `graph_distance`,
   `graph_context`, **and** `project_name` onto every source dict returned by
   `retrieve()`.

3. **The standalone generator renders it.** `rag/rag_generator.py::_build_context`
   (lines 170–211) emits the `[graph: BLOCKS dist=1]` tag, the `[direct lookup]`
   tag, the `Links:` line, and `Project:` from `project_name`.

4. **The weaver drops it (the bug).** `agent/weaver_node.py::_format_rag_context`
   (lines 309–332) renders only `issue_key / status / priority / assignee / url /
   title / score / text`. It reads **`project`** (the retriever's field is
   `project_name`; `_normalise_hit` sets `project` to the `"N/A"` sentinel, so the
   prompt literally shows `Project: N/A`), and it renders **no** `relationship`,
   `graph_distance`, `from_graph/from_direct`, or `graph_context.links`.

   `agent/intent_agent.py::weaver_node` (lines 654–662) is the production path and
   calls `weave_stream(...)`, so this is what real answers are built from.

### Why it matters
The report and defense position the typed-relationship traversal as the central
contribution ("graph-sourced evidence cited in 6/10 queries"). With the weaver
dropping the relationship label, the model can only use the expanded ticket's
*text* — it cannot tell the user that *PRJ-1834 BLOCKS the ticket you asked about*,
which is the cited-relationship behaviour the evaluation rewards.

---

## The fix (non-breaking, drop-in)

`agent/weaver_node_test.py` — a complete drop-in for `weaver_node.py` (identical
public API: `weave`, `weave_stream`). The **only** change is `_format_rag_context`,
which now mirrors `rag_generator._build_context`:

- `[graph: <REL> dist=<n>]` tag for graph-expanded hits; `[direct lookup]` /
  `[graph-expanded]` tags otherwise;
- `Project:` from `project_name` (and the `"N/A"` sentinel is hidden);
- a `Links:` line from `graph_context.links` for direct-ticket fetches.

Every added field is guarded with `.get()`, so absent fields are simply skipped —
it cannot raise on any source shape. Streaming, `<think>` filtering, payloads,
prompts, budgets and the sentries formatter are byte-for-byte unchanged.

### Proof it works, offline (no Ollama / Neo4j)
```
python agent/weaver_ab_test.py
```
Builds a representative `rag_result` (vector hit + a `BLOCKS` graph hit + a direct
hit with links) and diffs the original vs fixed formatter. Current result: **6/6
PASS** — the fixed context shows `[graph: BLOCKS dist=1]`, `[direct lookup]`,
`Links: …`, and `Project: Product-A`, none of which appear in the original.

---

## How to activate (reversible — do this only after a quick live check)

Until you do this, **nothing changes** and your deposit runs exactly as before.

```bash
cd "agent"
cp weaver_node.py weaver_node_ORIG_backup.py   # keep the original
cp weaver_node_test.py weaver_node.py          # activate the fix
# ... ask a question that triggers graph expansion; confirm answers now
#     reference the relationship ("… which is blocked by TM-xxxx") ...
```
**Revert instantly:**
```bash
cd "agent"
cp weaver_node_ORIG_backup.py weaver_node.py
```
(`weaver_node_test.py` also remains as the import fallback, so the swap is safe.)

Prefer not to swap files? The change is ~25 lines in one function — open
`agent/weaver_node_test.py`, copy the new `_format_rag_context`, and paste it over
the one in `weaver_node.py`. Same result.

---

## Files added (none of your canonical files were modified)
- `agent/weaver_node_test.py` — fixed drop-in (same API).
- `agent/weaver_ab_test.py` — offline A/B proof.
- `RETRIEVAL_DIAGNOSIS.md` — this file.

## Also audited, found clean
- **Sentries → weaver** (`_format_sentries_context`): renders all non-skip fields
  generically; no data dropped.
- **retrieve() → rag_node → weaver**: `rag_result["sources"]` carries the full
  source dicts (not just text); the only loss was the weaver formatter above.
- `rag_node` dedup/threshold/cap (`intent_agent.py` 335–369): correct; the
  `RAG_MIN_SCORE` filter is genuinely applied (a prior bug, already fixed).
