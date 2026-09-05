import { useState, useEffect } from 'react'

/**
 * DataPanel — the RAFT training pool
 *
 * Shows the answers the evaluator graded GOLD with no failure tag: the pool
 * `export_raft` draws its training examples from. Records excluded by an
 * export criterion are listed separately with the reason, and can be restored.
 *
 * Data source:
 *   knowledge_base.chat_history — source of truth; candidates are derived from
 *   it on the fly, so there is no second collection to keep in step.
 *
 * Backend endpoint:
 *   GET    /training/qlora/candidates              → { candidates, excluded }
 *   DELETE /training/qlora/candidates/:id          → exclude one by hand
 *   POST   /training/qlora/candidates/:id/restore  → undo that
 *
 * The route keeps its historical `qlora` path: QLoRA is still the parameter
 * substrate every adapter here is built on, and the pool it serves is the same
 * one. Only the training objective changed.
 */

const API = ''

// ── Shared atoms ──────────────────────────────────────────────────────────────

const TAG_STYLE = {
  hallucination:    { bg: 'rgba(239,68,68,0.12)',    color: '#ef4444',  border: 'rgba(239,68,68,0.25)' },
  tool_misuse:      { bg: 'rgba(245,158,11,0.12)',   color: '#f59e0b',  border: 'rgba(245,158,11,0.25)' },
  format_violation: { bg: 'rgba(139,92,246,0.12)',   color: '#8b5cf6',  border: 'rgba(139,92,246,0.25)' },
  retrieval_miss:   { bg: 'rgba(59,130,246,0.12)',   color: '#3b82f6',  border: 'rgba(59,130,246,0.25)' },
}

const INTENT_STYLE = {
  rag:      { bg: 'rgba(16,185,129,0.12)',  color: '#10b981' },
  sentries: { bg: 'rgba(139,92,246,0.12)', color: '#8b5cf6' },
  both:     { bg: 'rgba(245,158,11,0.12)', color: '#f59e0b' },
}

function Tag({ label }) {
  const s = TAG_STYLE[label] || { bg: 'rgba(100,100,100,0.1)', color: '#94a3b8', border: 'rgba(100,100,100,0.2)' }
  return (
    <span style={{
      padding: '0.12rem 0.5rem', borderRadius: 999,
      fontFamily: 'var(--font-mono)', fontSize: '0.62rem', fontWeight: 600,
      textTransform: 'uppercase', letterSpacing: '0.05em',
      background: s.bg, color: s.color, border: `1px solid ${s.border || s.bg}`,
      whiteSpace: 'nowrap',
    }}>
      {label.replace('_', ' ')}
    </span>
  )
}

function IntentPill({ intent }) {
  if (!intent) return null
  const s = INTENT_STYLE[intent] || { bg: 'rgba(100,100,100,0.1)', color: '#94a3b8' }
  return (
    <span style={{
      padding: '0.1rem 0.45rem', borderRadius: 999,
      fontFamily: 'var(--font-mono)', fontSize: '0.6rem', fontWeight: 500,
      background: s.bg, color: s.color, whiteSpace: 'nowrap',
    }}>
      {intent}
    </span>
  )
}

function StatChip({ label, value, color }) {
  return (
    <span className="dp-stat">
      <strong style={color ? { color } : {}}>{value}</strong>
      <span>{label}</span>
    </span>
  )
}

function FilterPill({ label, active, onClick }) {
  return (
    <button className={`dp-filter-pill ${active ? 'active' : ''}`} onClick={onClick}>
      {label}
    </button>
  )
}

function exportJSONL(records, filename) {
  if (!records.length) { alert('Nothing to export with this filter.'); return }
  const lines = records.map(r => JSON.stringify(r)).join('\n')
  const blob = new Blob([lines], { type: 'application/jsonlines' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
}

// ── RAFT candidate pool ───────────────────────────────────────────────────────

const EXCLUSION_LABELS = {
  no_live_sentries:                   'no live sentries data',
  no_real_context:                    'scored against empty context',
  no_keys_cited_despite_jira_sources: 'no keys cited (generic prose)',
  manually_excluded:                  'manually excluded',
}

function RaftCandidatesPanel() {
  const [candidates, setCandidates] = useState([])
  const [excluded,   setExcluded]   = useState([])
  const [loading,    setLoading]    = useState(true)
  const [error,      setError]      = useState(null)
  const [filter,     setFilter]     = useState('all')   // all | rag | both | sentries
  const [showExcl,   setShowExcl]   = useState(false)

  useEffect(() => {
    fetch(`${API}/training/qlora/candidates`)
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json() })
      .then(d => { setCandidates(d.candidates || []); setExcluded(d.excluded || []); setLoading(false) })
      .catch(() => {
        setError('Could not reach GET /training/qlora/candidates — add this endpoint to your backend.')
        setLoading(false)
      })
  }, [])

  const intentCounts = candidates.reduce((acc, c) => {
    acc[c.intent] = (acc[c.intent] || 0) + 1; return acc
  }, {})

  const balanceOk = (intentCounts.sentries || 0) >= 35
                 && (intentCounts.rag      || 0) >= 25
                 && (intentCounts.both     || 0) >= 25

  const shown = filter === 'all'
    ? candidates
    : candidates.filter(c => c.intent === filter)

  const handleExport = () => {
    const rows = shown.map(c => ({ instruction: c.query, input: '', output: c.answer }))
    exportJSONL(rows, `raft_candidates_${filter === 'all' ? 'train' : filter}.jsonl`)
  }

  const handleDelete = async (id) => {
    try {
      const res = await fetch(`${API}/training/qlora/candidates/${id}`, { method: 'DELETE' })
      if (!res.ok) throw new Error(res.status)
      // Move from candidates → excluded locally so no refetch needed
      const rec = candidates.find(c => c.id === id)
      if (rec) {
        setCandidates(prev => prev.filter(c => c.id !== id))
        setExcluded(prev => [{
          ...rec,
          exclusion_reasons: ['manually_excluded'],
        }, ...prev])
      }
    } catch { alert('Delete failed — check the backend.') }
  }

  const handleRestore = async (id) => {
    try {
      const res = await fetch(`${API}/training/qlora/candidates/${id}/restore`, { method: 'POST' })
      if (!res.ok) throw new Error(res.status)
      const rec = excluded.find(c => c.id === id)
      if (rec) {
        setExcluded(prev => prev.filter(c => c.id !== id))
        setCandidates(prev => [{ ...rec, exclusion_reasons: undefined }, ...prev])
      }
    } catch { alert('Restore failed — check the backend.') }
  }

  return (
    <div className="dp-full-pane">
      <div className="dp-stats-row">
        <StatChip value={candidates.length} label="candidates" />
        <StatChip value={excluded.length}   label="excluded" color="var(--text-muted)" />
        {Object.entries(intentCounts).map(([k, v]) => (
          <span key={k} className="dp-stat" style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <IntentPill intent={k} /> <strong>{v}</strong>
          </span>
        ))}
        {!balanceOk && candidates.length > 0 && (
          <span className="dp-balance-warn">⚠ not balanced — don't train yet</span>
        )}
        <button className="dp-export-btn" onClick={handleExport} disabled={shown.length === 0}>
          Export JSONL
        </button>
      </div>

      <div className="dp-filter-row">
        {[
          { key: 'all',      label: 'All' },
          { key: 'rag',      label: 'RAG' },
          { key: 'both',     label: 'Synthesis (both)' },
          { key: 'sentries', label: 'Sentries' },
        ].map(f => (
          <FilterPill key={f.key} label={f.label} active={filter === f.key} onClick={() => setFilter(f.key)} />
        ))}
      </div>

      {loading && <div className="dp-state-msg">Loading candidates…</div>}
      {error   && <div className="dp-error">{error}</div>}

      {!loading && !error && (
        <>
          <div className="qlora-table-wrapper">
            <table className="qlora-table">
              <thead>
                <tr>
                  <th>Score</th><th>Faith</th><th>Intent</th><th>Method</th><th>Query</th><th></th>
                </tr>
              </thead>
              <tbody>
                {shown.map((c, i) => (
                  <tr key={c.id || i}>
                    <td className="qlora-mono">{c.weighted_score?.toFixed(3)}</td>
                    <td className="qlora-mono">{c.faithfulness?.toFixed(2)}</td>
                    <td><IntentPill intent={c.intent} /></td>
                    <td className="qlora-method">{c.scoring_method}</td>
                    <td className="qlora-query">{c.query}</td>
                    <td>
                      <button
                        onClick={() => handleDelete(c.id)}
                        title="Exclude from training"
                        style={{
                          background: 'none', border: '1px solid rgba(239,68,68,0.25)',
                          borderRadius: 4, padding: '0.2rem 0.35rem', cursor: 'pointer',
                          color: '#ef4444', opacity: 0.6, lineHeight: 1,
                          transition: 'opacity 0.15s',
                        }}
                        onMouseEnter={e => e.currentTarget.style.opacity = 1}
                        onMouseLeave={e => e.currentTarget.style.opacity = 0.6}
                      >
                        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <polyline points="3 6 5 6 21 6"/>
                          <path d="M19 6l-1 14H6L5 6"/>
                          <path d="M10 11v6M14 11v6"/>
                        </svg>
                      </button>
                    </td>
                  </tr>
                ))}
                {shown.length === 0 && (
                  <tr>
                    <td colSpan={5} className="dp-state-msg" style={{ padding: '2rem' }}>
                      No candidates in this filter.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          {/* Excluded accordion */}
          {excluded.length > 0 && (
            <div className="qlora-excluded">
              <button className="sources-toggle" onClick={() => setShowExcl(o => !o)}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                  style={{ transform: showExcl ? 'rotate(90deg)' : 'none', transition: 'transform 0.15s' }}>
                  <polyline points="9 18 15 12 9 6"/>
                </svg>
                {excluded.length} excluded (not safe for training)
              </button>

              {showExcl && (
                <table className="qlora-table" style={{ opacity: 0.55, marginTop: '0.5rem' }}>
                  <thead>
                    <tr><th>Score</th><th>Intent</th><th>Excluded because</th><th>Query</th><th></th></tr>
                  </thead>
                  <tbody>
                    {excluded.map((r, i) => (
                      <tr key={r.id || i}>
                        <td className="qlora-mono">{r.weighted_score?.toFixed(3)}</td>
                        <td><IntentPill intent={r.intent} /></td>
                        <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: '#ef4444' }}>
                          {(r.exclusion_reasons || [])
                            .map(k => EXCLUSION_LABELS[k.replace(/\(.*\)/, '').trim()] || k)
                            .join(', ')}
                        </td>
                        <td className="qlora-query" style={{ color: 'var(--text-muted)' }}>{r.query}</td>
                        <td>
                          {(r.exclusion_reasons || []).includes('manually_excluded') && (
                            <button
                              onClick={() => handleRestore(r.id)}
                              title="Restore to candidates"
                              style={{
                                background: 'none', border: '1px solid rgba(16,185,129,0.3)',
                                borderRadius: 4, padding: '0.2rem 0.4rem', cursor: 'pointer',
                                color: '#10b981', opacity: 0.7, fontSize: '0.75rem',
                                lineHeight: 1, transition: 'opacity 0.15s',
                              }}
                              onMouseEnter={e => e.currentTarget.style.opacity = 1}
                              onMouseLeave={e => e.currentTarget.style.opacity = 0.7}
                            >
                              ↩
                            </button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}

// ── Main DataPanel ─────────────────────────────────────────────────────────────

// The RAFT training pool is the GOLD pool: answers the evaluator graded GOLD,
// with no failure tag, are what `export_raft` turns into training examples.
//
// An earlier rename relabelled the tabs without moving the panels, so the tab
// reading "RAFT Candidates" rendered the preference-rejection pool
// (training_signal == "dpo_rejected") -- the opposite of what RAFT trains on --
// while the GOLD pool sat under "Scored Answers". Preference tuning is no
// longer a method here, so that panel and its curation editor are gone and the
// GOLD pool is what the view shows.
export default function DataPanel() {
  return (
    <div className="data-panel">
      <div className="dp-header">
        <div className="dp-header-left">
          <span className="dp-title">Training Data</span>
          <div className="dp-tabs">
            <button className="dp-tab active">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
              </svg>
              RAFT Candidates
            </button>
          </div>
        </div>
      </div>

      <div className="dp-content">
        <RaftCandidatesPanel />
      </div>
    </div>
  )
}
