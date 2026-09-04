import { useState, useEffect, useRef, useCallback } from 'react'

/**
 * TrainingPanel — Pipeline Control + Run History + Live Loss Curve
 *
 * Sub-tabs:
 *   Pipeline  — Config form → Prepare → Train (SSE live logs + loss curve)
 *   Runs      — MLflow run history, promote button, current production model
 *
 * Endpoints:
 *   POST /training/prepare           → dataset stats
 *   POST /training/run               → SSE stream (training logs)
 *   POST /training/run/stop          → kill training
 *   GET  /training/runs              → MLflow run list
 *   POST /training/promote/:run_id   → promote to production
 *   GET  /training/model/current     → current production model
 *   GET  /training/config            → default config
 */

const API = ''

// ── Tiny reusable atoms ────────────────────────────────────────────────────

function Chip({ label, value, color }) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      padding: '3px 10px', borderRadius: 999,
      background: 'var(--bg-secondary)', fontSize: '0.72rem',
      fontFamily: 'var(--font-mono)',
    }}>
      <span style={{ opacity: 0.6 }}>{label}</span>
      <strong style={color ? { color } : {}}>{value}</strong>
    </span>
  )
}

function SectionTitle({ children }) {
  return (
    <h3 style={{
      fontSize: '0.78rem', fontWeight: 600, textTransform: 'uppercase',
      letterSpacing: '0.08em', color: 'var(--text-secondary)',
      margin: '16px 0 8px', borderBottom: '1px solid var(--border)',
      paddingBottom: 4,
    }}>{children}</h3>
  )
}

// ── Mini loss curve (SVG) ──────────────────────────────────────────────────

function LossCurve({ logs }) {
  if (!logs || logs.length < 2) return null

  const losses = logs.map(l => l.loss ?? l.train_loss ?? l['train/loss']).filter(v => typeof v === 'number')
  if (losses.length < 2) return null

  const W = 500, H = 160, PAD = 30
  const minL = Math.min(...losses) * 0.95
  const maxL = Math.max(...losses) * 1.05
  const rangeL = maxL - minL || 1

  const points = losses.map((v, i) => {
    const x = PAD + (i / (losses.length - 1)) * (W - 2 * PAD)
    const y = PAD + (1 - (v - minL) / rangeL) * (H - 2 * PAD)
    return `${x},${y}`
  })

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxHeight: 160, marginTop: 8 }}>
      <rect x={PAD} y={PAD} width={W - 2 * PAD} height={H - 2 * PAD}
        fill="none" stroke="var(--border)" strokeWidth="0.5" />
      {/* Y axis labels */}
      <text x={PAD - 4} y={PAD + 4} textAnchor="end" fontSize="8" fill="var(--text-secondary)">
        {maxL.toFixed(3)}
      </text>
      <text x={PAD - 4} y={H - PAD + 4} textAnchor="end" fontSize="8" fill="var(--text-secondary)">
        {minL.toFixed(3)}
      </text>
      {/* Loss curve */}
      <polyline
        points={points.join(' ')}
        fill="none" stroke="#e94560" strokeWidth="1.5"
        strokeLinejoin="round" strokeLinecap="round"
      />
      {/* Latest point */}
      {points.length > 0 && (
        <circle
          cx={parseFloat(points[points.length - 1].split(',')[0])}
          cy={parseFloat(points[points.length - 1].split(',')[1])}
          r="3" fill="#e94560"
        />
      )}
      {/* Label */}
      <text x={W / 2} y={H - 4} textAnchor="middle" fontSize="9" fill="var(--text-secondary)">
        Steps ({losses.length})
      </text>
    </svg>
  )
}


// ═════════════════════════════════════════════════════════════════════════════
// Pipeline Tab
// ═════════════════════════════════════════════════════════════════════════════

function PipelineTab() {
  const [config, setConfig] = useState({
    method: 'raft',
    lora_rank: 16,
    lora_alpha: 32,
    learning_rate: 0.0002,
    epochs: 3,
    batch_size: 1,
    gradient_accumulation: 8,
    max_seq_length: 1024,
    dpo_beta: 0.1,
    base_model: '',
  })

  const [prepStats, setPrepStats]     = useState(null)
  const [preparing, setPreparing]     = useState(false)
  const [training, setTraining]       = useState(false)
  const [trainLogs, setTrainLogs]     = useState([])
  const [trainStatus, setTrainStatus] = useState(null) // null | 'running' | 'complete' | 'error' | 'stopped'
  const logEndRef = useRef(null)

  // Load default config on mount
  useEffect(() => {
    fetch(`${API}/training/config`)
      .then(r => r.json())
      .then(data => {
        setConfig(prev => ({
          ...prev,
          lora_rank: data.lora_rank ?? prev.lora_rank,
          lora_alpha: data.lora_alpha ?? prev.lora_alpha,
          learning_rate: data.learning_rate ?? prev.learning_rate,
          epochs: data.epochs ?? prev.epochs,
          base_model: data.base_model ?? '',
        }))
      })
      .catch(() => {})
  }, [])

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [trainLogs])

  // ── Prepare ──────────────────────────────────────────────────────────────

  const handlePrepare = async () => {
    setPreparing(true)
    setPrepStats(null)
    try {
      const res = await fetch(`${API}/training/prepare`, { method: 'POST' })
      const data = await res.json()
      setPrepStats(data)
    } catch (e) {
      setPrepStats({ error: e.message })
    } finally {
      setPreparing(false)
    }
  }

  // ── Train (SSE) ──────────────────────────────────────────────────────────

  const handleTrain = async () => {
    setTraining(true)
    setTrainLogs([])
    setTrainStatus('running')

    try {
      const res = await fetch(`${API}/training/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const evt = JSON.parse(line.slice(6))
            setTrainLogs(prev => [...prev, evt])

            if (evt.type === 'complete')  setTrainStatus('complete')
            if (evt.type === 'error')     setTrainStatus('error')
            if (evt.type === 'stopped')   setTrainStatus('stopped')
          } catch {}
        }
      }
    } catch (e) {
      setTrainLogs(prev => [...prev, { type: 'error', message: e.message }])
      setTrainStatus('error')
    } finally {
      setTraining(false)
    }
  }

  const handleStop = async () => {
    try {
      await fetch(`${API}/training/run/stop`, { method: 'POST' })
    } catch {}
  }

  // ── Config field helper ──────────────────────────────────────────────────

  const Field = ({ label, field, type = 'number', options, step }) => (
    <label style={{ display: 'flex', flexDirection: 'column', gap: 2, fontSize: '0.72rem' }}>
      <span style={{ opacity: 0.7, fontWeight: 500 }}>{label}</span>
      {options ? (
        <select
          value={config[field]}
          onChange={e => setConfig(c => ({ ...c, [field]: e.target.value }))}
          style={{
            padding: '5px 8px', borderRadius: 6, border: '1px solid var(--border)',
            background: 'var(--bg-primary)', color: 'var(--text-primary)',
            fontSize: '0.72rem', fontFamily: 'var(--font-mono)',
          }}
        >
          {options.map(o => <option key={o} value={o}>{o}</option>)}
        </select>
      ) : (
        <input
          type={type}
          step={step}
          value={config[field]}
          onChange={e => setConfig(c => ({
            ...c,
            [field]: type === 'number' ? parseFloat(e.target.value) : e.target.value,
          }))}
          style={{
            padding: '5px 8px', borderRadius: 6, border: '1px solid var(--border)',
            background: 'var(--bg-primary)', color: 'var(--text-primary)',
            fontSize: '0.72rem', fontFamily: 'var(--font-mono)', width: '100%',
          }}
        />
      )}
    </label>
  )

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <div style={{ padding: '16px 20px', overflow: 'auto', height: '100%' }}>

      {/* ── Config Form ──────────────────────────────────────────────── */}
      <SectionTitle>Training Configuration</SectionTitle>
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))',
        gap: 10, marginBottom: 16,
      }}>
        <Field label="Method" field="method" options={['raft']} />
        <Field label="LoRA Rank" field="lora_rank" />
        <Field label="LoRA Alpha" field="lora_alpha" />
        <Field label="Learning Rate" field="learning_rate" step={0.00001} />
        <Field label="Epochs" field="epochs" />
        <Field label="Batch Size" field="batch_size" />
        <Field label="Grad Accumulation" field="gradient_accumulation" />
        <Field label="Max Seq Length" field="max_seq_length" />
      </div>

      {config.base_model && (
        <div style={{ fontSize: '0.68rem', opacity: 0.5, marginBottom: 12, fontFamily: 'var(--font-mono)' }}>
          Base model: {config.base_model}
        </div>
      )}

      {/* ── Action Buttons ───────────────────────────────────────────── */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <button
          onClick={handlePrepare}
          disabled={preparing || training}
          className="tp-btn tp-btn-secondary"
        >
          {preparing ? '⟳ Preparing...' : '1. Prepare Dataset'}
        </button>

        <button
          onClick={handleTrain}
          disabled={training || !prepStats || prepStats.error}
          className="tp-btn tp-btn-primary"
        >
          {training ? '⟳ Training...' : '2. Start Training'}
        </button>

        {training && (
          <button onClick={handleStop} className="tp-btn tp-btn-danger">
            Stop
          </button>
        )}
      </div>

      {/* ── Prep Stats ───────────────────────────────────────────────── */}
      {prepStats && !prepStats.error && (
        <div style={{
          background: 'var(--bg-secondary)', borderRadius: 8, padding: 12,
          marginBottom: 16, border: '1px solid var(--border)',
        }}>
          <div style={{ fontSize: '0.7rem', fontWeight: 600, marginBottom: 6 }}>Dataset Ready</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            <Chip label="QLoRA train" value={prepStats.qlora_train} color="#10b981" />
            <Chip label="QLoRA eval" value={prepStats.qlora_eval} />
            <Chip label="DPO train" value={prepStats.dpo_train} color="#e94560" />
            <Chip label="DPO eval" value={prepStats.dpo_eval} />
            <Chip label="Deduped" value={prepStats.qlora_deduped} />
            <Chip label="DPO uncurated" value={prepStats.dpo_uncurated} color="#f59e0b" />
          </div>
          <div style={{ fontSize: '0.62rem', opacity: 0.5, marginTop: 6, fontFamily: 'var(--font-mono)' }}>
            Snapshot: {prepStats.snapshot}
          </div>
        </div>
      )}

      {prepStats?.error && (
        <div style={{
          background: 'rgba(239,68,68,0.1)', borderRadius: 8, padding: 12,
          marginBottom: 16, color: '#ef4444', fontSize: '0.72rem',
        }}>
          Preparation failed: {prepStats.error}
        </div>
      )}

      {/* ── Loss Curve ───────────────────────────────────────────────── */}
      {trainLogs.length > 0 && (
        <>
          <SectionTitle>
            Training Progress
            {trainStatus === 'running' && <span style={{ color: '#10b981' }}> (live)</span>}
            {trainStatus === 'complete' && <span style={{ color: '#10b981' }}> — done</span>}
            {trainStatus === 'error' && <span style={{ color: '#ef4444' }}> — failed</span>}
            {trainStatus === 'stopped' && <span style={{ color: '#f59e0b' }}> — stopped</span>}
          </SectionTitle>

          <LossCurve logs={trainLogs.filter(l => l.type === 'log' || l.type === 'metrics')} />

          {/* Log stream */}
          <div style={{
            maxHeight: 200, overflow: 'auto', background: 'var(--bg-primary)',
            borderRadius: 6, padding: 8, marginTop: 8, border: '1px solid var(--border)',
            fontFamily: 'var(--font-mono)', fontSize: '0.65rem', lineHeight: 1.6,
          }}>
            {trainLogs.map((l, i) => (
              <div key={i} style={{
                color: l.type === 'error' ? '#ef4444'
                     : l.type === 'complete' ? '#10b981'
                     : l.type === 'stopped' ? '#f59e0b'
                     : 'var(--text-secondary)',
                opacity: l.type === 'log' ? 0.7 : 1,
              }}>
                {l.message || JSON.stringify(l)}
              </div>
            ))}
            <div ref={logEndRef} />
          </div>
        </>
      )}
    </div>
  )
}


// ═════════════════════════════════════════════════════════════════════════════
// Runs Tab (MLflow history + promote)
// ═════════════════════════════════════════════════════════════════════════════

function RunsTab({ view = 'runs' }) {
  const [runs, setRuns]               = useState([])
  const [verdict, setVerdict]         = useState(null)
  const [currentModel, setCurrentModel] = useState(null)
  const [promoting, setPromoting]     = useState(null)
  const [serving, setServing]         = useState(null)
  const [activating, setActivating]   = useState(null)

  useEffect(() => {
    loadRuns()
    loadCurrentModel()
    loadServingModels()
  }, [])

  const loadRuns = async () => {
    try {
      const res = await fetch(`${API}/training/runs`)
      const data = await res.json()
      setRuns(data.runs || [])
    } catch {}
  }

  const loadCurrentModel = async () => {
    try {
      const res = await fetch(`${API}/training/model/current`)
      setCurrentModel(await res.json())
    } catch {}
  }

  // Serving model — a trained adapter reaches production as an Ollama model
  // layering it onto the same base (FROM qwen3:8b + ADAPTER), so activating a
  // fine-tune only changes which tag Weaver generates with. Prompts and decoding
  // parameters stay identical, which is what keeps before/after comparable.
  const loadServingModels = async () => {
    try {
      const res = await fetch(`${API}/training/model/available`)
      const data = await res.json()
      setServing(data)
    } catch {}
  }

  const handleActivate = async (name) => {
    setActivating(name)
    try {
      const res = await fetch(`${API}/training/model/activate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: name }),
      })
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText)
      await loadServingModels()
    } catch (e) {
      alert(`Activation failed: ${e.message}`)
    } finally {
      setActivating(null)
    }
  }

  // Promotion is a request for a decision, not an instruction to ship. The
  // gate scores the run against thresholds fixed before training and answers
  // PROMOTE or REJECT; only a PROMOTE registers the run and switches the
  // served model. The verdict is rendered either way, so a refusal shows the
  // rule that produced it instead of failing silently.
  const handlePromote = async (runId, metricsAfter) => {
    setPromoting(runId)
    setVerdict(null)
    try {
      const res = await fetch(`${API}/training/promote/${runId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ metrics_after: metricsAfter || null }),
      })
      const data = await res.json()
      if (!res.ok && !data.verdict) throw new Error(data.detail || res.statusText)
      setVerdict(data)
      if (data.promoted) {
        await loadRuns()
        await loadCurrentModel()
        await loadServingModels()
      }
    } catch (e) {
      alert(`Gate evaluation failed: ${e.message}`)
    } finally {
      setPromoting(null)
    }
  }

  const fmtTime = (ts) => {
    if (!ts) return '—'
    return new Date(ts).toLocaleString(undefined, {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    })
  }

  return (
    <div style={{ padding: '16px 20px', overflow: 'auto', height: '100%' }}>

      {/* Current Production */}
      {view === 'runs' && (<>
      {/* Promotion-gate verdict.
          Promotion is not a button that ships a model: it is a request for a
          decision. The gate answers, and the answer is shown here with the
          rule that produced it -- including when it refuses. */}
      {verdict && (
        <div style={{
          background: verdict.verdict === 'PROMOTE'
            ? 'rgba(16,185,129,0.07)' : 'rgba(239,68,68,0.07)',
          border: `1px solid ${verdict.verdict === 'PROMOTE'
            ? 'rgba(16,185,129,0.35)' : 'rgba(239,68,68,0.35)'}`,
          borderRadius: 8, padding: '14px 16px', marginBottom: 16,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
            <span style={{
              fontSize: '0.72rem', fontWeight: 800, letterSpacing: '.06em',
              color: verdict.verdict === 'PROMOTE' ? '#10b981' : '#ef4444',
            }}>
              GATE: {verdict.verdict}
            </span>
            <span style={{ fontSize: '0.64rem', opacity: 0.55, fontFamily: 'var(--font-mono)' }}>
              n = {verdict.n_measurable}
            </span>
            {verdict.promoted && (
              <span style={{
                background: '#10b981', color: '#fff', padding: '1px 8px',
                borderRadius: 999, fontSize: '0.58rem', fontWeight: 700,
              }}>
                SERVING {verdict.activated}
              </span>
            )}
          </div>

          <div style={{ fontSize: '0.7rem', opacity: 0.75, marginBottom: 10, lineHeight: 1.5 }}>
            {verdict.reason}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {[
              ['PRIMARY',   'golden-passage recall', verdict.metrics?.primary,   true],
              ['SECONDARY', 'fabrication rate',      verdict.metrics?.secondary, false],
              ['GUARD',     'refusal rate',          verdict.metrics?.guard,     false],
              ['MECHANISM', 'quote grounding',       verdict.metrics?.mechanism, false],
            ].map(([role, label, m, withSigma]) => m && (
              <div key={role} style={{
                display: 'grid',
                gridTemplateColumns: '84px 1fr 128px 74px',
                alignItems: 'center', fontSize: '0.66rem',
                fontFamily: 'var(--font-mono)',
              }}>
                <span style={{ color: '#3b82f6', fontWeight: 700 }}>{role}</span>
                <span style={{ opacity: 0.7 }}>{label}</span>
                <span style={{ opacity: 0.85 }}>
                  {Number(m.before).toFixed(1)}% → {Number(m.after).toFixed(1)}%
                </span>
                <span style={{
                  textAlign: 'right', fontWeight: 700,
                  color: m.delta_pts > 0 ? '#10b981' : '#9ca3af',
                }}>
                  {m.delta_pts > 0 ? '+' : ''}{Number(m.delta_pts).toFixed(1)} pts
                  {withSigma && m.sigma != null && (
                    <span style={{
                      marginLeft: 6,
                      color: m.sigma >= (verdict.thresholds?.min_sigma ?? 2)
                        ? '#10b981' : '#ef4444',
                    }}>
                      {Number(m.sigma).toFixed(1)} SE
                    </span>
                  )}
                </span>
              </div>
            ))}
          </div>

          <div style={{
            fontSize: '0.6rem', opacity: 0.45, marginTop: 9,
            fontFamily: 'var(--font-mono)',
          }}>
            thresholds fixed before training · min {verdict.thresholds?.min_sigma} SE ·
            refusal ≤ +{verdict.thresholds?.refusal_degenerate_pts} pts ·
            fabrication ≤ +{verdict.thresholds?.fabrication_tolerance_pts} pts
          </div>
        </div>
      )}

      </>)}

      {view === 'serving' && (<>
      <SectionTitle>Active Production Model</SectionTitle>
      {currentModel?.active ? (
        <div style={{
          background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.25)',
          borderRadius: 8, padding: 12, marginBottom: 16,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ color: '#10b981', fontWeight: 700, fontSize: '0.78rem' }}>
              {currentModel.model_name} v{currentModel.version}
            </span>
            <Chip label="run" value={currentModel.run_id?.slice(0, 8)} />
          </div>
        </div>
      ) : (
        <div style={{
          background: 'var(--bg-secondary)', borderRadius: 8, padding: 12,
          marginBottom: 16, fontSize: '0.72rem', opacity: 0.6,
        }}>
          No trained adapter active — using base Ollama model
        </div>
      )}

      {/* Serving model — attach / detach a trained adapter */}
      <SectionTitle>Serving Model</SectionTitle>
      <div style={{ marginBottom: 16 }}>
        {serving?.models?.length ? (
          <>
            <div style={{ fontSize: '0.68rem', opacity: 0.55, marginBottom: 8 }}>
              A trained adapter is served as <code>FROM {serving.boot_default} + ADAPTER</code>.
              Switching changes the weights only — prompts and decoding parameters are unchanged.
            </div>
            {serving.models.filter(m => !m.name.includes('-cloud')).map(m => (
              <div key={m.name} style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                background: m.active ? 'rgba(16,185,129,0.08)' : 'var(--bg-secondary)',
                border: m.active ? '1px solid rgba(16,185,129,0.25)' : '1px solid transparent',
                borderRadius: 8, padding: '8px 12px', marginBottom: 6,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{
                    fontWeight: m.active ? 700 : 500, fontSize: '0.74rem',
                    color: m.active ? '#10b981' : 'var(--text-primary)',
                  }}>{m.name}</span>
                  {m.is_adapter && <Chip label="adapter" value="LoRA" />}
                  {m.size && <Chip label="size" value={`${(m.size / 1e9).toFixed(1)} GB`} />}
                </div>
                {m.active ? (
                  <span style={{ fontSize: '0.68rem', color: '#10b981', fontWeight: 700 }}>ACTIVE</span>
                ) : (
                  <button
                    onClick={() => handleActivate(m.name)}
                    disabled={!!activating}
                    style={{
                      background: 'var(--accent)', color: '#fff', border: 'none',
                      borderRadius: 6, padding: '4px 12px', fontSize: '0.68rem',
                      fontWeight: 600, cursor: activating ? 'wait' : 'pointer',
                      opacity: activating ? 0.5 : 1,
                    }}>
                    {activating === m.name ? 'Activating…' : 'Activate'}
                  </button>
                )}
              </div>
            ))}
          </>
        ) : (
          <div style={{
            background: 'var(--bg-secondary)', borderRadius: 8, padding: 12,
            fontSize: '0.72rem', opacity: 0.6,
          }}>
            {serving?.error ? `Ollama unreachable — ${serving.error}` : 'Loading models…'}
          </div>
        )}
      </div>

      </>)}

      {view === 'runs' && (<>
      {/* Run History */}
      <SectionTitle>Training Runs ({runs.length})</SectionTitle>
      {runs.length === 0 ? (
        <div style={{ fontSize: '0.72rem', opacity: 0.5, padding: 16, textAlign: 'center' }}>
          No training runs yet. Use the Pipeline tab to start training.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {runs.map(run => (
            <div key={run.run_id} style={{
              background: run.is_production ? 'rgba(16,185,129,0.06)' : 'var(--bg-secondary)',
              border: `1px solid ${run.is_production ? 'rgba(16,185,129,0.3)' : 'var(--border)'}`,
              borderRadius: 8, padding: 12,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontWeight: 600, fontSize: '0.76rem' }}>
                    {run.run_name}
                  </span>
                  {run.is_production && (
                    <span style={{
                      background: '#10b981', color: '#fff', padding: '1px 8px',
                      borderRadius: 999, fontSize: '0.6rem', fontWeight: 700,
                    }}>
                      PRODUCTION
                    </span>
                  )}
                  <span style={{
                    fontSize: '0.62rem', opacity: 0.5, fontFamily: 'var(--font-mono)',
                  }}>
                    {fmtTime(run.start_time)}
                  </span>
                </div>

                {!run.is_production && run.status === 'FINISHED' && (
                  <button
                    onClick={() => handlePromote(run.run_id)}
                    disabled={promoting === run.run_id}
                    className="tp-btn tp-btn-sm"
                  >
                    {promoting === run.run_id ? 'Gating…' : 'Promote'}
                  </button>
                )}
              </div>

              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                <Chip label="method" value={run.params?.method || '?'} />
                <Chip label="rank" value={run.params?.lora_rank || '?'} />
                <Chip label="lr" value={run.params?.learning_rate || '?'} />
                {run.metrics?.final_train_loss != null && (
                  <Chip label="loss" value={run.metrics.final_train_loss.toFixed(4)} color="#e94560" />
                )}
                {run.metrics?.final_eval_loss != null && (
                  <Chip label="eval" value={run.metrics.final_eval_loss.toFixed(4)} color="#3b82f6" />
                )}
                {run.metrics?.hammer_weighted_score != null && (
                  <Chip label="hammer" value={run.metrics.hammer_weighted_score.toFixed(4)} color="#10b981" />
                )}
              </div>
            </div>
          ))}
        </div>
      )}
      </>)}
    </div>
  )
}


// ═════════════════════════════════════════════════════════════════════════════
// Main TrainingPanel export
// ═════════════════════════════════════════════════════════════════════════════

export default function TrainingPanel() {
  const [tab, setTab] = useState('pipeline')

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Tab bar */}
      <div style={{
        display: 'flex', gap: 0, borderBottom: '1px solid var(--border)',
        padding: '0 16px', background: 'var(--bg-secondary)',
      }}>
        {[
          { key: 'pipeline', label: 'Pipeline' },
          { key: 'runs',     label: 'MLflow Runs' },
          { key: 'serving',  label: 'Serving' },
        ].map(t => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            style={{
              padding: '8px 16px', fontSize: '0.72rem', fontWeight: 600,
              border: 'none', cursor: 'pointer',
              background: tab === t.key ? 'var(--bg-primary)' : 'transparent',
              color: tab === t.key ? 'var(--text-primary)' : 'var(--text-secondary)',
              borderBottom: tab === t.key ? '2px solid #e94560' : '2px solid transparent',
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div style={{ flex: 1, overflow: 'hidden' }}>
        {tab === 'pipeline' && <PipelineTab />}
        {tab === 'runs'     && <RunsTab view="runs" />}
        {tab === 'serving'  && <RunsTab view="serving" />}
      </div>
    </div>
  )
}
