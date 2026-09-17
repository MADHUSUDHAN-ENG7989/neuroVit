import { useState, useRef, useEffect, useCallback } from 'react'
import './index.css'

const CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

const CLASS_META = {
  glioma:     { icon: '🔴', desc: 'Aggressive tumor from glial cells', type: 'tumor-yes' },
  meningioma: { icon: '🟡', desc: 'Tumor on the brain membranes', type: 'tumor-yes' },
  notumor:    { icon: '✅', desc: 'No tumor detected in this scan', type: 'tumor-no' },
  pituitary:  { icon: '🟣', desc: 'Tumor on the pituitary gland', type: 'tumor-yes' },
}

const STEPS = [
  'Preprocessing with CLAHE',
  'PDSCNN feature extraction',
  'ViT transformer encoding',
  'RRELM ensemble fusion',
  'Generating Grad-CAM',
  'Computing SHAP values',
]

function fmt(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`
}

/* ── Hero ── */
function Hero() {
  return (
    <div className="hero">
      <div className="hero-chip">✦ AI-Powered Neuro-Diagnostics</div>
      <h1 className="hero-title">NeuroSight AI</h1>
      <p className="hero-desc">
        Upload a brain MRI scan and receive an instant AI-powered tumor classification
        with Grad-CAM heatmaps and SHAP explainability.
      </p>
      <div className="hero-metrics">
        <div className="metric"><span className="metric-val">4</span><span className="metric-lbl">Tumor Classes</span></div>
        <div className="metric"><span className="metric-val">224px</span><span className="metric-lbl">Input Size</span></div>
        <div className="metric"><span className="metric-val">2</span><span className="metric-lbl">Ensemble Models</span></div>
        <div className="metric"><span className="metric-val">XAI</span><span className="metric-lbl">Explainable</span></div>
      </div>
    </div>
  )
}

/* ── Upload card ── */
function UploadCard({ file, preview, dragging, ref_, onDragOver, onDragLeave, onDrop, onClickZone, onChange }) {
  return (
    <div className="card">
      <div className="card-hd">
        <div className="card-ico ico-blue">📤</div>
        <span className="card-title">Upload MRI Scan</span>
        <span className="card-chip chip-blue">DICOM / IMG</span>
      </div>
      <div className="card-body">
        {!preview ? (
          <div
            className={`drop-zone${dragging ? ' dragging' : ''}`}
            onDragOver={onDragOver} onDragLeave={onDragLeave}
            onDrop={onDrop} onClick={onClickZone}
          >
            <div className="dz-brain">🧠</div>
            <p className="dz-label">Drop MRI scan here</p>
            <p className="dz-sub">or click to browse files</p>
            <div className="dz-fmts">
              {['jpg', 'png', 'jpeg', 'bmp', 'tiff'].map(f => <span key={f} className="fmt">.{f}</span>)}
            </div>
            <input className="file-input" type="file" ref={ref_} onChange={onChange} accept="image/*" />
          </div>
        ) : (
          <div className="img-preview">
            <img src={preview} alt="MRI preview" />
            <div className="img-meta">
              <span className="img-name">{file?.name}</span>
              <span className="img-size">{fmt(file?.size || 0)}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

/* ── Model info ── */
function ModelCard() {
  const rows = [
    { l: 'Architecture',   v: 'PDSCNN + ViT-B/16', hi: false },
    { l: 'Fusion',         v: 'RRELM Classifier',  hi: true },
    { l: 'Preprocessing',  v: 'CLAHE Adaptive',    hi: false },
    { l: 'Explainability', v: 'Grad-CAM + SHAP',   hi: false },
    { l: 'Input size',     v: '224 × 224 px',      hi: false },
    { l: 'Classes',        v: '4 (Glioma / Mening / None / Pit.)', hi: false },
  ]
  return (
    <div className="card">
      <div className="card-hd">
        <div className="card-ico ico-violet">⚙️</div>
        <span className="card-title">Model Configuration</span>
      </div>
      <div className="card-body">
        <div className="info-rows">
          {rows.map(r => (
            <div key={r.l} className="info-row">
              <span className="info-lbl">{r.l}</span>
              <span className={`info-val${r.hi ? ' hi' : ''}`}>{r.v}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

/* ── Loading ── */
function LoadingPanel({ step }) {
  return (
    <div className="card">
      <div className="card-hd">
        <div className="card-ico ico-blue">⚡</div>
        <span className="card-title">Analyzing Scan…</span>
      </div>
      <div className="loading-card">
        <div className="spinner">
          <div className="spin-ring" />
          <div className="spin-ring" />
          <div className="spin-ring" />
          <div className="spin-center">🧠</div>
        </div>
        <div style={{ textAlign: 'center', marginTop: '1.5rem', color: '#a0a0b0', fontSize: '0.9rem' }}>
          <p>Processing MRI and generating AI diagnosis...</p>
          <p style={{ marginTop: '0.5rem', color: '#f59e0b', fontSize: '0.8rem' }}>
            Note: Generating SHAP XAI explanations requires heavy computation. This may take 1-2 minutes on standard CPUs. Please don't close this page.
          </p>
        </div>
        <div className="steps">
          {STEPS.map((s, i) => {
            const state = i < step ? 'done' : i === step ? 'active' : 'pending'
            return (
              <div key={s} className={`step ${state}`}>
                <div className="step-dot">{state === 'done' ? '✓' : i + 1}</div>
                {s}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

/* ── Result banner ── */
function ResultBanner({ pred }) {
  const m = CLASS_META[pred.prediction] || {}
  const pct = (pred.confidence * 100).toFixed(1)
  return (
    <div className={`result-banner anim ${m.type || 'tumor-no'}`}>
      <div className="res-icon">{m.icon || '🔬'}</div>
      <div className="res-body">
        <div className="res-eyebrow">AI Diagnosis</div>
        <div className="res-name">{pred.prediction}</div>
        <div className="res-desc">{m.desc}</div>
      </div>
      <div className="res-conf">
        <div className="conf-val">{pct}%</div>
        <div className="conf-lbl">Confidence</div>
      </div>
    </div>
  )
}

/* ── Probabilities ── */
function ProbCard({ pred }) {
  const c = pred.confidence
  const rest = (1 - c) / (CLASSES.length - 1)
  const probs = CLASSES.reduce((a, k) => ({ ...a, [k]: k === pred.prediction ? c : rest }), {})

  return (
    <div className="card anim anim-1">
      <div className="card-hd">
        <div className="card-ico ico-amber">📊</div>
        <span className="card-title">Class Probabilities</span>
      </div>
      <div className="card-body">
        <div className="prob-list">
          {CLASSES.map(cls => {
            const pct = (probs[cls] * 100).toFixed(1)
            const active = cls === pred.prediction
            return (
              <div key={cls} className="prob-item">
                <span className={`prob-name${active ? ' active-cls' : ''}`}>
                  {cls.charAt(0).toUpperCase() + cls.slice(1)}
                </span>
                <div className="prob-track">
                  <div className={`prob-bar ${cls}`} style={{ width: `${pct}%` }} />
                </div>
                <span className={`prob-pct${active ? ' active-cls' : ''}`}>{pct}%</span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

/* ── Visualizations ── */
function VizCard({ pred, onZoom, API }) {
  const tiles = [
    { label: 'Original MRI', badge: 'INPUT', url: `${API}/${pred.original}` },
    ...(pred.gradcam ? [{ label: 'Grad-CAM Heatmap', badge: 'PDSCNN', url: `${API}/${pred.gradcam}` }] : []),
    ...(pred.shap    ? [{ label: 'SHAP Explanation', badge: 'XAI',    url: `${API}/${pred.shap}` }] : []),
  ]
  const cols = tiles.length === 1 ? '1fr' : tiles.length === 2 ? '1fr 1fr' : 'repeat(3,1fr)'
  return (
    <div className="card anim anim-2">
      <div className="card-hd">
        <div className="card-ico ico-green">🔬</div>
        <span className="card-title">XAI Visualizations</span>
        <span className="card-chip chip-green">Click to expand</span>
      </div>
      <div className="card-body">
        <div className="viz-grid" style={{ gridTemplateColumns: cols }}>
          {tiles.map(t => (
            <div key={t.label} className="viz-card" onClick={() => onZoom(t)}>
              <div className="viz-top">
                <span className="viz-label">{t.label}</span>
                <span className="viz-badge">{t.badge}</span>
              </div>
              <img src={t.url} alt={t.label} loading="lazy" />
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

/* ── Lightbox ── */
function Lightbox({ tile, onClose }) {
  useEffect(() => {
    const h = e => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [onClose])
  return (
    <div className="lightbox" onClick={onClose}>
      <div className="lightbox-box" onClick={e => e.stopPropagation()}>
        <button className="lightbox-close" onClick={onClose}>✕</button>
        <img src={tile.url} alt={tile.label} />
        <p className="lightbox-cap">{tile.label} — {tile.badge}</p>
      </div>
    </div>
  )
}

/* ── Error ── */
function ErrBox({ msg, onReset }) {
  return (
    <div className="err-box">
      <span className="err-ico">⚠️</span>
      <div>
        <div className="err-ttl">Analysis Failed</div>
        <div className="err-msg">{msg}</div>
        <button className="btn btn-outline" onClick={onReset} style={{ marginTop: '0.7rem', fontSize: '0.8rem' }}>
          ↩ Try Again
        </button>
      </div>
    </div>
  )
}

/* ── Empty state ── */
function Empty() {
  return (
    <div className="empty">
      <div className="empty-ico">🔬</div>
      <div className="empty-title">No scan analyzed yet</div>
      <p className="empty-sub">Upload an MRI scan on the left and click <strong>Analyze Scan</strong> to get the AI diagnosis.</p>
    </div>
  )
}

/* ══════════════════════════════════════════════
   Main App
══════════════════════════════════════════════ */
export default function Home({ API, token }) {
  const [file, setFile]           = useState(null)
  const [preview, setPreview]     = useState(null)
  const [dragging, setDragging]   = useState(false)
  const [pred, setPred]           = useState(null)
  const [loading, setLoading]     = useState(false)
  const [step, setStep]           = useState(0)
  const [error, setError]         = useState(null)
  const [zoom, setZoom]           = useState(null)
  const ref_ = useRef(null)
  const timer = useRef(null)

  useEffect(() => {
    if (loading) {
      setStep(0)
    }
  }, [loading])

  const pickFile = useCallback(f => {
    if (!f.type.startsWith('image/')) { setError('Please upload a valid image file.'); return }
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setError(null)
    setPred(null)
  }, [])

  const onDrop = e => { e.preventDefault(); setDragging(false); if (e.dataTransfer.files?.[0]) pickFile(e.dataTransfer.files[0]) }
  const onChange = e => { if (e.target.files?.[0]) pickFile(e.target.files[0]) }

  const analyze = async () => {
    if (!file) return
    setLoading(true); setError(null); setPred(null)
    const fd = new FormData(); fd.append('image', file)
    try {
      const res = await fetch(`${API}/api/predict`, { 
        method: 'POST', 
        body: fd,
        headers: { 'Authorization': `Bearer ${token}` }
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Analysis failed')
      setStep(STEPS.length - 1)
      setTimeout(() => setPred(data), 350)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const reset = () => { setFile(null); setPreview(null); setPred(null); setError(null); setStep(0) }

  return (
    <>
      <div className="page">
        <Hero />

        <div className="workspace">
          {/* ── LEFT: Upload + controls + model info ── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <UploadCard
              file={file} preview={preview} dragging={dragging} ref_={ref_}
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={e => { e.preventDefault(); setDragging(false) }}
              onDrop={onDrop}
              onClickZone={() => !preview && ref_.current?.click()}
              onChange={onChange}
            />

            {preview && !loading && (
              <div className="btns">
                <button className="btn btn-outline" onClick={reset}>✕ Clear</button>
                <button className="btn btn-cta" onClick={analyze} disabled={!file}>
                  🔬 Analyze Scan
                </button>
              </div>
            )}

            {pred && !loading && (
              <div className="btns">
                <button className="btn btn-outline" style={{ flex: 1 }} onClick={reset}>
                  ↩ Analyze Another Scan
                </button>
              </div>
            )}

            <ModelCard />
          </div>

          {/* ── RIGHT: Results ── */}
          <div className="panel-right">
            {loading && <LoadingPanel step={step} />}

            {error && !loading && <ErrBox msg={error} onReset={reset} />}

            {pred && !loading && (
              <>
                <ResultBanner pred={pred} />
                <ProbCard pred={pred} />
                <VizCard pred={pred} onZoom={setZoom} API={API} />
              </>
            )}

            {!loading && !error && !pred && <Empty />}
          </div>
        </div>
      </div>

      {zoom && <Lightbox tile={zoom} onClose={() => setZoom(null)} />}
    </>
  )
}
