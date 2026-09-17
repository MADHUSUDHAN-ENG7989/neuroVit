import React, { useState, useEffect } from 'react';
import './index.css';

export default function Logs({ API, token }) {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/logs`, {
      headers: { 'Authorization': `Bearer ${token}` }
    })
      .then(res => res.json())
      .then(data => {
        if (data.error) throw new Error(data.error);
        setLogs(data.logs || []);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [API, token]);

  const CLASS_COLORS = {
    glioma: '#ef4444',
    meningioma: '#f59e0b',
    notumor: '#22c55e',
    pituitary: '#a855f7',
  };

  return (
    <div className="page anim-fade-in" style={{ padding: '2rem' }}>
      <div className="hero" style={{ marginBottom: '2rem' }}>
        <h1 className="hero-title">Query Logs</h1>
        <p className="hero-desc">All MRI analysis requests stored in DynamoDB — real-time history.</p>
      </div>

      <div className="card" style={{ maxWidth: '1000px', margin: '0 auto' }}>
        <div className="card-hd">
          <div className="card-ico ico-amber">📋</div>
          <span className="card-title">Prediction History</span>
          <span className="card-chip chip-blue">{logs.length} records</span>
        </div>
        <div className="card-body">
          {loading ? (
            <div style={{ textAlign: 'center', padding: '2rem', color: '#a0a0b0' }}>
              <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>⏳</div>
              Loading logs from DynamoDB...
            </div>
          ) : error ? (
            <div style={{ textAlign: 'center', padding: '2rem', color: '#ef4444' }}>
              ⚠️ {error}
            </div>
          ) : logs.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '2rem', color: '#a0a0b0' }}>
              No query logs found yet. Run an MRI analysis to see records here.
            </div>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', color: '#e0e0e0', fontSize: '0.9rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.15)', color: '#a0a0b0', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    <th style={{ padding: '0.85rem 1rem' }}>Timestamp</th>
                    <th style={{ padding: '0.85rem 1rem' }}>File</th>
                    <th style={{ padding: '0.85rem 1rem' }}>Prediction</th>
                    <th style={{ padding: '0.85rem 1rem' }}>Confidence</th>
                    <th style={{ padding: '0.85rem 1rem' }}>User</th>
                  </tr>
                </thead>
                <tbody>
                  {logs.map((log, i) => (
                    <tr key={log.logId || i} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', transition: 'background 0.15s' }}
                      onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.03)'}
                      onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                    >
                      <td style={{ padding: '0.85rem 1rem' }}>{new Date(log.timestamp).toLocaleString()}</td>
                      <td style={{ padding: '0.85rem 1rem', color: '#a0a0b0', fontFamily: 'monospace', fontSize: '0.8rem' }}>{log.filename}</td>
                      <td style={{ padding: '0.85rem 1rem' }}>
                        <span style={{
                          background: `${CLASS_COLORS[log.prediction] || '#6366f1'}22`,
                          color: CLASS_COLORS[log.prediction] || '#6366f1',
                          border: `1px solid ${CLASS_COLORS[log.prediction] || '#6366f1'}44`,
                          padding: '0.2rem 0.6rem',
                          borderRadius: '999px',
                          fontSize: '0.8rem',
                          fontWeight: 600,
                          textTransform: 'capitalize',
                        }}>{log.prediction}</span>
                      </td>
                      <td style={{ padding: '0.85rem 1rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <div style={{ flex: 1, height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', maxWidth: '80px' }}>
                            <div style={{ height: '100%', borderRadius: '2px', background: CLASS_COLORS[log.prediction] || '#6366f1', width: `${(parseFloat(log.confidence) * 100).toFixed(0)}%` }} />
                          </div>
                          <span style={{ color: '#e0e0e0', fontWeight: 600 }}>{(parseFloat(log.confidence) * 100).toFixed(1)}%</span>
                        </div>
                      </td>
                      <td style={{ padding: '0.85rem 1rem', color: '#a0a0b0' }}>{log.userEmail || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
