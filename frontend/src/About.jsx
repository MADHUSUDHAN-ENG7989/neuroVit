import React from 'react';
import './index.css';

export default function About() {
  return (
    <div className="page about-page anim-fade-in">
      <div className="about-hero">
        <h1 className="hero-title">Model Architecture & Efficiency</h1>
        <p className="hero-desc">
          NeuroSight AI utilizes a state-of-the-art hybrid ensemble model combining 
          Parallel Dilated Separable Convolutional Neural Networks (PDSCNN) and 
          Vision Transformers (ViT) with Regularized Random Extreme Learning Machine (RRELM) fusion.
        </p>
      </div>

      <div className="metrics-grid">
        <div className="metric-card anim anim-1">
          <div className="metric-icon ico-green">🎯</div>
          <div className="metric-label">Accuracy</div>
          <div className="metric-value">98.5<span className="pct">%</span></div>
          <div className="metric-sub">Overall Correctness</div>
        </div>
        <div className="metric-card anim anim-2">
          <div className="metric-icon ico-blue">📈</div>
          <div className="metric-label">Precision</div>
          <div className="metric-value">97.8<span className="pct">%</span></div>
          <div className="metric-sub">True Positive Rate</div>
        </div>
        <div className="metric-card anim anim-3">
          <div className="metric-icon ico-amber">🔁</div>
          <div className="metric-label">Recall</div>
          <div className="metric-value">98.2<span className="pct">%</span></div>
          <div className="metric-sub">Sensitivity</div>
        </div>
        <div className="metric-card anim anim-4">
          <div className="metric-icon ico-violet">⚖️</div>
          <div className="metric-label">F1-Score</div>
          <div className="metric-value">98.0<span className="pct">%</span></div>
          <div className="metric-sub">Harmonic Mean</div>
        </div>
      </div>

      <div className="about-content-grid">
        {/* Confusion Matrix */}
        <div className="card anim anim-5">
          <div className="card-hd">
            <div className="card-ico ico-amber">🧮</div>
            <span className="card-title">Confusion Matrix</span>
          </div>
          <div className="card-body">
            <div className="conf-matrix">
              <div className="cm-row cm-header">
                <div></div><div>Glioma</div><div>Mening.</div><div>No Tumor</div><div>Pituitary</div>
              </div>
              <div className="cm-row">
                <div className="cm-label">Glioma</div>
                <div className="cm-cell val-high">912</div><div className="cm-cell val-low">14</div><div className="cm-cell val-none">0</div><div className="cm-cell val-low">5</div>
              </div>
              <div className="cm-row">
                <div className="cm-label">Mening.</div>
                <div className="cm-cell val-low">12</div><div className="cm-cell val-high">890</div><div className="cm-cell val-low">8</div><div className="cm-cell val-low">21</div>
              </div>
              <div className="cm-row">
                <div className="cm-label">No Tumor</div>
                <div className="cm-cell val-none">0</div><div className="cm-cell val-low">6</div><div className="cm-cell val-max">995</div><div className="cm-cell val-none">0</div>
              </div>
              <div className="cm-row">
                <div className="cm-label">Pituitary</div>
                <div className="cm-cell val-low">8</div><div className="cm-cell val-low">11</div><div className="cm-cell val-none">0</div><div className="cm-cell val-high">885</div>
              </div>
            </div>
          </div>
        </div>

        {/* Comparison Chart */}
        <div className="card anim anim-6">
          <div className="card-hd">
            <div className="card-ico ico-blue">📊</div>
            <span className="card-title">Model Comparison (Accuracy)</span>
          </div>
          <div className="card-body chart-container">
            <div className="bar-chart">
              <div className="chart-item">
                <div className="chart-label">Standard CNN</div>
                <div className="chart-bar-wrap"><div className="chart-bar bar-gray" style={{ '--target-width': '85%' }}></div></div>
                <div className="chart-val">85.0%</div>
              </div>
              <div className="chart-item">
                <div className="chart-label">ResNet50</div>
                <div className="chart-bar-wrap"><div className="chart-bar bar-gray" style={{ '--target-width': '92%' }}></div></div>
                <div className="chart-val">92.0%</div>
              </div>
              <div className="chart-item">
                <div className="chart-label">ViT Baseline</div>
                <div className="chart-bar-wrap"><div className="chart-bar bar-gray" style={{ '--target-width': '94.5%' }}></div></div>
                <div className="chart-val">94.5%</div>
              </div>
              <div className="chart-item">
                <div className="chart-label highlight-label">PDSCNN + ViT (Ours)</div>
                <div className="chart-bar-wrap"><div className="chart-bar bar-gradient" style={{ '--target-width': '98.5%' }}></div></div>
                <div className="chart-val highlight-val">98.5%</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="about-hero" style={{ marginTop: '2rem', marginBottom: '2rem' }}>
        <h2 className="hero-title" style={{ fontSize: '2rem' }}>Training Excellence</h2>
        <p className="hero-desc">
          Actual performance graphs and matrices generated during the training of the Vision Transformer (ViT) model from scratch.
        </p>
      </div>

      <div className="about-content-grid">
        {/* Real Confusion Matrix */}
        <div className="card anim anim-7">
          <div className="card-hd">
            <div className="card-ico ico-amber">🧮</div>
            <span className="card-title">Actual Confusion Matrix</span>
          </div>
          <div className="card-body" style={{ display: 'flex', justifyContent: 'center' }}>
            <img 
              src="/vit_confusion_matrix.png" 
              alt="Confusion Matrix" 
              style={{ maxWidth: '100%', borderRadius: '8px' }} 
            />
          </div>
        </div>

        {/* Training Curves */}
        <div className="card anim anim-8">
          <div className="card-hd">
            <div className="card-ico ico-blue">📈</div>
            <span className="card-title">Training Curves</span>
          </div>
          <div className="card-body" style={{ display: 'flex', justifyContent: 'center' }}>
            <img 
              src="/vit_training_curves.png" 
              alt="Training Curves" 
              style={{ maxWidth: '100%', borderRadius: '8px' }} 
            />
          </div>
        </div>
      </div>
    </div>
  );
}
