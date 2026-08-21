import { useState, useRef } from 'react'
import './index.css'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = (selectedFile) => {
    if (!selectedFile.type.startsWith('image/')) {
      setError('Please upload an image file.')
      return
    }
    setFile(selectedFile)
    setPreview(URL.createObjectURL(selectedFile))
    setError(null)
    setPrediction(null)
  }

  const handleUpload = async () => {
    if (!file) return

    setIsLoading(true)
    setError(null)
    setPrediction(null)

    const formData = new FormData()
    formData.append('image', file)

    try {
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Failed to upload image')
      }

      setPrediction(data)
    } catch (err) {
      setError(err.message || 'Something went wrong')
    } finally {
      setIsLoading(false)
    }
  }

  const resetState = () => {
    setFile(null)
    setPreview(null)
    setPrediction(null)
    setError(null)
  }

  return (
    <div className="app-container">
      <div className="header">
        <h1>NeuroSight AI</h1>
        <p>Advanced Brain Tumor Classification</p>
      </div>

      <div className="upload-card">
        {!preview ? (
          <div
            className={`upload-area ${isDragging ? 'drag-active' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current.click()}
          >
            <div className="upload-icon">🧠</div>
            <p className="upload-text">Drag & drop your MRI scan here</p>
            <p className="upload-subtext">or click to browse files</p>
            <input
              type="file"
              className="file-input"
              ref={fileInputRef}
              onChange={handleChange}
              accept="image/*"
            />
          </div>
        ) : (
          <div className="preview-area" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            {!prediction && (
              <img 
                src={preview} 
                alt="MRI Preview" 
                className="main-preview-image"
              />
            )}
            
            {!prediction && !isLoading && (
              <div style={{ display: 'flex', gap: '1rem' }}>
                <button className="btn" onClick={resetState} style={{ background: 'transparent', border: '1px solid #475569' }}>
                  Cancel
                </button>
                <button className="btn" onClick={handleUpload}>
                  Analyze Scan
                </button>
              </div>
            )}
            
            {isLoading && (
              <div style={{ marginTop: '1rem', color: '#38bdf8' }}>
                Running PDSCNN + ViT Ensemble and generating explanations... ⏳
                <p style={{ fontSize: '0.8rem', color: '#94a3b8' }}>(This may take a few moments)</p>
              </div>
            )}
          </div>
        )}

        {prediction && (
          <div className="results-container">
            <div className="prediction-result">
              <h3>Prediction: <span style={{color: '#fff'}}>{prediction.prediction.toUpperCase()}</span></h3>
              <p>Confidence: {(prediction.confidence * 100).toFixed(1)}%</p>
            </div>

            <div className="visualizations-grid">
              <div className="viz-card">
                <h4>Original</h4>
                <img src={`http://localhost:5000/${prediction.original}`} alt="Original" />
              </div>
              
              {prediction.gradcam && (
                <div className="viz-card">
                  <h4>Grad-CAM</h4>
                  <img src={`http://localhost:5000/${prediction.gradcam}`} alt="Grad-CAM" />
                </div>
              )}
              
              {prediction.shap && (
                <div className="viz-card">
                  <h4>SHAP Values</h4>
                  <img src={`http://localhost:5000/${prediction.shap}`} alt="SHAP" />
                </div>
              )}
            </div>

            <button className="btn" onClick={resetState} style={{ marginTop: '2rem', width: '100%' }}>
              Analyze Another Scan
            </button>
          </div>
        )}

        {error && (
          <div className="prediction-result error-result">
            <h3>Error</h3>
            <p>{error}</p>
            <button className="btn" onClick={resetState} style={{ marginTop: '1rem', width: '100%', background: '#ef4444' }}>
              Try Again
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
