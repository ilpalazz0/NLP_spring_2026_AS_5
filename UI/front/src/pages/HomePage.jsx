import { useEffect, useState } from 'react'
import { fetchManifest, fetchMetrics } from '../api/client'
import MetricCard from '../components/MetricCard'

function HomePage() {
  const [manifest, setManifest] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    async function load() {
      try {
        const [manifestData, metricsData] = await Promise.all([
          fetchManifest(),
          fetchMetrics()
        ])
        setManifest(manifestData)
        setMetrics(metricsData)
      } catch (err) {
        setError(err.message)
      }
    }
    load()
  }, [])

  return (
    <div className="page">
      <div className="page-header">
        <h2>Overview</h2>
        <p>Offline-built RAG artifacts, model configuration, and quick performance summary.</p>
      </div>

      {error ? <div className="error-box">{error}</div> : null}

      {manifest ? (
        <div className="grid two-columns">
          <div className="card">
            <h3>Build Manifest</h3>
            <div className="kv-list">
              <div><span>Dataset</span><strong>{manifest.manifest.dataset_name}</strong></div>
              <div><span>Documents</span><strong>{manifest.manifest.document_count}</strong></div>
              <div><span>Chunks</span><strong>{manifest.manifest.chunk_count}</strong></div>
              <div><span>Embedding Model</span><strong>{manifest.manifest.embedding_model_name}</strong></div>
              <div><span>Generation Provider</span><strong>{manifest.generation_provider}</strong></div>
            </div>
          </div>

          <div className="card">
            <h3>Runtime</h3>
            <div className="kv-list">
              <div><span>Device</span><strong>{manifest.device.device}</strong></div>
              <div><span>CUDA Available</span><strong>{String(manifest.device.cuda_available)}</strong></div>
              <div><span>GPU</span><strong>{manifest.device.gpu_name || 'CPU only'}</strong></div>
            </div>
          </div>
        </div>
      ) : null}

      {metrics && metrics.rag ? (
        <div className="metric-grid">
          <MetricCard label="RAG EM" value={metrics.rag.exact_match} />
          <MetricCard label="RAG F1" value={metrics.rag.token_f1} />
          <MetricCard label="Baseline EM" value={metrics.baseline.exact_match} />
          <MetricCard label="Baseline F1" value={metrics.baseline.token_f1} />
        </div>
      ) : null}
    </div>
  )
}

export default HomePage
