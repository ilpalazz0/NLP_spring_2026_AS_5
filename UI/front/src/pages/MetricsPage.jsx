import { useEffect, useState } from 'react'
import { fetchMetrics } from '../api/client'
import MetricCard from '../components/MetricCard'

function formatValue(value, digits = 4) {
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : value.toFixed(digits)
  }
  return value ?? '-'
}

function MetricsPage() {
  const [metrics, setMetrics] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    async function load() {
      try {
        const data = await fetchMetrics()
        setMetrics(data)
      } catch (err) {
        setError(err.message)
      }
    }
    load()
  }, [])

  return (
    <div className="page">
      <div className="page-header">
        <h2>Evaluation Metrics</h2>
        <p>Side-by-side comparison between the baseline LLM and the RAG system.</p>
      </div>

      {error ? <div className="error-box">{error}</div> : null}

      {metrics ? (
        <>
          <div className="metric-grid">
            <MetricCard label="Examples" value={formatValue(metrics?.num_examples)} />
            <MetricCard label="RAG EM" value={formatValue(metrics?.rag?.exact_match)} />
            <MetricCard label="RAG F1" value={formatValue(metrics?.rag?.token_f1)} />
            <MetricCard label="Baseline F1" value={formatValue(metrics?.baseline?.token_f1)} />
          </div>

          <div className="metric-grid">
            <MetricCard label="EM Delta" value={formatValue(metrics?.improvement?.exact_match_delta)} />
            <MetricCard label="F1 Delta" value={formatValue(metrics?.improvement?.token_f1_delta)} />
            <MetricCard
              label="F1 Relative Lift (%)"
              value={formatValue(metrics?.improvement?.relative_token_f1_lift_pct, 2)}
            />
            <MetricCard
              label="RAG Better Rate"
              value={formatValue((metrics?.pairwise_outcomes?.rag_better_rate ?? 0) * 100, 2)}
              subtitle="% of questions where RAG F1 > baseline"
            />
          </div>

          <div className="card">
            <h3>Retrieval Quality</h3>
            <div className="kv-list">
              <div>
                <span>Doc Hit@k</span>
                <strong>{formatValue(metrics?.retrieval_docs?.hit_at_k ?? metrics?.retrieval?.hit_at_k)}</strong>
              </div>
              <div>
                <span>Doc MRR</span>
                <strong>{formatValue(metrics?.retrieval_docs?.mrr ?? metrics?.retrieval?.mrr)}</strong>
              </div>
              <div>
                <span>Chunk Hit@k</span>
                <strong>{formatValue(metrics?.retrieval_chunks?.hit_at_k)}</strong>
              </div>
              <div>
                <span>Chunk MRR</span>
                <strong>{formatValue(metrics?.retrieval_chunks?.mrr)}</strong>
              </div>
            </div>
          </div>

          <div className="grid two-columns">
            <div className="card">
              <h3>RAG Detailed Stats</h3>
              <div className="kv-list">
                <div><span>EM Median</span><strong>{formatValue(metrics?.rag?.exact_match_stats?.median)}</strong></div>
                <div><span>EM Std</span><strong>{formatValue(metrics?.rag?.exact_match_stats?.std)}</strong></div>
                <div><span>F1 Median</span><strong>{formatValue(metrics?.rag?.token_f1_stats?.median)}</strong></div>
                <div><span>F1 Std</span><strong>{formatValue(metrics?.rag?.token_f1_stats?.std)}</strong></div>
                <div>
                  <span>Avg Answer Length (tokens)</span>
                  <strong>{formatValue(metrics?.rag?.answer_length_tokens?.mean, 2)}</strong>
                </div>
                <div>
                  <span>Empty Answer Rate</span>
                  <strong>{formatValue((metrics?.rag?.empty_answer_rate ?? 0) * 100, 2)}%</strong>
                </div>
              </div>
            </div>

            <div className="card">
              <h3>Baseline Detailed Stats</h3>
              <div className="kv-list">
                <div><span>EM Median</span><strong>{formatValue(metrics?.baseline?.exact_match_stats?.median)}</strong></div>
                <div><span>EM Std</span><strong>{formatValue(metrics?.baseline?.exact_match_stats?.std)}</strong></div>
                <div><span>F1 Median</span><strong>{formatValue(metrics?.baseline?.token_f1_stats?.median)}</strong></div>
                <div><span>F1 Std</span><strong>{formatValue(metrics?.baseline?.token_f1_stats?.std)}</strong></div>
                <div>
                  <span>Avg Answer Length (tokens)</span>
                  <strong>{formatValue(metrics?.baseline?.answer_length_tokens?.mean, 2)}</strong>
                </div>
                <div>
                  <span>Empty Answer Rate</span>
                  <strong>{formatValue((metrics?.baseline?.empty_answer_rate ?? 0) * 100, 2)}%</strong>
                </div>
              </div>
            </div>
          </div>

          <div className="card">
            <h3>Pairwise Outcome Counts</h3>
            <div className="kv-list">
              <div><span>RAG Better</span><strong>{formatValue(metrics?.pairwise_outcomes?.rag_better_count)}</strong></div>
              <div>
                <span>Baseline Better</span>
                <strong>{formatValue(metrics?.pairwise_outcomes?.baseline_better_count)}</strong>
              </div>
              <div><span>Tied</span><strong>{formatValue(metrics?.pairwise_outcomes?.tied_count)}</strong></div>
            </div>
          </div>
        </>
      ) : null}
    </div>
  )
}

export default MetricsPage
