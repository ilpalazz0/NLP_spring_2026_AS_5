import { useEffect, useState } from 'react'
import { fetchDatasetSummary } from '../api/client'
import MetricCard from '../components/MetricCard'

function DocumentDetails({ title, document }) {
  if (!document) return null

  return (
    <div className="card">
      <h3>{title}</h3>
      <div className="document-meta-grid">
        <div><span>Title</span><strong>{document.title}</strong></div>
        <div><span>Author</span><strong>{document.author}</strong></div>
        <div><span>Book</span><strong>{document.book_title}</strong></div>
        <div><span>Characters</span><strong>{document.char_count}</strong></div>
      </div>
      <div className="document-content-box">{document.text}</div>
    </div>
  )
}

function DatasetPage() {
  const [summary, setSummary] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    async function load() {
      try {
        const data = await fetchDatasetSummary()
        setSummary(data)
      } catch (err) {
        setError(err.message)
      }
    }
    load()
  }, [])

  return (
    <div className="page">
      <div className="page-header">
        <h2>Dataset Summary</h2>
        <p>Overview of the imported dataset after cleaning and chunking.</p>
      </div>

      {error ? <div className="error-box">{error}</div> : null}

      {summary ? (
        <>
          <div className="metric-grid dataset-metric-grid">
            <MetricCard label="Documents" value={summary.document_count} />
            <MetricCard label="Chunks" value={summary.chunk_count} />
            <MetricCard label="Authors" value={summary.author_count} />
            <MetricCard label="Books" value={summary.book_count} />
            <MetricCard label="Avg Chunk Tokens" value={summary.avg_chunk_tokens} />
            <MetricCard label="Avg Document Characters" value={summary.avg_document_characters} />
            <MetricCard label="Total Characters" value={summary.total_characters} />
            <MetricCard label="Shortest / Longest" value={`${summary.min_document_characters} / ${summary.max_document_characters}`} />
          </div>

          <div className="grid two-columns">
            <div className="card">
              <h3>Chunk Distribution</h3>
              <div className="kv-list">
                <div><span>Minimum chunk tokens</span><strong>{summary.min_chunk_tokens}</strong></div>
                <div><span>Maximum chunk tokens</span><strong>{summary.max_chunk_tokens}</strong></div>
              </div>
            </div>

            <div className="card">
              <h3>Most Frequent Authors</h3>
              <div className="kv-list">
                {summary.top_authors.map(([author, count]) => (
                  <div key={author}><span>{author}</span><strong>{count}</strong></div>
                ))}
              </div>
            </div>
          </div>

          <div className="card">
            <h3>Sample Documents</h3>
            <div className="table-like">
              <div className="table-head sample-table-head">
                <span>Document ID</span>
                <span>Title</span>
                <span>Author</span>
                <span>Characters</span>
              </div>
              {summary.sample_documents.map((doc) => (
                <div className="table-row sample-table-row" key={doc.doc_id}>
                  <span>{doc.doc_id}</span>
                  <span>{doc.title}</span>
                  <span>{doc.author}</span>
                  <span>{doc.char_count}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="grid two-columns">
            <DocumentDetails title="Shortest Document" document={summary.shortest_document} />
            <DocumentDetails title="Longest Document" document={summary.longest_document} />
          </div>
        </>
      ) : null}
    </div>
  )
}

export default DatasetPage
