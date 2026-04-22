import { useState } from 'react'
import { askQuestion } from '../api/client'
import AnswerPanel from '../components/AnswerPanel'

const SAMPLE_QUESTIONS = [
  'III Böyük Mənuçöhr haqqında bu korpus nə deyir?',
  'Koroğlu dastanı ilə Aşıq Qərib dastanı necə fərqləndirilir?',
  'Məhəmməd Füzuli kimdir və Azərbaycan ədəbiyyatında hansı rola malikdir?',
  'Füzulinin "Leyli və Məcnun" əsəri kim tərəfindən yazılıb və hansı dövrə aid edilir?',
  'Nizami Gəncəvi kimdir və onun Azərbaycan ədəbiyyatındakı yeri necə izah olunur?',
  'Nizami Gəncəvinin "Xəmsə"si hansı əsərlərdən ibarətdir?',
  '"Yeddi gözəl" əsəri kim tərəfindən yazılıb və nə vaxt qələmə alınıb?',
  '"Xosrov və Şirin" əsərinin müəllifi kimdir və bu əsər hansı mövzunu işləyir?',
  'Şah İsmayıl Xətai həm hökmdar, həm də şair kimi necə təqdim olunur?',
  'Mirzə Fətəli Axundzadə Azərbaycan ədəbiyyatında hansı yeniliklərlə tanınır?',
  '"Hophopnamə" əsərinin müəllifi kimdir və bu əsər hansı ictimai məsələləri qabardır?',
  '"Dədə Qorqud" abidəsi nədir, hansı dövrə aid edilir və Azərbaycan ədəbiyyatında niyə vacib sayılır?'
]

function pickFirstNonEmpty(...values) {
  for (const value of values) {
    if (typeof value === 'string' && value.trim()) {
      return value.trim()
    }
  }
  return ''
}

function extractDisplayText(value) {
  if (typeof value !== 'string') {
    return ''
  }
  const text = value.trim()
  if (!text) {
    return ''
  }
  if (text.startsWith('{')) {
    try {
      const parsed = JSON.parse(text)
      if (parsed && typeof parsed.answer === 'string' && parsed.answer.trim()) {
        return parsed.answer.trim()
      }
    } catch {
      const strictMatch = text.match(/"answer"\s*:\s*"([\s\S]*?)"\s*,\s*"citations"/i)
      if (strictMatch?.[1]) {
        return strictMatch[1]
          .replace(/\\"/g, '"')
          .replace(/\\n/g, '\n')
          .replace(/\\t/g, '\t')
          .trim()
      }
      const fallbackMatch = text.match(/"answer"\s*:\s*"([\s\S]*)/i)
      if (fallbackMatch?.[1]) {
        return fallbackMatch[1]
          .replace(/\\"/g, '"')
          .replace(/\\n/g, '\n')
          .replace(/\\t/g, '\t')
          .trim()
      }
      return text
    }
  }
  return text
}

function normalizeAskResponse(response) {
  const ragAnswerRaw = pickFirstNonEmpty(
    response?.rag_answer,
    response?.with_rag_answer,
    response?.answer_with_retrieval,
    response?.retrieval_answer,
    response?.rag,
    response?.answers?.with_retrieval
  )

  const baselineAnswerRaw = pickFirstNonEmpty(
    response?.baseline_answer,
    response?.plain_answer,
    response?.without_rag_answer,
    response?.answer_without_retrieval,
    response?.direct_answer,
    response?.no_rag,
    response?.answers?.without_retrieval
  )

  const ragAnswer = extractDisplayText(ragAnswerRaw)
  const baselineAnswer = extractDisplayText(baselineAnswerRaw)

  return {
    ...response,
    rag_answer: ragAnswer || response?.rag_error || 'No answer yet.',
    baseline_answer: baselineAnswer || response?.plain_error || 'No answer yet.',
    rag_citations: Array.isArray(response?.rag_output?.citations) ? response.rag_output.citations : [],
    baseline_citations: Array.isArray(response?.baseline_output?.citations) ? response.baseline_output.citations : [],
  }
}

function AskPage() {
  const [question, setQuestion] = useState('')
  const [topK, setTopK] = useState(10)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function handleSubmit(event) {
    event.preventDefault()
    setError('')
    setLoading(true)
    setResult(null)

    try {
      const response = await askQuestion(question, Number(topK))
      setResult(normalizeAskResponse(response))
    } catch (err) {
      setError(err?.message || 'Request failed.')
    } finally {
      setLoading(false)
    }
  }

  function handleSampleClick(sampleQuestion) {
    setQuestion(sampleQuestion)
  }

  return (
    <div className="page ask-page-layout">
      <div className="ask-main-column">
        <div className="page-header">
          <h2>Ask Questions</h2>
          <p>Compare generation with retrieval against the same model without retrieval.</p>
        </div>

        <form className="card ask-form" onSubmit={handleSubmit}>
          <label className="field-label">Question</label>
          <textarea
            className="question-input"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Sualınızı Azərbaycan dilində yazın..."
          />

          <div className="form-row">
            <div>
              <label className="field-label">Top-k</label>
              <input
                className="number-input"
                type="number"
                min="1"
                max="10"
                value={topK}
                onChange={(event) => setTopK(event.target.value)}
              />
            </div>

            <button
              className="primary-button"
              type="submit"
              disabled={loading || !question.trim()}
            >
              {loading ? 'Generating...' : 'Ask'}
            </button>
          </div>
        </form>

        {error ? <div className="error-box">{error}</div> : null}

        <div className="grid answer-grid">
          <AnswerPanel title="LLM with Retrieval" content={result?.rag_answer} />
          <AnswerPanel title="LLM without Retrieval" content={result?.baseline_answer} />
        </div>

        {result?.rag_citations?.length ? (
          <div className="card">
            <h3>RAG Citations</h3>
            <div className="retrieved-list">
              {result.rag_citations.map((citation, index) => (
                <div key={`${citation.chunk_id || citation.doc_id || 'cite'}-${index}`} className="retrieved-item">
                  <div className="retrieved-header">
                    <strong>{citation.title || 'Untitled source'}</strong>
                    <span>{citation.section_title || 'Section N/A'}</span>
                  </div>
                  <p>{citation.url || 'URL not provided'}</p>
                </div>
              ))}
            </div>
          </div>
        ) : null}

        {result?.retrieved_chunks?.length ? (
          <div className="card">
            <h3>Retrieved Chunks</h3>
            <div className="retrieved-list">
              {result.retrieved_chunks.map((chunk, index) => (
                <div
                  key={chunk.chunk_id || `${chunk.doc_id || 'doc'}-${chunk.chunk_index || index}`}
                  className="retrieved-item"
                >
                  <div className="retrieved-header">
                    <strong>{chunk.title || 'Untitled chunk'}</strong>
                    <span>{chunk.doc_id || 'Unknown document'}</span>
                  </div>
                  <p>{chunk.text}</p>
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </div>

      <aside className="card sample-questions-panel">
        <div className="sample-questions-header">
          <h3>Sample Questions</h3>
          <p>Click any question to copy it into the input field.</p>
        </div>

        <div className="sample-question-list">
          {SAMPLE_QUESTIONS.map((sampleQuestion, index) => (
            <button
              key={sampleQuestion}
              type="button"
              className="sample-question-button"
              onClick={() => handleSampleClick(sampleQuestion)}
            >
              <span className="sample-question-index">{index + 1}.</span>
              <span>{sampleQuestion}</span>
            </button>
          ))}
        </div>
      </aside>
    </div>
  )
}

export default AskPage