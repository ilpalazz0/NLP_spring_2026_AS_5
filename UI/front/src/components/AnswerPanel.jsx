function AnswerPanel({ title, content }) {
  return (
    <div className="card answer-panel">
      <h3>{title}</h3>
      <div className="answer-content">
        {content || 'No answer yet.'}
      </div>
    </div>
  )
}

export default AnswerPanel
