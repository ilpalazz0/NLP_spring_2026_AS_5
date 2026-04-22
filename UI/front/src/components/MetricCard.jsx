function MetricCard({ label, value, subtitle, valueClass }) {
  return (
    <div className="card metric-card">
      <div className="metric-label">{label}</div>
      <div className={`metric-value ${valueClass || ''}`}>{value ?? '-'}</div>
      {subtitle ? <div className="metric-subtitle">{subtitle}</div> : null}
    </div>
  )
}

export default MetricCard
