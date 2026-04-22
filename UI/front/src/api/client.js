const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  })

  const contentType = response.headers.get('content-type') || ''
  const isJson = contentType.includes('application/json')

  let data = null
  if (isJson) {
    try {
      data = await response.json()
    } catch {
      data = null
    }
  } else {
    try {
      const text = await response.text()
      data = { detail: text }
    } catch {
      data = null
    }
  }

  if (!response.ok) {
    const detail =
      data?.detail ||
      data?.message ||
      `Request failed with status ${response.status}`
    throw new Error(detail)
  }

  return data
}

export { API_BASE_URL }

export function fetchHealth() {
  return request('/health')
}

export function fetchManifest() {
  return request('/manifest')
}

export function fetchDatasetSummary() {
  return request('/dataset/summary')
}

export function fetchMetrics() {
  return request('/metrics')
}

export function fetchLibrary() {
  return request('/library')
}

export function askQuestion(question, topK = 4) {
  return request('/ask', {
    method: 'POST',
    body: JSON.stringify({
      question,
      top_k: topK,
    }),
  })
}