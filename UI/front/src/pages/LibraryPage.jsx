import { useEffect, useMemo, useState } from 'react'
import { fetchLibrary } from '../api/client'

function LibraryPage() {
  const [library, setLibrary] = useState(null)
  const [error, setError] = useState('')
  const [openAuthor, setOpenAuthor] = useState('')
  const [search, setSearch] = useState('')

  useEffect(() => {
    async function load() {
      try {
        const data = await fetchLibrary()
        setLibrary(data)
        if (data.authors.length > 0) {
          setOpenAuthor(data.authors[0].author)
        }
      } catch (err) {
        setError(err.message)
      }
    }
    load()
  }, [])

  const filteredAuthors = useMemo(() => {
    if (!library) return []
    const term = search.trim().toLowerCase()
    if (!term) return library.authors
    return library.authors.filter((item) => item.author.toLowerCase().includes(term))
  }, [library, search])

  return (
    <div className="page">
      <div className="page-header">
        <h2>Library</h2>
        <p>Browse authors alphabetically and inspect the books and cleaned documents available for each one.</p>
      </div>

      {error ? <div className="error-box">{error}</div> : null}

      <div className="card">
        <div className="library-toolbar">
          <div>
            <strong>{library ? library.author_count : 0}</strong> authors
          </div>
          <input
            className="search-input"
            placeholder="Search authors"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </div>
      </div>

      <div className="library-list">
        {filteredAuthors.map((item) => {
          const isOpen = openAuthor === item.author
          return (
            <div className="card library-card" key={item.author}>
              <button
                type="button"
                className="author-toggle"
                onClick={() => setOpenAuthor(isOpen ? '' : item.author)}
              >
                <span>{item.author}</span>
                <span>{isOpen ? '−' : '+'}</span>
              </button>

              {isOpen ? (
                <div className="library-details">
                  <div className="library-stats">
                    <div><span>Books</span><strong>{item.book_count}</strong></div>
                    <div><span>Documents</span><strong>{item.document_count}</strong></div>
                  </div>

                  <div className="library-section">
                    <h3>Books</h3>
                    <ul className="tag-list">
                      {item.books.map((book) => <li key={book}>{book}</li>)}
                    </ul>
                  </div>

                  <div className="library-section">
                    <h3>10 Random Documents</h3>
                    <div className="table-like">
                      <div className="table-head docs-table-head">
                        <span>Title</span>
                        <span>Book</span>
                      </div>
                      {item.random_documents.map((doc) => (
                        <div className="table-row docs-table-row" key={doc.doc_id}>
                          <span>{doc.title}</span>
                          <span>{doc.book_title}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default LibraryPage
