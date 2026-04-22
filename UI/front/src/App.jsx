import { NavLink, Route, Routes } from 'react-router-dom'
import HomePage from './pages/HomePage'
import AskPage from './pages/AskPage'
import DatasetPage from './pages/DatasetPage'
import MetricsPage from './pages/MetricsPage'
import LibraryPage from './pages/LibraryPage'

function App() {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <h1>Azerbaijani RAG</h1>
          <p>Course project dashboard</p>
        </div>

        <nav className="nav-links">
          <NavLink to="/" end className={({ isActive }) => isActive ? 'active-link' : ''}>
            Home
          </NavLink>
          <NavLink to="/ask" className={({ isActive }) => isActive ? 'active-link' : ''}>
            Ask Questions
          </NavLink>
          <NavLink to="/dataset" className={({ isActive }) => isActive ? 'active-link' : ''}>
            Dataset Summary
          </NavLink>
          <NavLink to="/library" className={({ isActive }) => isActive ? 'active-link' : ''}>
            Library
          </NavLink>
          <NavLink to="/metrics" className={({ isActive }) => isActive ? 'active-link' : ''}>
            Evaluation Metrics
          </NavLink>
        </nav>
      </aside>

      <main className="content">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/ask" element={<AskPage />} />
          <Route path="/dataset" element={<DatasetPage />} />
          <Route path="/library" element={<LibraryPage />} />
          <Route path="/metrics" element={<MetricsPage />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
