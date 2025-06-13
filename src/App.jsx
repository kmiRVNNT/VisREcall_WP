import { useState } from 'react'

import './App.css'

function App() {
  const [searchText, setSearchText] = useState('')
  const [currentPage, setCurrentPage] = useState({ title: '', url: '' })
  const [searchResults, setSearchResults] = useState('')

  // Handle search input
  const handleSearch = (e) => {
    const text = e.target.value
    setSearchText(text)
    
    if (text.length > 2) {
      setSearchResults(`Searching for: ${text}`)
      console.log('Searching for:', text)
    } else {
      setSearchResults('')
    }
  }

  // Handle save current page
  const handleSavePage = () => {
    // For now, we'll just show a success message
    // Later we can add actual saving functionality
    setSearchResults('Page saved successfully!')
  }

  // Handle open new tab
  const handleOpenNewTab = () => {
    window.open('https://www.google.com', '_blank')
  }

  return (
    <div className="app">
      <h1>VisREcall</h1>
      
      <div className="search-container">
        <input 
          type="text" 
          className="search-box"
          value={searchText}
          onChange={handleSearch}
          placeholder="Search your saved items..."
        />
      </div>

      <div className="current-page">
        <h3>Current Page</h3>
        <p>{currentPage.title || 'No page selected'}</p>
        <p>{currentPage.url || 'No URL available'}</p>
      </div>

      <div className="results">
        {searchResults}
      </div>

      <div className="buttons">
        <button onClick={handleSavePage}>
          Save Current Page
        </button>
        <button onClick={handleOpenNewTab}>
          Open New Tab
        </button>
      </div>
    </div>
  )
}

export default App 