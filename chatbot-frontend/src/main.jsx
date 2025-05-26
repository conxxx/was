import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import { BrowserRouter } from 'react-router-dom';
// Import global CSS if you have one, e.g., './index.css'
import './index.css' // Uncommented to load Tailwind styles
// TODO: Add ThemeProvider and ToastContainer later if needed

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
)
