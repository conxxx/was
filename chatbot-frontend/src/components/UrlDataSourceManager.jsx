import React, { useState, useEffect } from 'react';
import apiService from '../services/api';
import InputField from './fields/InputField'; // Import InputField
import { FiLink, FiLoader } from 'react-icons/fi'; // Import necessary icons

// Removed onDataSourceAdded from props as it's only needed for crawl results
function UrlDataSourceManager({ chatbotId, onAddSingleUrl, disabled }) {
  const [mode, setMode] = useState('single'); // 'single' or 'crawl' - Keep state but only show single for now
  const [singleUrl, setSingleUrl] = useState('');
  const [isLoadingSingle, setIsLoadingSingle] = useState(false);
  const [errorSingle, setErrorSingle] = useState('');
  // Crawl related state (kept for potential future use, but UI hidden)
  const [crawlStartUrl, setCrawlStartUrl] = useState('');
  const [isLoadingCrawl, setIsLoadingCrawl] = useState(false);
  const [errorCrawl, setErrorCrawl] = useState('');
  const [crawlTaskId, setCrawlTaskId] = useState(null);
  const [discoveredUrls, setDiscoveredUrls] = useState([]);
  const [selectedUrls, setSelectedUrls] = useState(new Set());
  const [urlSearchTerm, setUrlSearchTerm] = useState('');
  const [isAddingSelected, setIsAddingSelected] = useState(false);


  const handleAddSingleUrlSubmit = async (event) => { // Renamed to avoid conflict
    event.preventDefault();
    setErrorSingle('');
    if (!singleUrl.trim()) {
      setErrorSingle('URL cannot be empty.');
      return;
    }
    try {
      new URL(singleUrl); // Basic validation
    } catch (_) { // eslint-disable-line no-unused-vars
      setErrorSingle('Please enter a valid URL (e.g., https://example.com).');
      return;
    }

    setIsLoadingSingle(true);
    try {
      // Call the handler passed from DataManagementView
      await onAddSingleUrl(singleUrl);
      setSingleUrl(''); // Clear input on success
    } catch (err) {
      // Error is caught and displayed by the handler in DataManagementView
      // Or set local error if needed: setErrorSingle(err.message || 'Failed to add URL.');
      console.error("UrlDataSourceManager (Single): Error adding URL:", err); // Keep console log
      setErrorSingle(err.message || 'Failed to add URL.'); // Set local error for display
    } finally {
      setIsLoadingSingle(false);
    }
  };

  // Crawl handlers remain but are not currently used by the UI shown

  // Common button style from CreateChatbotPage reference
  const buttonClass = "flex items-center justify-center gap-2 linear rounded-md bg-brand-500 px-4 py-2 text-sm font-medium text-white transition duration-200 hover:bg-brand-600 active:bg-brand-700 dark:bg-brand-400 dark:hover:bg-brand-300 dark:active:bg-brand-200 disabled:opacity-50 disabled:cursor-not-allowed";

  return (
    // Removed the outer container div with border/background - handled by parent
    <div>
       {/* Removed Title */}
       {/* Removed Tabs */}

      {/* Only render Single URL mode based on reference image */}
      {/* Always show single mode form as tabs are removed */}
      <form onSubmit={handleAddSingleUrlSubmit} className="space-y-3">
        <InputField
          variant="outlined" // Use the standard InputField
          label="Web Page URL" // Label matches Create page
          placeholder="https://example.com/specific-page"
          id="singleUrl"
          type="url"
          value={singleUrl}
          onChange={(e) => setSingleUrl(e.target.value)}
          disabled={isLoadingSingle || disabled}
          required
          extra="mb-2" // Add margin like in CreateChatbotPage
        />
        <button type="submit" disabled={isLoadingSingle || disabled} className={buttonClass}>
          {isLoadingSingle ? <FiLoader className="animate-spin h-4 w-4" /> : <FiLink className="h-4 w-4" />}
          {isLoadingSingle ? 'Adding...' : 'Add URL'}
        </button>
        {errorSingle && <p className="text-sm text-red-500 dark:text-red-400 mt-1">{errorSingle}</p>}
      </form>


      {/* Crawl UI is hidden */}
      {/* {mode === 'crawl' && ( ... crawl UI ... )} */}

    </div>
  );
}

export default UrlDataSourceManager;