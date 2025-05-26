import React, { useState } from 'react';
import apiService from '../services/api';
import UrlDataSourceManager from './UrlDataSourceManager';
import AddFileForm from './AddFileForm';
import { FiLink, FiFileText } from 'react-icons/fi'; // Import icons for checkboxes

function DataManagementView({
  chatbotId,
  onDataSourceAdded,
  disabled,
  actionError,
  saveSuccessMessage
}) {
  const [isMutating, setIsMutating] = useState(false);
  // State for controlling visibility based on checkboxes (default to visible in edit mode)
  const [showUrlInput, setShowUrlInput] = useState(true);
  const [showFileInput, setShowFileInput] = useState(true);

  // Handler for adding files (calls parent handler)
  const handleAddFiles = async (files) => {
    setIsMutating(true);
    console.log('DataManagementView: Attempting to add files:', files.map(f => f.name).join(', '));
    try {
      await apiService.addChatbotFiles(chatbotId, files);
      if (onDataSourceAdded) {
        onDataSourceAdded(); // Notify parent of success
      }
      // Clear local state if needed, e.g., if AddFileForm managed its own list visually
    } catch (err) {
       console.error('DataManagementView: Error adding files:', err);
       // Error display is handled by the parent component
    } finally {
       setIsMutating(false);
    }
  };

  // Handler for adding a single URL (passed down to UrlDataSourceManager)
   const handleAddSingleUrl = async (url) => {
    setIsMutating(true);
    console.log('DataManagementView: Attempting to add single URL:', url);
    try {
      await apiService.addChatbotUrl(chatbotId, url);
      if (onDataSourceAdded) {
        onDataSourceAdded(); // Notify parent of success
      }
      // UrlDataSourceManager should clear its own input on success handled via its submit handler
    } catch (err) {
      console.error('DataManagementView: Error adding single URL:', err);
      // Error display is handled by the parent component
      throw err; // Allow UrlDataSourceManager to catch and display its local error
    } finally {
      setIsMutating(false);
    }
  };


  return (
    // Main Card structure from CreateChatbotPage
    <div className="p-6 bg-white dark:bg-navy-800 rounded-lg shadow-md border border-gray-200 dark:border-navy-700">
      {/* Title and Correct Subtitle */}
      <h2 className="text-xl font-semibold text-navy-700 dark:text-white mb-1">Data Sources</h2>
      {/* Corrected subtitle to match Create page */}
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">Select one or more sources to train your chatbot.</p>

      {/* Display errors/success messages passed from parent */}
      {actionError && <p className="text-red-500 dark:text-red-400 mb-4">Error: {actionError}</p>}
      {saveSuccessMessage && <p className="text-green-500 dark:text-green-400 mb-4">{saveSuccessMessage}</p>}

      {/* Checkbox Selection */}
      <div className="flex flex-wrap gap-4 mb-6">
        <label htmlFor="showUrlInput" className="flex items-center gap-2 cursor-pointer text-sm font-medium text-navy-700 dark:text-white">
          <input
            type="checkbox"
            id="showUrlInput"
            checked={showUrlInput}
            onChange={(e) => setShowUrlInput(e.target.checked)}
            className="form-checkbox h-5 w-5 text-brand-500 rounded border-gray-300 focus:ring-brand-500 dark:border-gray-600 dark:bg-navy-700 dark:focus:ring-brand-400 dark:checked:bg-brand-400"
            disabled={disabled}
          />
          <FiLink className="h-5 w-5" /> Website URL
        </label>
        <label htmlFor="showFileInput" className="flex items-center gap-2 cursor-pointer text-sm font-medium text-navy-700 dark:text-white">
          <input
            type="checkbox"
            id="showFileInput"
            checked={showFileInput}
            onChange={(e) => setShowFileInput(e.target.checked)}
            className="form-checkbox h-5 w-5 text-brand-500 rounded border-gray-300 focus:ring-brand-500 dark:border-gray-600 dark:bg-navy-700 dark:focus:ring-brand-400 dark:checked:bg-brand-400"
            disabled={disabled}
          />
          <FiFileText className="h-5 w-5" /> File Upload
        </label>
      </div>

      {/* Conditional Rendering of Input Components within styled containers */}
      <div className="grid grid-cols-1 gap-6">
        {showUrlInput && (
          // Apply the bordered container style HERE
          <div className="border-l-4 border-brand-500 dark:border-brand-400 pl-4 py-4 bg-gray-50 dark:bg-navy-700 rounded-r-lg">
            {/* UrlDataSourceManager no longer has its own container/title */}
            <UrlDataSourceManager
              chatbotId={chatbotId}
              onAddSingleUrl={handleAddSingleUrl} // Pass the specific handler
              disabled={disabled || isMutating}
              // onDataSourceAdded is not needed here as crawl is hidden
            />
          </div>
        )}

        {showFileInput && (
           // Apply the bordered container style HERE
          <div className="border-l-4 border-brand-500 dark:border-brand-400 pl-4 py-4 bg-gray-50 dark:bg-navy-700 rounded-r-lg">
             {/* AddFileForm no longer has its own container/title */}
            <AddFileForm
               onAddFiles={handleAddFiles}
               disabled={disabled || isMutating}
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default DataManagementView;