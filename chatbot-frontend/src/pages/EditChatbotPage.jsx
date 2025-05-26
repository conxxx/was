import React, { useState, useEffect, useCallback } from 'react'; // Import useCallback
import { useParams, Link, useNavigate } from 'react-router-dom';
import apiService, { API_URL } from '../services/api'; // Import API_URL
import axios from 'axios'; // Import axios
import WidgetCustomizationSettings from '../components/WidgetCustomizationSettings';
import DataManagementView from '../components/DataManagementView';
import DataSourceList from '../components/DataSourceList'; // Import DataSourceList
import { FiUpload, FiCopy } from 'react-icons/fi'; // Import Upload and Copy icons

function EditChatbotPage() {
  const { id } = useParams();
  const chatbotId = id;
  const navigate = useNavigate(); // Initialize navigate hook

  // State for controlling the view (tabs)
  const [currentView, setCurrentView] = useState('editSource'); // 'editSource' or 'customizeWidget'
  // State for API Key Regeneration
  const [newApiKey, setNewApiKey] = useState(null);
  const [isLoading, setIsLoading] = useState(false); // General loading state (e.g., for API key regen, details fetch)
  const [error, setError] = useState(null); // General error state
  const [successMessage, setSuccessMessage] = useState(null); // General success message
  // State for chatbot details (including sources)
  const [chatbotDetails, setChatbotDetails] = useState(null);
  const [sources, setSources] = useState({ // State specifically for source details
    added_files: [],
    added_urls: [],
    files_uploaded: [],
    original_sitemap: '',
    original_url: '',
    selected_urls: [],
    crawled_urls_added: []
  });
  // const [sourceDocumentLanguage, setSourceDocumentLanguage] = useState('en'); // REMOVED
  const [isDetailsLoading, setIsDetailsLoading] = useState(true); // Specific loading for initial details
  const [detailsError, setDetailsError] = useState(null);
  // const [isSavingLanguage, setIsSavingLanguage] = useState(false); // REMOVED
  const [actionError, setActionError] = useState(null); // Separate error for actions (like delete source)
  // const [saveSuccessMessage, setSaveSuccessMessage] = useState(null); // REMOVED (Using general successMessage)

  // Logo state variables
  const [selectedLogoFile, setSelectedLogoFile] = useState(null);
  const [logoPreviewUrl, setLogoPreviewUrl] = useState('');
  const [currentLogoUrl, setCurrentLogoUrl] = useState(null);
  const [isUploadingLogo, setIsUploadingLogo] = useState(false); // State for logo upload loading

  // Function to copy text to clipboard
  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text).then(() => {
      setSuccessMessage('API Key copied to clipboard!'); // Use general success message state
      setTimeout(() => setSuccessMessage(null), 3000);
    }).catch(err => {
      console.error('Failed to copy text: ', err);
      setError('Failed to copy API key.'); // Use general error state
    });
  };

  // --- Logo Handling Functions ---
  const fetchCurrentLogo = async () => {
    try {
      const response = await axios.get(`${API_URL}/settings/logo`);
      if (response.data && response.data.logo_filename) {
        const logoUrl = `${API_URL}/uploads/logos/${response.data.logo_filename}?t=${new Date().getTime()}`;
        setCurrentLogoUrl(logoUrl);
      } else {
        setCurrentLogoUrl(null);
      }
    } catch (error) {
      console.error('Failed to fetch current logo:', error);
      setCurrentLogoUrl(null);
    }
  };

  const handleLogoSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedLogoFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setLogoPreviewUrl(reader.result);
      };
      reader.readAsDataURL(file);
    } else {
      setSelectedLogoFile(null);
      setLogoPreviewUrl('');
      alert("Invalid file type selected. Please select an image.");
    }
  };

  const handleLogoUpload = async () => {
    if (!selectedLogoFile) {
      alert("Please select a logo file first.");
      return;
    }

    const clientId = localStorage.getItem('clientId');
    if (!clientId) {
       alert('Error: Not logged in. Cannot upload logo.');
       return;
    }

    setIsUploadingLogo(true); // Start loading
    setActionError(null); // Clear previous errors
    setSuccessMessage(null); // Clear success message
    const formData = new FormData();
    formData.append('logo', selectedLogoFile);
    formData.append('client_id', clientId);

    const uploadUrl = `${API_URL}/settings/logo`;

    try {
      const response = await axios.post(uploadUrl, formData);
      setSuccessMessage("Logo uploaded successfully!"); // Use general success
      fetchCurrentLogo(); // Refresh the displayed logo
      setSelectedLogoFile(null);
      setLogoPreviewUrl('');
      console.log("Upload response:", response);
      setTimeout(() => setSuccessMessage(null), 3000); // Clear after delay

    } catch (err) {
      console.error("Logo upload failed:", err);
      const errorMessage = err.response?.data?.message || err.message || 'Logo upload failed. Please try again.';
      setActionError(`Logo Upload Error: ${errorMessage}`); // Use actionError state
      // alert(`Error: ${errorMessage}`); // Keep alert for immediate feedback? Maybe rely on actionError display
    } finally {
      setIsUploadingLogo(false); // Stop loading
    }
  };
  // -----------------------------


  // Handler for regenerating the API key
  const handleRegenerateKey = async () => {
    const clientId = localStorage.getItem('clientId');
    if (!clientId) {
      setError('Error: Not logged in. Cannot regenerate key.');
      return;
    }

    const confirmation = window.confirm(
      'Are you sure you want to regenerate the API key? The old key will stop working immediately.'
    );

    if (confirmation) {
      setIsLoading(true); // Use general loading state
      setError(null); // Use general error state
      setSuccessMessage(null); // Use general success state
      setNewApiKey(null); // Clear previous new key

      try {
        const response = await apiService.regenerateChatbotApiKey(chatbotId, clientId);
        if (response && response.new_api_key) {
          setNewApiKey(response.new_api_key);
          setSuccessMessage('API Key regenerated successfully. Copy the new key below.');
          // Don't clear success message immediately here, it's shown with the key
        } else {
          throw new Error('Invalid response from server during key regeneration.');
        }
      } catch (err) {
        console.error('Regenerate key error:', err);
        setError(err.message || 'Failed to regenerate API key.');
      } finally {
        setIsLoading(false);
      }
    }
  };

  // Fetch chatbot details and logo - wrapped in useCallback
  const fetchDetails = useCallback(async () => {
    if (!chatbotId) {
      setDetailsError("Chatbot ID is missing.");
      setIsDetailsLoading(false);
      return;
    }
    setIsDetailsLoading(true);
    setDetailsError(null);
    setActionError(null); // Clear action errors on fetch
    setSuccessMessage(null); // Clear general success messages on fetch
    try {
      const details = await apiService.getChatbotDetails(chatbotId);
      setChatbotDetails(details);

      // Set language state - REMOVED

      // Set sources state
      if (details && details.source_details) {
        const fetchedSources = details.source_details;
        setSources({
          added_files: Array.isArray(fetchedSources.added_files) ? fetchedSources.added_files : [],
          added_urls: Array.isArray(fetchedSources.added_urls) ? fetchedSources.added_urls : [],
          files_uploaded: Array.isArray(fetchedSources.files_uploaded) ? fetchedSources.files_uploaded : [],
          original_sitemap: fetchedSources.original_sitemap || '',
          original_url: fetchedSources.original_url || '',
          selected_urls: Array.isArray(fetchedSources.selected_urls) ? fetchedSources.selected_urls : [],
          crawled_urls_added: Array.isArray(fetchedSources.crawled_urls_added) ? fetchedSources.crawled_urls_added : []
        });
      } else {
        // Reset sources if details are missing
        setSources({ added_files: [], added_urls: [], files_uploaded: [], original_sitemap: '', original_url: '', selected_urls: [], crawled_urls_added: [] });
      }

    } catch (err) {
      console.error("Failed to fetch chatbot details:", err);
      setDetailsError(err.message || "Failed to load chatbot details.");
      // Reset sources on error
      setSources({ added_files: [], added_urls: [], files_uploaded: [], original_sitemap: '', original_url: '', selected_urls: [], crawled_urls_added: [] });
    } finally {
      setIsDetailsLoading(false);
    }
  }, [chatbotId]); // Depend only on chatbotId

  useEffect(() => {
    console.log(`EditChatbotPage: Mounting for chatbotId: ${chatbotId}`);
    fetchCurrentLogo(); // Fetch logo
    fetchDetails(); // Fetch chatbot details
  }, [chatbotId, fetchDetails]); // Add fetchDetails to dependency array

  // Log view changes
  useEffect(() => {
    console.log(`EditChatbotPage: Switched to view: ${currentView} for chatbotId: ${chatbotId}`);
  }, [currentView, chatbotId]);

  // Function to handle successful save/update (e.g., after adding source) and refetch
  const handleDataSourceUpdate = useCallback(() => {
    console.log('EditChatbotPage: Data source updated, refetching details...');
    fetchDetails(); // Refetch details to update the list
  }, [fetchDetails]); // Depend on fetchDetails

  // Handler for language change - REMOVED
  // const handleLanguageChange = (newLanguage) => { ... }; // REMOVED

  // Handler for saving language setting - REMOVED
  // const handleSaveLanguage = async () => { ... }; // REMOVED

  // Handler for deleting a source
  const handleDeleteSource = async (sourceType, sourceId) => {
    setActionError(null);
    setSuccessMessage(null); // Clear previous success messages
    // Consider adding a loading state specific to deletion if needed
    console.log(`EditChatbotPage: Attempting to delete ${sourceType} source:`, sourceId);
    const confirmation = window.confirm(`Are you sure you want to delete this ${sourceType}? This action cannot be undone.`);
    if (!confirmation) return;

    // Add a temporary loading state for deletion?
    // setIsLoading(true); // Or a specific deletion loading state

    try {
      await apiService.deleteChatbotSource(chatbotId, sourceId);
      setSuccessMessage(`${sourceType} deleted successfully!`); // Use general success message
      fetchDetails(); // Refetch details to update the list
      setTimeout(() => setSuccessMessage(null), 3000); // Clear general success message
    } catch (err) {
       console.error(`EditChatbotPage: Error deleting ${sourceType} source:`, err);
       setActionError(`Failed to delete ${sourceType} source: ${err.message}`);
    } finally {
       // Stop deletion-specific loading state if added
       // setIsLoading(false);
    }
  };


  // Helper function for tab styling
  const getTabClass = (viewName) => {
    const baseClass = "py-2 px-4 rounded-t-lg font-medium transition duration-150 ease-in-out focus:outline-none";
    // Use brand color for active tab background and white text
    const activeClass = "bg-brand-500 dark:bg-brand-400 text-white border-b-2 border-brand-600 dark:border-brand-500";
    const inactiveClass = "text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-navy-700";
    return `${baseClass} ${currentView === viewName ? activeClass : inactiveClass}`;
  };

  return (
    <div className="p-4 md:p-8 flex flex-col min-h-screen"> {/* Responsive padding & ensure full height */}
      <div> {/* Top content container */}
        <Link to="/admin/dashboard" className="text-blue-600 hover:underline mb-4 inline-block">{'<'} Back to Dashboard</Link>
        <h1 className="text-2xl font-bold text-gray-800 dark:text-white mb-2">Edit Chatbot (ID: {chatbotId})</h1>

        {/* Display Created Date */}
        {isDetailsLoading && <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">Loading details...</p>}
        {detailsError && <p className="text-sm text-red-500 dark:text-red-400 mb-4">Error loading details: {detailsError}</p>}
        {chatbotDetails && chatbotDetails.created_at && (
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
            Created: {new Date(chatbotDetails.created_at).toLocaleString()}
          </p>
        )}

        {/* Tab Navigation */}
        <div className="border-b border-gray-200 dark:border-navy-700 mb-6">
          <nav className="-mb-px flex space-x-4" aria-label="Tabs">
            <button
              onClick={() => setCurrentView('editSource')}
              className={getTabClass('editSource')}
              aria-current={currentView === 'editSource' ? 'page' : undefined}
            >
              Edit Source
            </button>
            <button
              onClick={() => setCurrentView('customizeWidget')}
              className={getTabClass('customizeWidget')}
              aria-current={currentView === 'customizeWidget' ? 'page' : undefined}
            >
              Customize Widget
            </button>
          </nav>
        </div>
      </div>

      {/* Main Content Area (Grows to fill space) */}
      <div className="flex-grow">
        {/* Conditional Rendering based on currentView */}
        {!chatbotId ? (
           <p className="text-red-500 dark:text-red-400">Error: Chatbot ID not found in URL.</p>
        ) : currentView === 'editSource' ? (
          // --- Single Column Layout for Edit Source View ---
          <div className="flex flex-col gap-6"> {/* Main content column */}
            {/* Display action/save errors/success */}
            {actionError && <p className="text-red-500 dark:text-red-400 mb-4">Error: {actionError}</p>}
            {successMessage && !newApiKey && <p className="text-green-500 dark:text-green-400 mb-4">{successMessage}</p>} {/* Show general success unless new API key is displayed */}

            <DataManagementView
              chatbotId={chatbotId}
              onDataSourceAdded={handleDataSourceUpdate} // Use refetch handler
              disabled={isDetailsLoading || isLoading} // Disable forms while loading/saving
              actionError={actionError} // Pass down errors/success messages if needed within DataManagementView
              saveSuccessMessage={successMessage} // Pass general success message
            />

            {/* Existing Data Source List */}
            <DataSourceList
              sources={sources} // Pass the sources state
              onDeleteSource={handleDeleteSource} // Pass the delete handler
              disabled={isDetailsLoading || isLoading} // Disable list actions while loading/saving
            />
            {/* End Existing Data Source List */}

            {/* Company Logo and API Key moved to bottom */}

          </div>
          // --- End Layout ---

        ) : currentView === 'customizeWidget' ? (
          <WidgetCustomizationSettings chatbotId={chatbotId} onSaveSuccess={handleDataSourceUpdate} /> // Use refetch handler on save
        ) : null /* Should not happen */}
      </div>

      {/* Bottom Section (Logo and API Key) */}
      <div className="mt-8 flex flex-col md:flex-row gap-6">
         {/* Company Logo Upload Section */}
         <div className="md:w-1/2 bg-white p-3 md:p-4 rounded-lg shadow-md border border-gray-200 dark:bg-navy-800 dark:border-navy-700">
            <h2 className="text-lg font-semibold text-gray-700 dark:text-white mb-4">Company Logo</h2>
            <div className="flex items-center space-x-4">
              {/* Current Logo / Preview */}
              <div className="flex-shrink-0">
                {logoPreviewUrl ? (
                  <img src={logoPreviewUrl} alt="Logo Preview" className="w-16 h-16 object-cover border border-gray-300 dark:border-navy-600 rounded-full" /> /* Smaller logo */
                ) : currentLogoUrl ? (
                  <img src={currentLogoUrl} alt="Current Company Logo" className="w-16 h-16 object-cover border border-gray-300 dark:border-navy-600 rounded-full" /> /* Smaller logo */
                ) : (
                  <div className="w-16 h-16 bg-gray-200 dark:bg-navy-700 rounded-full flex items-center justify-center text-gray-500 dark:text-gray-400"> {/* Smaller logo */}
                    No Logo
                  </div>
                )}
              </div>
              {/* Upload Controls */}
              <div className="flex-grow">
                <label htmlFor="logoUpload" className="block mb-2 text-sm font-medium text-gray-600 dark:text-gray-300">
                  {currentLogoUrl ? 'Change Logo:' : 'Upload Logo:'}
                </label>
                <div className="flex items-center space-x-2">
                  <input
                    type="file"
                    id="logoUpload"
                    accept="image/*"
                    onChange={handleLogoSelect}
                    className="block w-full text-sm text-gray-500 dark:text-gray-400 file:mr-2 file:py-1 file:px-3 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 dark:file:bg-navy-700 file:text-blue-700 dark:file:text-blue-300 hover:file:bg-blue-100 dark:hover:file:bg-navy-600 cursor-pointer"
                    disabled={isUploadingLogo || isLoading}
                  />
                  <button
                    onClick={handleLogoUpload}
                    disabled={!selectedLogoFile || isUploadingLogo || isLoading}
                    className="p-2 bg-green-500 hover:bg-green-600 text-white rounded-full transition duration-150 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed dark:bg-green-400 dark:hover:bg-green-500"
                    aria-label="Upload selected logo"
                  >
                    {isUploadingLogo ? (
                      <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                    ) : (
                      <FiUpload className="h-5 w-5" />
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>
          {/* End Company Logo Upload Section */}

          {/* API Key Regeneration Section */}
          <div className="md:w-1/2 p-3 md:p-4 border border-gray-300 dark:border-navy-700 rounded-lg bg-gray-50 dark:bg-navy-700"> {/* Reduced padding */}
            <h2 className="text-lg font-semibold text-gray-700 dark:text-white mb-2">API Key Management</h2>
            <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">Regenerate the API key if you suspect it has been compromised. The old key will stop working immediately.</p>

            {error && <p className="text-red-500 dark:text-red-400 mb-2">Error: {error}</p>}
            {successMessage && !newApiKey && <p className="text-green-500 dark:text-green-400 mb-2">{successMessage}</p>} {/* Show general success unless new API key is displayed */}

            <button
              onClick={handleRegenerateKey}
              disabled={isLoading}
              className="bg-orange-500 hover:bg-orange-600 text-white font-semibold py-2 px-4 rounded transition duration-150 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed dark:bg-orange-400 dark:hover:bg-orange-500"
            >
              {isLoading ? 'Regenerating...' : 'Regenerate API Key'}
            </button>

            {newApiKey && (
              <div className="mt-4 p-3 border border-gray-200 dark:border-navy-600 bg-white dark:bg-navy-800 rounded"> {/* Reduced padding */}
                <p className="text-green-600 dark:text-green-400 font-bold mb-2">{successMessage}</p> {/* Show success message here */}
                <p className="font-semibold text-gray-700 dark:text-white mb-1">New API Key:</p>
                <div className="flex items-center space-x-2">
                  <pre className="flex-grow whitespace-pre-wrap break-all bg-gray-100 dark:bg-navy-900 p-2 border border-dashed border-gray-300 dark:border-navy-600 rounded text-sm text-gray-800 dark:text-gray-200">
                    {newApiKey}
                  </pre>
                  <button
                    onClick={() => copyToClipboard(newApiKey)}
                    className="p-2 bg-blue-500 hover:bg-blue-600 text-white rounded-full transition duration-150 ease-in-out dark:bg-blue-400 dark:hover:bg-blue-500"
                    aria-label="Copy API Key to Clipboard"
                  >
                    <FiCopy className="h-4 w-4" />
                  </button>
                </div>
                <p className="text-orange-600 dark:text-orange-400 font-bold mt-2">
                  Copy this key now! It will not be shown again.
                </p>
              </div>
            )}
          </div>
          {/* End API Key Regeneration Section */}
      </div> {/* End Bottom Section */}

    </div> // End main page container
  );
}

export default EditChatbotPage;
