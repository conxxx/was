import React, { useState, useEffect } from 'react';
import { useNavigate, useParams, Link } from 'react-router-dom'; // Import useParams and Link
import Modal from 'react-modal'; // Import Modal
import apiService from '../services/api';
import Card from '../components/card/Card'; // Import Card component
import InputField from '../components/fields/InputField'; // Import InputField component
import { FiLink, FiFileText, FiUpload, FiTrash2, FiCheckSquare, FiSquare, FiFilter, FiArrowLeft, FiLoader, FiAlertCircle, FiCheckCircle, FiXCircle, FiCopy } from 'react-icons/fi'; // Import icons

// TODO: Import toast for notifications

// Make sure to bind modal to your appElement (http://reactcommunity.org/react-modal/accessibility/)
Modal.setAppElement('#root'); // Assuming your root element has id 'root'


function CreateChatbotPage() {
  const [chatbotName, setChatbotName] = useState('');
  // State for enabling/disabling source types
  const [useUrlSource, setUseUrlSource] = useState(true); // Default to URL enabled
  const [useSitemapSource, setUseSitemapSource] = useState(false);
  const [useFileSource, setUseFileSource] = useState(false);
  // State for source values
  const [urlValue, setUrlValue] = useState('');
  const [sitemapValue, setSitemapValue] = useState('');
  const [files, setFiles] = useState([]); // Array for File objects or { name: string, isExisting: true }
  const [filesToRemove, setFilesToRemove] = useState([]); // Track names of existing files to remove
  // State for discovery process
  const [discoveredUrls, setDiscoveredUrls] = useState([]);
  const [selectedUrls, setSelectedUrls] = useState(new Set());
  const [discoveryTaskId, setDiscoveryTaskId] = useState(null);
  const [isDiscovering, setIsDiscovering] = useState(false);
  const [discoveryError, setDiscoveryError] = useState('');
  const [urlSearchTerm, setUrlSearchTerm] = useState(''); // State for URL filter
  const [isLoading, setIsLoading] = useState(false); // For main form submission/loading details
  const [error, setError] = useState(''); // For main form submission/loading error
  const navigate = useNavigate();
  const { id: editChatbotId } = useParams(); // Get ID from URL for edit mode
  const isEditMode = Boolean(editChatbotId);
  const [isApiKeyModalOpen, setIsApiKeyModalOpen] = useState(false); // State for API key modal
  const [newApiKey, setNewApiKey] = useState(''); // State to hold the new API key

  // Fetch chatbot details if in edit mode
  useEffect(() => {
    console.log("Edit mode detected:", isEditMode, "Chatbot ID:", editChatbotId); 
    if (isEditMode && editChatbotId) { 
      setIsLoading(true);
      setError('');
      // No need to get clientId here, apiService.getChatbotDetails handles it
      console.log(`Fetching details for chatbot ID: ${editChatbotId}`); 
      apiService.getChatbotDetails(editChatbotId) // clientId retrieved inside service
        .then(data => {
          console.log("Fetched chatbot details:", data); 
          setChatbotName(data.name || '');
          const details = data.source_details || {};
          const types = data.source_type ? data.source_type.split('+') : [];
          setUseUrlSource(!!details.original_url); 
          setUrlValue(details.original_url || '');
          setUseSitemapSource(!!details.original_sitemap);
          setSitemapValue(details.original_sitemap || '');
          setUseFileSource(types.includes('Files') || details.files_uploaded?.length > 0);
          setFiles(details.files_uploaded?.map(name => ({ name: name, isExisting: true })) || []); 
          setSelectedUrls(new Set(details.selected_urls || []));
          setDiscoveredUrls(details.selected_urls || []); 
        })
        .catch(err => {
          console.error("Error fetching chatbot details:", err); 
          setError(err.message || 'Failed to load chatbot details for editing.');
          if (err.message.includes('Unauthorized') || err.message.includes('Not logged in')) {
             apiService.logout(); // Logout if unauthorized or not logged in
             navigate('/login'); 
          }
        })
        .finally(() => {
          setIsLoading(false);
        });
    }
  }, [isEditMode, editChatbotId, navigate]); 


  const handleFileChange = (event) => {
    console.log("handleFileChange triggered"); 
    const selectedFiles = event.target.files;
    console.log("Selected files:", selectedFiles); 
    if (!selectedFiles || selectedFiles.length === 0) {
      console.log("No files selected or FileList empty.");
      return; 
    }
    const filesArray = Array.from(selectedFiles);
    console.log("Files converted to array:", filesArray); 
    setFiles(prevFiles => {
      const updatedFiles = [...prevFiles, ...filesArray];
      console.log("Updating files state to:", updatedFiles); 
      return updatedFiles;
    });
    event.target.value = null; 
  };

  const handleRemoveFile = (fileToRemove) => {
    if (fileToRemove.isExisting) {
      setFilesToRemove(prev => [...prev, fileToRemove.name]);
    }
    setFiles(prevFiles => prevFiles.filter(file => file !== fileToRemove));
  };

  const handleDiscoverLinks = async (discoveryType) => { 
    const sourceValue = discoveryType === 'url' ? urlValue : sitemapValue;
    if (!sourceValue) {
      alert(`Please enter a ${discoveryType === 'url' ? 'Website URL' : 'Sitemap URL'}.`);
      return;
    }
    setDiscoveryError('');
    setIsDiscovering(true);
    setDiscoveredUrls([]); 
    setSelectedUrls(new Set());
    setDiscoveryTaskId(null);
    try {
      const discoveryData = { source_url: sourceValue, source_type: discoveryType };
      const response = await apiService.discoverLinks(discoveryData);
      if (response && response.task_id) {
        setDiscoveryTaskId(response.task_id);
        console.log("Discovery task started:", response.task_id); 
        alert("Link discovery started. This may take a moment."); 
      } else {
        throw new Error("Failed to start discovery task.");
      }
    } catch (err) {
      console.error("Discovery error:", err);
      setDiscoveryError(err.message || `Failed to start ${discoveryType} discovery.`); 
      setIsDiscovering(false); 
    } 
  };

  // Effect for polling discovery results
  useEffect(() => {
    let intervalId = null;
    const pollTimeout = 180000;
    const pollInterval = 3000;
    let timeElapsed = 0;
    let isPollingStopped = false; // Flag to prevent race conditions

    const pollResults = async () => {
      if (!discoveryTaskId || isPollingStopped) return; // Check flag

      try {
        const response = await apiService.getDiscoveryResults(discoveryTaskId);

        if (isPollingStopped) return; // Check flag again after await

        if (response.status === 'SUCCESS') {
          // Access the nested result array from the API response
          const urls = response.result?.result || []; // Use optional chaining and default to empty array
          setDiscoveredUrls(urls);
          setSelectedUrls(new Set(urls));
          // Now stop polling and update status
          isPollingStopped = true; // Set flag
          clearInterval(intervalId); // Clear interval
          setIsDiscovering(false);
          setDiscoveryError('');
        } else if (response.status === 'FAILURE') {
          isPollingStopped = true; // Set flag
          clearInterval(intervalId); // Clear interval
          setDiscoveryError(response.error || 'Discovery task failed.');
          setIsDiscovering(false);
        } else if (timeElapsed >= pollTimeout) {
          isPollingStopped = true; // Set flag
          clearInterval(intervalId); // Clear interval
          setDiscoveryError('Discovery task timed out.');
          setIsDiscovering(false);
        }
        // If status is PROGRESS, continue polling (interval is still active)
      } catch (err) {
        if (isPollingStopped) return; // Check flag after await in catch
        isPollingStopped = true; // Set flag
        clearInterval(intervalId); // Clear interval
        console.error("Polling error:", err);
        setDiscoveryError(err.message || 'Error fetching discovery results.');
        setIsDiscovering(false);
      }
      timeElapsed += pollInterval;
    };

    if (isDiscovering && discoveryTaskId) {
      pollResults(); // Initial poll
      intervalId = setInterval(pollResults, pollInterval);
    }

    // Cleanup function
    return () => {
      isPollingStopped = true; // Ensure polling stops on cleanup
      if (intervalId) clearInterval(intervalId);
    };
  }, [isDiscovering, discoveryTaskId]); // Dependencies remain the same

  const handleUrlSelectionChange = (url) => {
    setSelectedUrls(prevSelected => {
      const newSelected = new Set(prevSelected);
      if (newSelected.has(url)) newSelected.delete(url);
      else newSelected.add(url);
      return newSelected;
    });
  };

  const filteredUrls = discoveredUrls.filter(url => 
    url.toLowerCase().includes(urlSearchTerm.toLowerCase())
  );

  const handleSelectAll = () => {
    setSelectedUrls(prevSelected => {
       const newSelected = new Set(prevSelected);
       filteredUrls.forEach(url => newSelected.add(url));
       return newSelected;
    });
  };

  const handleDeselectAll = () => setSelectedUrls(new Set());

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    if (!chatbotName) {
      setError('Please enter a name for your chatbot.');
      return;
    }
    let hasSource = false; 
    if (useFileSource && files.length > 0) hasSource = true;
    if ((useUrlSource || useSitemapSource) && selectedUrls.size > 0) hasSource = true; 

    if (!hasSource) {
       setError('Please provide at least one data source (Selected URLs or Files).');
       return;
    }

    setIsLoading(true);
    try {
      // clientId is retrieved from localStorage within apiService functions
      const chatbotData = {
        name: chatbotName, 
        useWebsite: useUrlSource, 
        websiteUrl: useUrlSource ? urlValue : '', 
        useSitemap: useSitemapSource, 
        sitemapUrl: useSitemapSource ? sitemapValue : '', 
        useFiles: useFileSource, 
        // Only include actual File objects (new uploads)
        files: useFileSource ? files.filter(f => f instanceof File) : [], 
        selectedUrls: Array.from(selectedUrls), 
        // Only include removed files in edit mode
        removed_files: isEditMode && useFileSource ? filesToRemove : [], 
      };

      // Clean up empty arrays before sending
      if (chatbotData.files.length === 0) delete chatbotData.files;
      if (chatbotData.removed_files.length === 0) delete chatbotData.removed_files;

      if (isEditMode) {
        // No need to delete empty arrays again here
        console.log("Submitting update data:", chatbotData); 
        const clientId = localStorage.getItem('clientId'); // Get clientId
        if (!clientId) {
          throw new Error("Client ID not found. Cannot update."); // Add check
        }
        // Pass chatbotId, data, AND clientId
        await apiService.updateChatbot(editChatbotId, chatbotData, clientId); 
        navigate('/dashboard'); 
      } else {
        // clientId handled by service.


        console.log('Submitting chatbotData:', chatbotData); // DEBUG: Check data before sending

        const response = await apiService.createChatbot(chatbotData);
        // Display the returned plaintext API key in a modal
        if (response && response.api_key && response.chatbot_id) {
          console.log(`Received API key for chatbot ${response.chatbot_id}`);
          setNewApiKey(response.api_key); // Store key in state
          setIsApiKeyModalOpen(true); // Open the modal
          // DO NOT navigate immediately, wait for modal close
        } else {
          console.warn("CreateChatbot response did not contain expected api_key or chatbot_id.");
          // If no key, navigate back anyway or show error? For now, navigate.
          navigate('/admin/dashboard');
        }
        // Removed: navigate('/dashboard'); - Navigation happens on modal close now
      }
    } catch (err) {
      setError(err.message || 'Failed to save chatbot.');
      if (err.message.includes('Maximum chatbot limit reached')) {
         setError('You have reached the maximum chatbot limit.');
      } else if (err.message.includes('Not logged in') || err.message.includes('Invalid session') || err.message.includes('Unauthorized')) {
         apiService.logout();
         navigate('/login');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    // AdminLayout provides the outer structure and padding
    <div className="flex flex-col gap-6"> {/* Use flex column and gap for spacing between cards */}
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-navy-700 dark:text-white">
          {isEditMode ? 'Edit Chatbot' : 'Create New Chatbot'}
        </h1>
        <Link
          to="/dashboard"
          className="flex items-center gap-2 text-sm font-medium text-brand-500 hover:text-brand-600 dark:text-brand-400 dark:hover:text-brand-300 transition duration-150 ease-in-out"
        >
          <FiArrowLeft />
          Back to Dashboard
        </Link>
      </div>

      {isLoading && isEditMode && !chatbotName && (
        <Card extra="p-6">
          <div className="flex items-center justify-center gap-2 text-gray-600 dark:text-gray-300">
            <FiLoader className="animate-spin h-5 w-5" />
            <span>Loading chatbot details...</span>
          </div>
        </Card>
      )}

      <form onSubmit={handleSubmit}> {/* Ensure layout classes are removed */}
        {/* Flex container for main content and discovered links */}
        <div className="flex flex-col lg:flex-row gap-6 mb-6"> {/* Added mb-6 */}
          {/* Main Content Column (2/3 width on large screens) */}
          <div className="lg:w-2/3 flex flex-col gap-6">
            {/* Basic Information Card */}
            <Card extra="p-6">
          <h2 className="text-xl font-semibold text-navy-700 dark:text-white mb-4">Basic Information</h2>
          <InputField
            variant="outlined" // Assuming an 'outlined' variant exists or using default
            extra="mb-3"
            label="Chatbot Name*"
            placeholder="My Awesome Bot"
            id="chatbotName"
            type="text"
            value={chatbotName}
            onChange={(e) => setChatbotName(e.target.value)}
            disabled={isLoading}
            required
          />
        </Card>

        {/* Data Sources Card */}
        <Card extra="p-6">
          <h2 className="text-xl font-semibold text-navy-700 dark:text-white mb-4">Data Sources</h2>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">Select one or more sources to train your chatbot.</p>

          {/* Horizontal Checkbox Selection */}
          <div className="flex flex-wrap gap-4 mb-6">
            <label htmlFor="useUrlSource" className="flex items-center gap-2 cursor-pointer text-sm font-medium text-navy-700 dark:text-white">
              <input
                type="checkbox"
                id="useUrlSource"
                checked={useUrlSource}
                onChange={(e) => setUseUrlSource(e.target.checked)}
                className="form-checkbox h-5 w-5 text-brand-500 rounded border-gray-300 focus:ring-brand-500 dark:border-gray-600 dark:bg-navy-700 dark:focus:ring-brand-400 dark:checked:bg-brand-400"
              />
              <FiLink className="h-5 w-5" /> Website URL
            </label>
            <label htmlFor="useSitemapSource" className="flex items-center gap-2 cursor-pointer text-sm font-medium text-navy-700 dark:text-white">
              <input
                type="checkbox"
                id="useSitemapSource"
                checked={useSitemapSource}
                onChange={(e) => setUseSitemapSource(e.target.checked)}
                className="form-checkbox h-5 w-5 text-brand-500 rounded border-gray-300 focus:ring-brand-500 dark:border-gray-600 dark:bg-navy-700 dark:focus:ring-brand-400 dark:checked:bg-brand-400"
              />
              <FiLink className="h-5 w-5" /> Sitemap
            </label>
            <label htmlFor="useFileSource" className="flex items-center gap-2 cursor-pointer text-sm font-medium text-navy-700 dark:text-white">
              <input
                type="checkbox"
                id="useFileSource"
                checked={useFileSource}
                onChange={(e) => setUseFileSource(e.target.checked)}
                className="form-checkbox h-5 w-5 text-brand-500 rounded border-gray-300 focus:ring-brand-500 dark:border-gray-600 dark:bg-navy-700 dark:focus:ring-brand-400 dark:checked:bg-brand-400"
              />
              <FiFileText className="h-5 w-5" /> File Upload
            </label>
          </div>

          {/* Conditional Inputs - Using Grid for better alignment */}
          <div className="grid grid-cols-1 md:grid-cols-1 gap-6"> {/* Changed to 1 col for now, can adjust */}
            {useUrlSource && (
              <div className="border-l-4 border-brand-500 dark:border-brand-400 pl-4 py-2 bg-gray-50 dark:bg-navy-700 rounded-r-lg">
                <InputField
                  variant="outlined"
                  label="Website URL"
                  placeholder="https://example.com"
                  id="urlValue"
                  type="url"
                  value={urlValue}
                  onChange={(e) => setUrlValue(e.target.value)}
                  disabled={isLoading}
                  extra="mb-2"
                />
                <button
                  type="button"
                  onClick={() => handleDiscoverLinks('url')}
                  disabled={isDiscovering || !urlValue}
                  className="flex items-center justify-center gap-2 linear rounded-md bg-brand-500 px-4 py-2 text-sm font-medium text-white transition duration-200 hover:bg-brand-600 active:bg-brand-700 dark:bg-brand-400 dark:hover:bg-brand-300 dark:active:bg-brand-200 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isDiscovering ? <FiLoader className="animate-spin h-4 w-4" /> : <FiLink className="h-4 w-4" />}
                  {isDiscovering ? 'Discovering...' : 'Discover Links'}
                </button>
              </div>
            )}

            {useSitemapSource && (
              <div className="border-l-4 border-brand-500 dark:border-brand-400 pl-4 py-2 bg-gray-50 dark:bg-navy-700 rounded-r-lg">
                <InputField
                  variant="outlined"
                  label="Sitemap URL"
                  placeholder="https://example.com/sitemap.xml"
                  id="sitemapValue"
                  type="url"
                  value={sitemapValue}
                  onChange={(e) => setSitemapValue(e.target.value)}
                  disabled={isLoading}
                  extra="mb-2"
                />
                <button
                  type="button"
                  onClick={() => handleDiscoverLinks('sitemap')}
                  disabled={isDiscovering || !sitemapValue}
                  className="flex items-center justify-center gap-2 linear rounded-md bg-brand-500 px-4 py-2 text-sm font-medium text-white transition duration-200 hover:bg-brand-600 active:bg-brand-700 dark:bg-brand-400 dark:hover:bg-brand-300 dark:active:bg-brand-200 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isDiscovering ? <FiLoader className="animate-spin h-4 w-4" /> : <FiLink className="h-4 w-4" />}
                  {isDiscovering ? 'Discovering...' : 'Discover Links'}
                </button>
              </div>
            )}

            {useFileSource && (
              <div className="border-l-4 border-brand-500 dark:border-brand-400 pl-4 py-2 bg-gray-50 dark:bg-navy-700 rounded-r-lg">
                <label htmlFor="fileUpload" className="text-sm font-bold text-navy-700 dark:text-white ml-3 mb-2 block">
                  Upload Files (.txt, .pdf, .docx)
                </label>
                {/* Basic File Input Styling */}
                <input
                  type="file"
                  id="fileUpload"
                  multiple
                  onChange={handleFileChange}
                  accept=".txt,.pdf,.doc,.docx,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                  className="mb-3 block w-full text-sm text-gray-500 dark:text-gray-400
                    file:mr-4 file:py-2 file:px-4
                    file:rounded-full file:border-0
                    file:text-sm file:font-semibold
                    file:bg-brand-50 file:text-brand-700
                    hover:file:bg-brand-100 dark:file:bg-navy-600 dark:file:text-brand-300 dark:hover:file:bg-navy-500"
                  disabled={isLoading}
                />
                {files.length > 0 && (
                  <ul className="mt-2 space-y-1">
                    {files.map((file, index) => (
                      <li
                        key={file.name + index}
                        className={`flex items-center justify-between p-2 rounded-md text-sm ${filesToRemove.includes(file.name) ? 'bg-red-100 dark:bg-red-900/50 opacity-60' : 'bg-gray-100 dark:bg-navy-600'}`}
                      >
                        <span className="flex items-center gap-2 truncate dark:text-gray-300">
                          <FiFileText className="h-4 w-4 flex-shrink-0" />
                          <span className="truncate" title={file.name}>{file.name}</span>
                          <span className="text-xs text-gray-500 dark:text-gray-400 ml-1">
                            {file.isExisting ? '(existing)' : '(new)'}
                          </span>
                        </span>
                        {!filesToRemove.includes(file.name) && (
                          <button
                            type="button"
                            onClick={() => handleRemoveFile(file)}
                            className="p-1 text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 rounded-full hover:bg-red-100 dark:hover:bg-red-900/50 transition duration-150"
                            title="Remove file"
                            disabled={isLoading}
                          >
                            <FiTrash2 className="h-4 w-4" />
                          </button>
                        )}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </div>
            </Card> {/* Close Data Sources Card */}
          </div> {/* End Main Content Column */}

          {/* Discovered Links Column (1/3 width on large screens) */}
          <div className="lg:w-1/3 flex flex-col gap-6">
            {/* Discovered Links Card (Only if URL or Sitemap is used) */}
            {(useUrlSource || useSitemapSource) && (
              <Card extra="p-6">
            <h2 className="text-xl font-semibold text-navy-700 dark:text-white mb-4">Discovered Links</h2>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">Select the links you want to include as knowledge sources.</p>

            {isDiscovering && !discoveryError && (
              <div className="flex items-center gap-2 text-blue-600 dark:text-blue-400 p-3 bg-blue-50 dark:bg-blue-900/50 rounded-md">
                <FiLoader className="animate-spin h-5 w-5" />
                <span>Discovering links... (Task ID: {discoveryTaskId || 'starting'})</span>
              </div>
            )}
            {discoveryError && (
              <div className="flex items-center gap-2 text-red-600 dark:text-red-400 p-3 bg-red-50 dark:bg-red-900/50 rounded-md">
                <FiAlertCircle className="h-5 w-5" />
                <span>Discovery Error: {discoveryError}</span>
              </div>
            )}

            {!isDiscovering && !discoveryError && (
              <>
                {discoveredUrls.length > 0 ? (
                  <div className="mt-4 space-y-4">
                    <div className="relative">
                      <InputField
                        variant="outlined"
                        label="Filter URLs"
                        placeholder="Search discovered links..."
                        id="urlFilter"
                        type="text"
                        value={urlSearchTerm}
                        onChange={(e) => setUrlSearchTerm(e.target.value)}
                        extra="!mb-0" // Remove default margin bottom
                      />
                       <FiFilter className="absolute right-3 top-9 h-5 w-5 text-gray-400 dark:text-gray-500" />
                    </div>

                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={handleSelectAll}
                          className="flex items-center gap-1 text-xs font-medium text-brand-500 hover:text-brand-700 dark:text-brand-300 dark:hover:text-brand-200 transition"
                          disabled={isLoading}
                        >
                          <FiCheckSquare /> Select Visible
                        </button>
                        <button
                          type="button"
                          onClick={handleDeselectAll}
                          className="flex items-center gap-1 text-xs font-medium text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition"
                          disabled={isLoading}
                        >
                          <FiSquare /> Deselect All
                        </button>
                      </div>
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        {selectedUrls.size} / {discoveredUrls.length} selected
                      </span>
                    </div>

                    <ul className="max-h-60 overflow-y-auto space-y-1 rounded-lg border border-gray-200 dark:border-navy-600 p-2 bg-gray-50 dark:bg-navy-700">
                      {filteredUrls.map((url) => {
                        const sanitizedId = `url-${url.replace(/[:/?=&#]/g, '-')}`;
                        return (
                          <li key={url} className="p-1 rounded hover:bg-gray-100 dark:hover:bg-navy-600">
                            <label htmlFor={sanitizedId} className="flex items-center gap-2 cursor-pointer text-sm text-gray-800 dark:text-gray-300">
                              <input
                                type="checkbox"
                                id={sanitizedId}
                                checked={selectedUrls.has(url)}
                                onChange={() => handleUrlSelectionChange(url)}
                                className="form-checkbox h-4 w-4 text-brand-500 rounded border-gray-300 focus:ring-brand-500 dark:border-gray-500 dark:bg-navy-600 dark:focus:ring-brand-400 dark:checked:bg-brand-400"
                                disabled={isLoading}
                              />
                              <span className="truncate" title={url}>{url}</span>
                            </label>
                          </li>
                        );
                      })}
                       {filteredUrls.length === 0 && urlSearchTerm && (
                         <li className="p-2 text-center text-sm text-gray-500 dark:text-gray-400">No URLs match your filter.</li>
                       )}
                       {filteredUrls.length === 0 && !urlSearchTerm && (
                         <li className="p-2 text-center text-sm text-gray-500 dark:text-gray-400">No URLs discovered yet or all filtered out.</li>
                       )}
                    </ul>
                  </div>
                ) : (
                  <div className="mt-4 p-4 text-center text-sm text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-navy-700 rounded-lg border border-dashed border-gray-300 dark:border-navy-600">
                    {discoveryTaskId ? (
                      <span>Discovery finished, no URLs found.</span>
                    ) : (
                      <span>Click "Discover Links" above to find links from the URL or Sitemap.</span>
                    )}
                  </div>
                )}
              </>
            )}
              </Card> // Close Discovered Links Card
            )} {/* Close conditional rendering */}
          </div> {/* End Discovered Links Column */}
        </div> {/* End Flex Container */}

        {/* Error Message (Now outside the flex container) */}
        {error && (
          <div className="flex items-center gap-2 text-red-600 dark:text-red-400 p-3 bg-red-50 dark:bg-red-900/50 rounded-md text-sm font-medium">
            <FiAlertCircle className="h-5 w-5" />
            <span>Error: {error}</span>
          </div>
        )}

        {/* Submit Button */}
        <div className="flex justify-end mt-2"> {/* Align button to the right */}
          <button
            type="submit"
            disabled={isLoading}
            className="flex items-center justify-center gap-2 linear rounded-xl bg-brand-500 px-6 py-3 text-base font-medium text-white transition duration-200 hover:bg-brand-600 active:bg-brand-700 dark:bg-brand-400 dark:text-white dark:hover:bg-brand-300 dark:active:bg-brand-200 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? <FiLoader className="animate-spin h-5 w-5" /> : (isEditMode ? <FiCheckCircle className="h-5 w-5" /> : <FiCheckCircle className="h-5 w-5" />)}
            {isLoading ? 'Saving...' : (isEditMode ? 'Update Chatbot' : 'Create Chatbot')}
          </button>
        </div>
      </form>

      {/* API Key Modal - Styled like Dashboard's delete modal */}
      <Modal
        isOpen={isApiKeyModalOpen}
        onRequestClose={() => {
          setIsApiKeyModalOpen(false);
          navigate('/admin/dashboard'); // Navigate after closing
        }}
        overlayClassName="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50" // Tailwind classes for overlay
        className="bg-white dark:bg-navy-800 p-6 md:p-8 rounded-lg shadow-xl max-w-lg w-full mx-auto outline-none border border-gray-200 dark:border-navy-700" // Tailwind classes for modal content
        contentLabel="API Key Generated"
      >
        <div className="flex justify-between items-center mb-4">
           <h2 className="text-xl font-semibold text-gray-800 dark:text-white">Chatbot Created Successfully!</h2>
           <button onClick={() => { setIsApiKeyModalOpen(false); navigate('/admin/dashboard'); }} className="text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300">
              <FiXCircle className="h-6 w-6" />
           </button>
        </div>
        <p className="text-gray-600 dark:text-gray-300 mb-4">Here is your API key. Please store it securely, as it will <strong className="font-medium">not</strong> be shown again:</p>
        <div className="relative bg-gray-100 dark:bg-navy-700 p-3 rounded-md border border-gray-200 dark:border-navy-600 mb-6">
          <pre className="text-sm text-gray-800 dark:text-gray-200 whitespace-pre-wrap break-all">
            <code>{newApiKey}</code>
          </pre>
          <button
             onClick={() => navigator.clipboard.writeText(newApiKey).then(() => alert('API Key copied!'), () => alert('Failed to copy API Key.'))}
             className="absolute top-2 right-2 p-1.5 text-gray-500 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-navy-600 rounded-md transition"
             title="Copy API Key"
          >
             <FiCopy className="h-4 w-4" />
          </button>
        </div>
        <div className="flex justify-end">
          <button
            onClick={() => {
              setIsApiKeyModalOpen(false);
              navigate('/admin/dashboard'); // Navigate after closing
            }}
            className="linear rounded-lg bg-brand-500 px-4 py-2 text-base font-medium text-white transition duration-200 hover:bg-brand-600 active:bg-brand-700 dark:bg-brand-400 dark:hover:bg-brand-300 dark:active:bg-brand-200"
          >
            Got it, Close
          </button>
        </div>
      </Modal>
    </div>
  );
}

export default CreateChatbotPage;
