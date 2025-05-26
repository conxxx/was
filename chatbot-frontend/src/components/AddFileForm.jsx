import React, { useState, useRef } from 'react'; // Import useRef
import { FiFileText, FiUpload, FiTrash2 } from 'react-icons/fi'; // Import icons

function AddFileForm({ onAddFiles, disabled }) {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null); // Ref for the file input

  const handleFileChange = (event) => {
    setError(''); // Clear previous errors
    const files = event.target.files;
    if (!files || files.length === 0) {
      return;
    }
    // Basic validation (can be expanded)
    const allowedTypes = ['text/plain', 'application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
    const newFiles = Array.from(files);
    const validFiles = newFiles.filter(file => allowedTypes.includes(file.type));
    const invalidFiles = newFiles.filter(file => !allowedTypes.includes(file.type));

    if (invalidFiles.length > 0) {
      setError(`Invalid file type(s): ${invalidFiles.map(f => f.name).join(', ')}. Allowed: .txt, .pdf, .doc, .docx`);
    }

    // Prevent duplicates
    setSelectedFiles(prevFiles => {
        const existingNames = new Set(prevFiles.map(f => f.name));
        const uniqueNewFiles = validFiles.filter(f => !existingNames.has(f.name));
        return [...prevFiles, ...uniqueNewFiles];
    });

    event.target.value = null; // Reset file input
  };

  const handleRemoveFile = (fileToRemove) => {
    setSelectedFiles(prevFiles => prevFiles.filter(file => file !== fileToRemove));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    if (selectedFiles.length === 0) {
      setError('Please select at least one file to upload.');
      return;
    }
    setError('');
    onAddFiles(selectedFiles); // Pass the File objects to the parent
    setSelectedFiles([]); // Clear selection after passing to parent
  };

  // Trigger file input click
  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  // Button styles from CreateChatbotPage reference
  const buttonClass = "flex items-center justify-center gap-2 linear rounded-md bg-brand-500 px-4 py-2 text-sm font-medium text-white transition duration-200 hover:bg-brand-600 active:bg-brand-700 dark:bg-brand-400 dark:hover:bg-brand-300 dark:active:bg-brand-200 disabled:opacity-50 disabled:cursor-not-allowed";
  // Style for the "Choose files" button to match the image (using gray/navy as brand isn't used there)
  const chooseFileButtonClass = "linear rounded-md bg-gray-200 px-4 py-2 text-sm font-medium text-gray-700 transition duration-200 hover:bg-gray-300 active:bg-gray-400 dark:bg-navy-600 dark:text-gray-200 dark:hover:bg-navy-500 dark:active:bg-navy-400 disabled:opacity-50 disabled:cursor-not-allowed";


  return (
    // Removed the outer container div with border/background - handled by parent
    <div>
      {/* Removed Title */}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          {/* Label matches the reference image style */}
          <label className="block text-sm font-medium text-gray-700 dark:text-white mb-2">
            Upload Files (.txt, .pdf, .docx)
          </label>
          {/* Hidden actual file input */}
          <input
            type="file"
            id="fileUploadHidden"
            ref={fileInputRef}
            multiple
            onChange={handleFileChange}
            accept=".txt,.pdf,.doc,.docx,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            className="hidden" // Hide the default input
            disabled={disabled}
          />
          {/* Custom "Choose files" button and text */}
          <div className="flex items-center gap-2">
             <button
               type="button"
               onClick={triggerFileInput}
               className={chooseFileButtonClass} // Apply specific style
               disabled={disabled}
             >
               Choose files
             </button>
             <span className="text-sm text-gray-500 dark:text-gray-400">
                {selectedFiles.length === 0 ? 'No file chosen' : `${selectedFiles.length} file(s) selected`}
             </span>
          </div>

          {error && <p className="text-sm text-red-500 dark:text-red-400 mt-1">{error}</p>}
        </div>

        {/* Display selected files (optional, can be simplified if not needed) */}
        {selectedFiles.length > 0 && (
          <div className="space-y-1">
             {/* <p className="text-sm font-medium text-gray-600 dark:text-gray-300">Files to upload:</p> */}
             <ul className="list-none p-0 m-0 max-h-32 overflow-y-auto border border-gray-300 dark:border-navy-600 rounded-md divide-y divide-gray-200 dark:divide-navy-600 bg-white dark:bg-navy-800">
              {selectedFiles.map((file, index) => (
                <li
                  key={index} // Use index as key for temporary list items
                  className="px-3 py-1 flex items-center justify-between text-sm text-gray-700 dark:text-gray-200" // Reduced padding
                >
                  <span className="flex items-center gap-2 truncate">
                    <FiFileText className="h-4 w-4 flex-shrink-0 text-gray-500 dark:text-gray-400" />
                    <span className="truncate" title={file.name}>{file.name}</span>
                  </span>
                  <button
                    type="button"
                    onClick={() => handleRemoveFile(file)}
                    className="p-1 text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 rounded-full hover:bg-red-100 dark:hover:bg-red-900/50 transition duration-150"
                    title="Remove file"
                    disabled={disabled}
                  >
                    <FiTrash2 className="h-4 w-4" />
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Submit button matches the reference image */}
        <button type="submit" disabled={disabled || selectedFiles.length === 0} className={buttonClass}>
          <FiUpload className="h-4 w-4" />
          Add Selected Files
        </button>
      </form>
    </div>
  );
}

export default AddFileForm;