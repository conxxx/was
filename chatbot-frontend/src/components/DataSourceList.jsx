import React from 'react';
import { FiTrash2 } from 'react-icons/fi'; // Import a trash icon

function DataSourceList({ sources, onDeleteSource, disabled }) {
  console.log('DataSourceList: Received sources prop:', sources);

  // Destructure all expected fields from the sources object, providing defaults
  const {
    added_files = [],
    added_urls = [],
    files_uploaded = [],
    original_sitemap = '',
    original_url = '',
    selected_urls = [],
    crawled_urls_added = []
  } = sources;

  // Combine all URL sources into a single unique list of objects
  const allUrlSources = [
    ...(original_url ? [{ type: 'Original URL', value: original_url, id: original_url }] : []),
    ...added_urls.map(url => ({ type: 'Added URL', value: url, id: url })),
    ...selected_urls.map(url => ({ type: 'Selected URL', value: url, id: url })),
    ...crawled_urls_added.map(url => ({ type: 'Crawled URL', value: url, id: url }))
  ];
  // Ensure uniqueness based on URL value
  const uniqueUrlSources = Array.from(new Map(allUrlSources.map(item => [item.value, item])).values());


  // Combine all file sources into a single unique list of objects
  const allFileSources = [
    ...added_files.map(file => ({ type: 'Added File', value: file, id: file })),
    ...files_uploaded.map(file => ({ type: 'Uploaded File', value: file, id: file }))
  ];
   // Ensure uniqueness based on filename value
  const uniqueFileSources = Array.from(new Map(allFileSources.map(item => [item.value, item])).values());

  // Add Sitemap to URL sources if it exists
  const finalUrlSources = [
      ...uniqueUrlSources,
      ...(original_sitemap ? [{ type: 'Sitemap', value: original_sitemap, id: original_sitemap }] : [])
  ];

  console.log('DataSourceList: Final URL Sources:', finalUrlSources);
  console.log('DataSourceList: Final File Sources:', uniqueFileSources);

  const handleDelete = (sourceType, sourceId) => {
    // Confirmation is now handled in the parent component (EditChatbotPage)
    // Adjust sourceType for API call if needed (e.g., 'URL' or 'File')
    const apiSourceType = sourceType.includes('URL') || sourceType === 'Sitemap' ? 'URL' : 'File';
    onDeleteSource(apiSourceType, sourceId); // Call parent handler directly
  };

  // Helper function to render a table for a specific source type
  const renderTable = (title, data, typeIdentifier) => (
    <div className="mb-8"> {/* Add margin between tables */}
      <h4 className="text-md font-semibold text-gray-600 dark:text-gray-300 mb-3 border-b border-gray-300 dark:border-navy-600 pb-2">{title}</h4>
      {data.length === 0 ? (
        <p className="text-sm text-gray-500 dark:text-gray-400">No {title.toLowerCase()} added yet.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-navy-700 border border-gray-200 dark:border-navy-700 rounded-lg">
            <thead className="bg-gray-50 dark:bg-navy-800">
              <tr>
                <th scope="col" className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Type
                </th>
                <th scope="col" className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Source
                </th>
                <th scope="col" className="px-4 py-2 text-right text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-navy-900 divide-y divide-gray-200 dark:divide-navy-700">
              {data.map((source) => (
                <tr key={`${typeIdentifier}-${source.id}`}>
                  <td className="px-4 py-2 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                    {source.type}
                  </td>
                  <td className="px-4 py-2 whitespace-normal text-sm text-gray-500 dark:text-gray-400 break-all">
                    {source.value}
                  </td>
                  <td className="px-4 py-2 whitespace-nowrap text-right text-sm font-medium">
                    <button
                      onClick={() => handleDelete(source.type, source.id)}
                      disabled={disabled}
                      className="text-red-600 hover:text-red-900 dark:text-red-400 dark:hover:text-red-600 disabled:opacity-50 disabled:cursor-not-allowed p-1 rounded-full hover:bg-red-100 dark:hover:bg-red-900/50"
                      aria-label={`Remove ${source.type} ${source.value}`}
                    >
                      <FiTrash2 className="inline h-4 w-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );


  return (
    <div className="mt-6 p-4 bg-white dark:bg-navy-800 rounded-lg shadow-md border border-gray-200 dark:border-navy-700">
      <h3 className="text-lg font-semibold text-gray-700 dark:text-white mb-4">Existing Data Sources</h3>
      {finalUrlSources.length === 0 && uniqueFileSources.length === 0 ? (
         <p className="text-gray-500 dark:text-gray-400">No data sources added yet.</p>
      ) : (
        <>
          {renderTable("Link Sources", finalUrlSources, "url")}
          {renderTable("File Sources", uniqueFileSources, "file")}
        </>
      )}
    </div>
  );
}

export default DataSourceList;