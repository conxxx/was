import React, { useState } from 'react';

function AddUrlForm({ onAddUrl }) {
  const [url, setUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false); // Optional: for loading state during submission
  const [error, setError] = useState(''); // Optional: for displaying input errors

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError(''); // Clear previous errors

    // Basic validation
    if (!url.trim()) {
      setError('URL cannot be empty.');
      return;
    }
    // Simple URL format check (can be improved)
    try {
      new URL(url); // Check if it's a valid URL structure
    } catch (_) { // eslint-disable-line no-unused-vars
      setError('Please enter a valid URL (e.g., https://example.com).');
      return;
    }

    setIsLoading(true);
    try {
      await onAddUrl(url); // Call the callback passed from parent
      setUrl(''); // Clear input on success
    } catch (err) {
      // Error handling can be done in the parent or here
      setError(err.message || 'Failed to add URL.');
      console.error("AddUrlForm Error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} style={{ marginTop: '1rem', marginBottom: '1rem', padding: '1rem', border: '1px solid #eee' }}>
      <h4>Add URL Source</h4>
      <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
        <input
          type="url" // Use type="url" for basic browser validation
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://example.com/page"
          disabled={isLoading}
          style={{ flexGrow: 1, padding: '0.5rem' }}
          aria-label="URL to add"
        />
        <button type="submit" disabled={isLoading} style={{ padding: '0.5rem 1rem' }}>
          {isLoading ? 'Adding...' : 'Add URL'}
        </button>
      </div>
      {error && <p style={{ color: 'red', marginTop: '0.5rem' }}>{error}</p>}
    </form>
  );
}

export default AddUrlForm;