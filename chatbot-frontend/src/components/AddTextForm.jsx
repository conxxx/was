import React, { useState } from 'react';

function AddTextForm({ onAddText }) {
  const [text, setText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError('');

    if (!text.trim()) {
      setError('Text content cannot be empty.');
      return;
    }

    // Optional: Add a character limit check if needed
    // const MAX_TEXT_LENGTH = 10000;
    // if (text.length > MAX_TEXT_LENGTH) {
    //   setError(`Text cannot exceed ${MAX_TEXT_LENGTH} characters.`);
    //   return;
    // }

    setIsLoading(true);
    try {
      await onAddText(text);
      setText(''); // Clear textarea on success
    } catch (err) {
      setError(err.message || 'Failed to add text snippet.');
      console.error("AddTextForm Error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} style={{ marginTop: '1rem', marginBottom: '1rem', padding: '1rem', border: '1px solid #eee' }}>
      <h4>Add Text Source</h4>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste or type text content here..."
        disabled={isLoading}
        rows={5}
        style={{ width: '100%', padding: '0.5rem', boxSizing: 'border-box', marginBottom: '0.5rem' }}
        aria-label="Text content to add"
      />
      <button type="submit" disabled={isLoading} style={{ padding: '0.5rem 1rem' }}>
        {isLoading ? 'Adding...' : 'Add Text'}
      </button>
      {error && <p style={{ color: 'red', marginTop: '0.5rem' }}>{error}</p>}
    </form>
  );
}

export default AddTextForm;