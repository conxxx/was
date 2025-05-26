import React, { useRef } from 'react'; // Import useRef
import { Link, useNavigate } from 'react-router-dom';
import { FiCopy, FiTrash2, FiEdit, FiMessageSquare, FiAlertCircle } from 'react-icons/fi'; // Relevant icons
import Card from './Card'; // Corrected import path

// Base Card component placeholder if not existing
// const Card = ({ children, extra }) => (
//   <div className={`relative flex flex-col break-words rounded-xl border border-gray-200 bg-white bg-clip-border shadow-md dark:!bg-navy-800 dark:text-white ${extra}`}>
//     {children}
//   </div>
// );

const ChatbotCard = ({ id, name, status, apiKey, createdAt, error, onDeleteClick, onCopyClick }) => {
  const navigate = useNavigate();
  const cardRef = useRef(null); // Create a ref for the card

  // --- 3D Hover Effect Logic ---
  const handleMouseMove = (e) => {
    if (!cardRef.current) return;

    const { left, top, width, height } = cardRef.current.getBoundingClientRect();
    const x = e.clientX - left; // x position within the element.
    const y = e.clientY - top;  // y position within the element.

    const rotateX = (y / height - 0.5) * -20; // Max rotation 10 degrees
    const rotateY = (x / width - 0.5) * 20;  // Max rotation 10 degrees

    cardRef.current.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.05, 1.05, 1.05)`; // Apply perspective and rotation, keep scale
    cardRef.current.style.transition = 'transform 0.1s ease-out'; // Faster transition when moving
  };

  const handleMouseLeave = () => {
    if (!cardRef.current) return;
    cardRef.current.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)'; // Reset transform
    cardRef.current.style.transition = 'transform 0.4s ease-in-out'; // Slower transition on leave
  };
  // -----------------------------

  const getStatusColor = (status) => {
    if (error) return 'text-red-500 dark:text-red-400';
    switch (status) {
      case 'Ready':
        return 'text-green-500 dark:text-green-400';
      case 'Processing':
      case 'Fetching':
      case 'Embedding':
        return 'text-yellow-500 dark:text-yellow-400';
      default:
        return 'text-gray-500 dark:text-gray-400';
    }
  };

  const handleEditClick = () => {
    navigate(`/admin/edit-chatbot/${id}`); // Added /admin prefix
  };

  const handleChatClick = () => {
    navigate(`/admin/chat/${id}`); // Corrected path
  };

  return (
    <Card
      ref={cardRef} // Attach the ref
      onMouseMove={handleMouseMove} // Add mouse move listener
      onMouseLeave={handleMouseLeave} // Add mouse leave listener
      // Removed hover:scale-105, adjusted transition (handled by JS now)
      // Removed bg-white and dark:shadow-none from extra to allow base Card glass effect
      extra="flex flex-col w-full h-full !p-4 3xl:p-![18px] transition-transform duration-400 ease-in-out hover:shadow-lg" // Keep base transition for leave
      style={{ transformStyle: 'preserve-3d' }} // Needed for perspective
    >
      <div className="flex h-full flex-col justify-between" style={{ transform: 'translateZ(20px)' }}> {/* Lift content slightly */}
        {/* Top Section: Name, Status, Created Date */}
        <div className="mb-4">
          <h3 className="text-lg font-bold text-navy-700 dark:text-white mb-1 truncate" title={name}>
            {name}
          </h3>
          <p className={`text-sm mb-1 font-medium ${getStatusColor(status)}`}>
            Status: {status}
            {error && (
              <span className="ml-2 inline-flex items-center text-red-500 dark:text-red-400" title={error}>
                <FiAlertCircle size={14} className="mr-1" /> Error
              </span>
            )}
          </p>
          {/* Created Date <p> removed */}
        </div>

        {/* Middle Section: API Key */}
        {apiKey && (
          <div className="mb-4 text-sm break-all bg-gray-100 dark:bg-navy-800 p-2 rounded border border-gray-200 dark:border-navy-600">
            <strong className="text-gray-600 dark:text-gray-300 text-xs block mb-1">API Key:</strong>
            <div className="flex items-center justify-between">
              <code className="text-navy-700 dark:text-white text-xs mr-2">{apiKey}</code>
              <button
                onClick={() => onCopyClick(apiKey)}
                className="text-gray-500 dark:text-gray-400 hover:text-brand-500 dark:hover:text-brand-400 p-1 rounded"
                title="Copy API Key"
              >
                <FiCopy size={16} />
              </button>
            </div>
          </div>
        )}

        {/* Bottom Section: Action Buttons */}
        <div className="mt-auto pt-3 border-t border-gray-200 dark:border-navy-600 flex flex-wrap gap-2 justify-end">
           {/* Updated button styles */}
          <button
            onClick={handleChatClick}
            className="linear flex items-center rounded-[20px] border border-white/20 bg-white/10 px-4 py-2 text-sm font-medium text-navy-700 backdrop-blur-sm transition duration-200 hover:bg-white/20 active:bg-white/30 dark:border-white/10 dark:bg-white/5 dark:text-white dark:hover:bg-white/15 dark:active:bg-white/20"
            title="Chat"
          >
            <FiMessageSquare className="mr-1" size={16}/> Chat
          </button>
          <button
            onClick={handleEditClick}
            className={`linear flex items-center rounded-[20px] border border-white/20 bg-white/10 px-4 py-2 text-sm font-medium text-navy-700 backdrop-blur-sm transition duration-200 hover:bg-white/20 active:bg-white/30 dark:border-white/10 dark:bg-white/5 dark:text-white dark:hover:bg-white/15 dark:active:bg-white/20 ${error ? 'opacity-50 cursor-not-allowed' : ''}`}
            disabled={!!error}
            title="Edit"
          >
             <FiEdit className="mr-1" size={16}/> Edit
          </button>
          <button
            onClick={() => onDeleteClick(id, name)} // Pass id and name to parent handler
            className="linear flex items-center rounded-[20px] border border-white/20 bg-white/10 px-4 py-2 text-sm font-medium text-navy-700 backdrop-blur-sm transition duration-200 hover:bg-white/20 active:bg-white/30 dark:border-white/10 dark:bg-white/5 dark:text-white dark:hover:bg-white/15 dark:active:bg-white/20"
            title="Delete"
          >
            <FiTrash2 className="mr-1" size={16}/> Delete
          </button>
        </div>
      </div>
    </Card>
  );
};

export default ChatbotCard;
