// chatbot-frontend/src/pages/DashboardPage.jsx
import React, { useState, useEffect, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import Modal from 'react-modal';
import apiService, { API_URL } from '../services/api';
import axios from 'axios';
import { FiPlus, FiCopy, FiTrash2, FiEdit, FiMessageSquare } from 'react-icons/fi'; // Removed FiLogOut as it's in Navbar now
import ChatbotCard from '../components/card/ChatbotCard'; // Import the new ChatbotCard

// Make sure to bind modal to your appElement (http://reactcommunity.org/react-modal/accessibility/)
Modal.setAppElement('#root'); // Assuming your root element has id 'root'

function DashboardPage() {
  const [chatbots, setChatbots] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const navigate = useNavigate();
  const eventSourceRef = useRef(null);
  // Removed logo state variables

  // --- Modal State ---
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [chatbotToDeleteId, setChatbotToDeleteId] = useState(null);
  const [chatbotToDeleteName, setChatbotToDeleteName] = useState('');
  // -------------------

  // Function to handle incoming status updates
  const handleStatusUpdate = (data) => {
    console.log("SSE Status Update Received:", data);
    if (data && data.chatbot_id && data.status) {
      setChatbots(prevChatbots =>
        prevChatbots.map(bot =>
          bot.id === data.chatbot_id ? { ...bot, status: data.status } : bot
        )
      );
    } else {
      console.warn("Received incomplete SSE message:", data);
    }
  };

  // Removed fetchCurrentLogo function

  const fetchChatbots = async () => { // Removed unused clientId parameter
    setIsLoading(true);
    setError('');
    try {
      const basicBotList = await apiService.getChatbots();
      if (!basicBotList || basicBotList.length === 0) {
        setChatbots([]);
        setIsLoading(false);
        return;
      }
      const detailedBotsPromises = basicBotList.map(bot =>
        apiService.getChatbotDetails(bot.id)
          .catch(err => {
            console.error(`Failed to fetch details for bot ${bot.id}:`, err);
            return { ...bot, error: 'Could not load details' };
          })
      );
      const detailedBots = await Promise.all(detailedBotsPromises);
      setChatbots(detailedBots || []);

    } catch (err) {
      setError(err.message || 'Failed to fetch chatbots list.');
      if (err.message.includes('Not logged in') || err.message.includes('Invalid session')) {
        apiService.logout();
        navigate('/login');
      }
    } finally {
      setIsLoading(false);
    }
  };


  useEffect(() => {
    const clientId = localStorage.getItem('clientId');
    if (!clientId) {
        navigate('/login'); // Redirect if no client ID (not logged in)
        return;
    };

    fetchChatbots(clientId);
    // Removed fetchCurrentLogo() call

    console.log("Attempting to establish SSE connection for client:", clientId);
    eventSourceRef.current = apiService.createStatusStream(clientId, handleStatusUpdate);
    console.log("EventSource instance:", eventSourceRef.current);

    return () => {
      if (eventSourceRef.current) {
        console.log("Closing SSE connection.");
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [navigate]); // Removed fetchChatbots from dependency array as it causes infinite loop if not memoized

  // Removed handleLogoSelect function
  // Removed handleLogoUpload function

  // handleLogout is now handled by the Navbar component

  // --- Modal Control Functions ---
  const openDeleteModal = (id, name) => {
    setChatbotToDeleteId(id);
    setChatbotToDeleteName(name);
    setIsModalOpen(true);
  };

  const closeDeleteModal = () => {
    setIsModalOpen(false);
    setChatbotToDeleteId(null);
    setChatbotToDeleteName('');
  };
  // -----------------------------

  const confirmDelete = async () => {
    if (!chatbotToDeleteId) return;

    const id = chatbotToDeleteId;
    closeDeleteModal();

    try {
      const clientId = localStorage.getItem('clientId');
      if (!clientId) {
         throw new Error("Client ID not found. Cannot delete.");
      }
      await apiService.deleteChatbot(id, clientId);

      setChatbots(prevChatbots => prevChatbots.filter(bot => bot.id !== id));
      alert('Chatbot deleted successfully');

    } catch (err) {
      console.error('Delete chatbot error:', err);
      alert(err.message || 'Failed to delete chatbot.');
      setError(err.message || 'Failed to delete chatbot.');
    }
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text).then(() => {
      alert('API Key copied to clipboard!');
    }, (err) => {
      console.error('Could not copy text: ', err);
      alert('Failed to copy API Key.');
    });
  };


  // The AdminLayout now provides the overall structure, padding, and background.
  // We only need to return the content specific to the dashboard page.
  return (
    <>
      {/* Company Logo Upload Section Removed */}

      {/* Chatbots List Section */}
      <h2 className="text-2xl font-semibold text-gray-800 mb-6">Your Chatbots</h2>
      {isLoading && <p className="text-gray-600">Loading chatbots...</p>}
      {error && <p className="text-red-600 font-semibold">Error: {error}</p>}

      {!isLoading && !error && chatbots.length === 0 && (
        <p className="text-gray-500 italic">No chatbots found. Create one!</p>
      )}

      {!isLoading && !error && chatbots.length > 0 && (
        // Use a div with grid layout instead of ul
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
          {chatbots.map(bot => (
            // Render ChatbotCard for each bot
            <ChatbotCard
              key={bot.id}
              id={bot.id}
              name={bot.name}
              status={bot.status}
              apiKey={bot.api_key}
              createdAt={bot.created_at}
              error={bot.error} // Pass the error message if any
              onDeleteClick={openDeleteModal} // Pass the function to open the modal
              onCopyClick={copyToClipboard} // Pass the function to copy API key
            />
          ))}
        </div>
      )}

      {/* --- Confirmation Modal --- */}
      <Modal
        isOpen={isModalOpen}
        onRequestClose={closeDeleteModal}
        overlayClassName="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50"
        className="bg-white p-8 rounded-lg shadow-xl max-w-md w-full mx-auto outline-none"
        contentLabel="Confirm Deletion"
      >
        <h2 className="text-xl font-semibold mb-4 text-gray-800">Confirm Deletion</h2>
        <p className="text-gray-600 mb-6">Are you sure you want to delete chatbot '<strong className="font-medium text-gray-700">{chatbotToDeleteName}</strong>'? This action cannot be undone.</p>
        <div className="flex justify-end gap-4">
          <button
            onClick={closeDeleteModal}
            className="bg-gray-300 hover:bg-gray-400 text-gray-800 font-semibold py-2 px-4 rounded transition duration-150 ease-in-out"
          >
            Cancel
          </button>
          <button
            onClick={confirmDelete}
            className="bg-red-600 hover:bg-red-700 text-white font-semibold py-2 px-4 rounded transition duration-150 ease-in-out"
          >
            Confirm Delete
          </button>
        </div>
      </Modal>
      {/* ------------------------ */}
    </>
  );
}

export default DashboardPage;
