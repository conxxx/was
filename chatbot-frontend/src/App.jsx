import React from 'react';
import { Routes, Route, Navigate, Outlet } from 'react-router-dom'; // Restore Outlet import

// Import pages and layouts
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import CreateChatbotPage from './pages/CreateChatbotPage';
import ChatPage from './pages/ChatPage';
import EditChatbotPage from './pages/EditChatbotPage';
import AiafLandingPage from './pages/AiafLandingPage'; // Landing page (aiaf)
import AdminLayout from './layouts/AdminLayout'; // Import the new Admin Layout

// ProtectedRoute component: Checks auth and renders Outlet or redirects to login
const ProtectedRoute = () => {
  const clientId = localStorage.getItem('clientId'); 
  // Keep console logs for debugging
  console.log("ProtectedRoute check - clientId:", clientId); 

  // If not authenticated, redirect to login.
  if (!clientId) { 
    console.log("ProtectedRoute: No clientId found, redirecting to /login"); 
    return <Navigate to="/login" replace />;
  }
  
  // If authenticated, allow the nested routes (which will include AdminLayout) to render.
  console.log("ProtectedRoute: ClientId found, rendering Outlet"); 
  return <Outlet />; 
};

function App() {
  return (
    // TODO: Add ThemeProvider, GlobalStyles, ToastContainer later
    <Routes>
      {/* --- Specific Public Routes (Defined First) --- */}
      <Route path="/" element={<AiafLandingPage />} /> 
      <Route path="/login" element={<LoginPage />} />
      
      {/* --- Protected Routes Wrapper (Under /admin base path) --- */}
      {/* This wrapper checks authentication */}
      <Route path="/admin" element={<ProtectedRoute />}> 
        {/* This wrapper applies the layout to authenticated routes */}
        <Route element={<AdminLayout />}> 
          {/* Index route for the /admin section redirects to /admin/dashboard */}
          <Route index element={<Navigate replace to="/admin/dashboard" />} /> 
          {/* Specific protected routes relative to /admin */}
          <Route path="dashboard" element={<DashboardPage />} />
          <Route path="create-chatbot" element={<CreateChatbotPage />} />
          <Route path="edit-chatbot/:id" element={<EditChatbotPage />} />
          <Route path="chat/:id" element={<ChatPage />} />
          {/* Add other protected routes here, relative to /admin */}
        </Route>
      </Route>

      {/* --- Fallback Route (Defined Last) --- */}
      {/* Catches any unmatched paths and redirects to the landing page */}
      <Route path="*" element={<Navigate replace to="/" />} /> 
    </Routes>
  );
}

export default App;
