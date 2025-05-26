import React from 'react';
import { HiX } from 'react-icons/hi';
import { NavLink, useLocation, Link } from 'react-router-dom'; // Added Link
import { LuLayoutDashboard } from "react-icons/lu"; // Example icon
import { FiPlus } from 'react-icons/fi'; // Added FiPlus

// Simple Link component for the sidebar
const SidebarLink = ({ route, active }) => {
  return (
    <NavLink
      to={route.path}
      className={`flex items-center p-3 my-1 rounded-lg transition-colors duration-200 ${
        active
          ? 'bg-gray-100 dark:bg-navy-700 font-semibold text-navy-700 dark:text-white' // Subtle active style
          : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-navy-700' // Subtle hover
      }`}
    >
      <span className="mr-3">{route.icon || <LuLayoutDashboard />}</span>
      <span>{route.name}</span>
    </NavLink>
  );
};

// Main Sidebar Component
const Sidebar = ({ open, onClose }) => {
  const location = useLocation();

  // Define routes for the sidebar
  // Add more routes here as needed (e.g., Settings, Profile)
  const sidebarRoutes = [
    {
      path: '/admin/dashboard', // Updated path with /admin prefix
      name: 'Dashboard',
      icon: <LuLayoutDashboard className="h-5 w-5" />,
    },
    // Add other routes like:
    // { path: '/settings', name: 'Settings', icon: <FiSettings className="h-5 w-5" /> },
  ];

  // Function to check if a route is active
  const isActive = (routePath) => {
    // Check if the current location pathname starts with the route path
    // This handles nested routes as well (e.g., /dashboard/details)
    return location.pathname === routePath || location.pathname.startsWith(routePath + '/');
  };


  return (
    <div
      className={`sm:none duration-175 linear fixed !z-50 flex min-h-full w-64 flex-col bg-white bg-opacity-75 backdrop-blur-md pb-10 shadow-2xl shadow-white/5 transition-all dark:!bg-navy-800 dark:bg-opacity-75 dark:text-white md:!z-50 lg:!z-50 xl:!z-0 ${ // Added bg-opacity, backdrop-blur
        open ? 'translate-x-0' : '-translate-x-full' // Use -translate-x-full for consistent hiding
      }`}
    >
      {/* Close button for mobile */}
      <span
        className="absolute top-4 right-4 block cursor-pointer xl:hidden"
        onClick={onClose}
      >
        <HiX className="h-6 w-6 text-gray-600 dark:text-white" />
      </span>

      {/* Logo / Brand */}
      <div className={`mx-[56px] mt-[50px] flex items-center`}>
        <div className="mt-1 ml-1 h-2.5 font-poppins text-[26px] font-bold uppercase text-navy-700 dark:text-white">
          Chatbot <span className="font-medium">UI</span>
        </div>
      </div>
      <div className="mt-[58px] mb-7 h-px bg-gray-300 dark:bg-white/30" />

      {/* Create New Chatbot Button */}
      <div className="px-4 mb-4"> {/* Add padding and margin */}
        <Link
          to="/admin/create-chatbot" // Updated path with /admin prefix
          className="flex w-full items-center justify-center rounded-lg bg-brand-500 py-3 px-4 text-sm font-medium text-white transition duration-200 hover:bg-brand-600 active:bg-brand-700 dark:bg-brand-400 dark:hover:bg-brand-300 dark:active:bg-brand-200"
        >
          <FiPlus className="mr-2 h-5 w-5" /> Create New Chatbot
        </Link>
      </div>

      {/* Navigation Links */}
      <ul className="mb-auto pt-1 px-4">
        {sidebarRoutes.map((route, index) => (
          <li key={index}>
            <SidebarLink route={route} active={isActive(route.path)} />
          </li>
        ))}
      </ul>

      {/* Optional: Add footer elements or user info here if needed */}
      {/* <div className="flex justify-center"> */}
        {/* Example: <UserProfileCard /> */}
      {/* </div> */}
    </div>
  );
};

export default Sidebar;
