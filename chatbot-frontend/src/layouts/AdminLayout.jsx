import React, { useState, useEffect } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import Navbar from '../components/navbar/Navbar'; // Corrected relative import path
import Sidebar from '../components/sidebar/Sidebar'; // Corrected relative import path
// import Footer from 'components/footer/Footer'; // Assuming a Footer component exists or will be created


export default function AdminLayout(props) {
  const location = useLocation();
  const [open, setOpen] = useState(true); // Default to open on desktop
  const [currentRoute, setCurrentRoute] = useState("Dashboard"); // Simplified route naming

  // Adjust sidebar open state based on window size
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 1200) {
        setOpen(false);
      } else {
        setOpen(true);
      }
    };
    window.addEventListener('resize', handleResize);
    // Initial check
    handleResize();
    // Cleanup listener on component unmount
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Update current route name based on location
  useEffect(() => {
    // Basic example: derive name from path
    const pathSegments = location.pathname.split('/').filter(Boolean);
    const lastSegment = pathSegments[pathSegments.length - 1];
    const routeName = lastSegment ? lastSegment.replace(/-/g, ' ') : 'Dashboard';
    setCurrentRoute(routeName.charAt(0).toUpperCase() + routeName.slice(1));
  }, [location.pathname]);

  return (
    <div className="flex h-full w-full bg-lightPrimary dark:!bg-navy-900">
      {/* Use the actual Sidebar component */}
      <Sidebar open={open} onClose={() => setOpen(false)} />

      {/* Navbar & Main Content */}
      <div className="h-full w-full transition-all duration-300 ease-in-out">
        {/* Main Content */}
        <main
          // Added background gradient to make card glass effect visible
          className={`mx-auto h-full flex-none transition-all duration-300 ease-in-out xl:pr-2 bg-gradient-to-br from-purple-50 via-white to-blue-50 dark:from-navy-900 dark:via-navy-800 dark:to-navy-700 ${
            open ? 'xl:ml-64' : 'xl:ml-0' // Adjust margin based on sidebar state
          }`}
        >
          {/* Routes */}
          <div className="h-full">
            {/* Use the actual Navbar component */}
            <Navbar
              onOpenSidenav={() => setOpen(true)}
              brandText={currentRoute}
              // logoText={"Chatbot UI"} // Optional: if you want a logo text
              {...props} // Pass any additional props
            />
            <div className="pt-5s mx-auto mb-auto h-full min-h-[calc(100vh-100px)] p-2 md:pr-2">
              {/* The Outlet renders the matched child route element (e.g., DashboardPage) */}
              <Outlet /> {/* Child routes will render here */}
            </div>
            <div className="p-3">
              {/* Use a real Footer component if available */}
              {/* <Footer /> */}
              <div className="text-center text-sm text-gray-500 dark:text-gray-400">
                © {new Date().getFullYear()} Chatbot Admin. All Rights Reserved.
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
