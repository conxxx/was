import React, { useState, useEffect } from 'react'; // Import useEffect
import Dropdown from '../dropdown/Dropdown'; // Corrected relative import path
import { FiAlignJustify, FiSearch } from 'react-icons/fi';
import { RiMoonFill, RiSunFill } from 'react-icons/ri';
import { IoMdNotificationsOutline, IoMdInformationCircleOutline } from 'react-icons/io';
import { Link, useNavigate } from 'react-router-dom'; // Use useNavigate for logout
// Removed import for non-existent avatar image
// import avatar from 'assets/img/avatars/avatar4.png';
import { FiUser } from 'react-icons/fi'; // Import a user icon as placeholder

// Placeholder Dropdown component if not already created
// const Dropdown = ({ button, children, classNames, animation }) => (
//   <div className="relative">
//     {button}
//     {/* Basic dropdown structure - replace with actual implementation */}
//     {/* <div className={`absolute right-0 mt-2 ${classNames} ${animation}`}>
//       {children}
//     </div> */}
//   </div>
// );

const DARK_MODE_KEY = 'darkModeEnabled';

const Navbar = (props) => {
  const { onOpenSidenav, brandText } = props;
  // Initialize state from localStorage
  const [darkmode, setDarkmode] = useState(() => {
    const savedMode = localStorage.getItem(DARK_MODE_KEY);
    // Check if savedMode is explicitly 'true', otherwise default based on body class or false
    return savedMode === 'true' || (!savedMode && document.body.classList.contains('dark'));
  });
  const navigate = useNavigate(); // Hook for navigation

  // Effect to apply class to body and update localStorage when darkmode state changes
  useEffect(() => {
    if (darkmode) {
      document.body.classList.add('dark');
      localStorage.setItem(DARK_MODE_KEY, 'true');
    } else {
      document.body.classList.remove('dark');
      localStorage.setItem(DARK_MODE_KEY, 'false');
    }
  }, [darkmode]); // Dependency array ensures this runs only when darkmode changes

  const handleLogout = () => {
    // Clear the client ID used for authentication
    console.log("Attempting to log out...");
    localStorage.removeItem('clientId');
    // Verify removal
    console.log("clientId after removal:", localStorage.getItem('clientId'));
    navigate('/'); // Redirect to the root/landing page
  };

  // Toggle function now only needs to update the state
  const toggleDarkMode = () => {
    setDarkmode(prevMode => !prevMode);
  };

  // Placeholder user data - replace with actual data from context or props
  const userName = "User Name";

  return (
    <nav className="sticky top-4 z-40 flex flex-row flex-wrap items-center justify-between rounded-xl bg-white/10 p-2 backdrop-blur-xl dark:bg-[#0b14374d]">
      {/* Left side: Page Title */}
      <div className="ml-[6px]">
        {/* Breadcrumb div removed */}
        <p className="shrink text-[33px] capitalize text-navy-700 dark:text-white">
          <Link
            to="#" // Link to the current page or dashboard home
            className="font-bold capitalize hover:text-navy-700 dark:hover:text-white"
          >
            {brandText}
          </Link>
        </p>
      </div>

      {/* Right side: Icons, Profile */}
      <div className="relative mt-[3px] flex h-[61px] w-auto flex-grow items-center justify-end gap-2 rounded-full bg-white px-2 py-2 shadow-xl shadow-shadow-500 dark:!bg-navy-800 dark:shadow-none md:flex-grow-0 md:gap-1 xl:gap-2">
        {/* Mobile Sidebar Toggle */}
        <span
          className="flex cursor-pointer text-xl text-gray-600 dark:text-white xl:hidden"
          onClick={onOpenSidenav}
        >
          <FiAlignJustify className="h-5 w-5" />
        </span>

        {/* Notification Icon - Placeholder Dropdown */}
        {/* <Dropdown
          button={
            <p className="cursor-pointer p-2">
              <IoMdNotificationsOutline className="h-5 w-5 text-gray-600 dark:text-white" />
            </p>
          }
          classNames={"py-2 top-4 -left-[230px] md:-left-[440px] w-max"}
        >
          <div className="flex w-[360px] flex-col gap-3 rounded-[20px] bg-white p-4 shadow-xl shadow-shadow-500 dark:!bg-navy-700 dark:text-white dark:shadow-none sm:w-[460px]">
            <p className="text-base font-bold text-navy-700 dark:text-white">Notifications Placeholder</p>
          </div>
        </Dropdown> */}

        {/* Info Icon - Placeholder Dropdown */}
        {/* <Dropdown
          button={
            <p className="cursor-pointer p-2">
              <IoMdInformationCircleOutline className="h-5 w-5 text-gray-600 dark:text-white" />
            </p>
          }
          classNames={"py-2 top-6 -left-[250px] md:-left-[330px] w-max"}
        >
           <div className="flex w-[350px] flex-col gap-2 rounded-[20px] bg-white p-4 shadow-xl shadow-shadow-500 dark:!bg-navy-700 dark:text-white dark:shadow-none">
             <p className="text-base font-bold text-navy-700 dark:text-white">Info Placeholder</p>
           </div>
        </Dropdown> */}

        {/* Dark Mode Toggle */}
        <div
          className="cursor-pointer p-2 text-gray-600"
          onClick={toggleDarkMode}
        >
          {darkmode ? (
            <RiSunFill className="h-5 w-5 text-gray-600 dark:text-white" />
          ) : (
            <RiMoonFill className="h-5 w-5 text-gray-600 dark:text-white" />
          )}
        </div>

        {/* Profile & Dropdown */}
        <Dropdown
          button={
            // Replace img with a placeholder icon or div
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gray-200 dark:bg-navy-700 cursor-pointer">
              <FiUser className="h-5 w-5 text-gray-600 dark:text-white" />
            </div>
          }
          classNames={"py-2 top-8 -right-4 md:-right-14 w-max"} // Adjusted position
        >
          <div className="flex w-56 flex-col justify-start rounded-[20px] bg-white bg-cover bg-no-repeat shadow-xl shadow-shadow-500 dark:!bg-navy-700 dark:text-white dark:shadow-none">
            <div className="p-4">
              <div className="flex items-center gap-2">
                <p className="text-sm font-bold text-navy-700 dark:text-white">
                  👋 Hey, {userName}
                </p>
              </div>
            </div>
            <div className="h-px w-full bg-gray-200 dark:bg-white/20 " />
            <div className="flex flex-col p-4">
              <Link
                to="/profile-settings" // Update with actual route
                className="text-sm text-gray-800 dark:text-white hover:dark:text-white mb-3"
              >
                Profile Settings
              </Link>
              {/* Add other links as needed */}
              <button
                onClick={handleLogout}
                className="mt-3 text-left text-sm font-medium text-red-500 hover:text-red-600 transition duration-150 ease-out hover:ease-in"
              >
                Log Out
              </button>
            </div>
          </div>
        </Dropdown>
      </div>
    </nav>
  );
};

export default Navbar;
