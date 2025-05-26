import React from 'react';
import { Link } from 'react-router-dom'; // Import Link for navigation

// Use new URL pattern for Vite asset handling
const logoVideoUrl = new URL('../assets/aiaf/logo-video.mp4', import.meta.url).href;

const AiafNavbar = () => {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between p-6 px-8 md:px-12 lg:px-24 text-white bg-[#0B1437]/80 backdrop-blur-sm shadow-md"> {/* Changed absolute to fixed, added background, blur, and shadow */}
      {/* Logo Video */}
      <div className="flex items-center">
        <video
          src={logoVideoUrl} // Use the generated URL
          autoPlay
          loop
          muted
          playsInline // Important for mobile playback
          className="w-10 h-10 md:w-12 md:h-12 object-cover rounded-full mr-3" // Circular video logo
        />
        <span className="text-xl md:text-2xl font-bold">aiaf</span> {/* Placeholder text next to logo */}
      </div>

      {/* Navigation Links */}
      <div className="hidden md:flex items-center space-x-8">
        <a href="#home" className="hover:text-gray-300">Home</a>
        <a href="#technology" className="hover:text-gray-300">Technology</a>
        <a href="#about" className="hover:text-gray-300">About</a>
      </div>

      {/* Action Buttons */}
      <div className="flex items-center space-x-4">
        {/* Learn More button (assuming it might link somewhere later, or trigger a modal) */}
        <button className="bg-white text-[#0B1437] px-4 py-2 rounded-lg text-sm font-semibold hover:bg-gray-200 transition duration-200">
          Learn more
        </button>
        {/* Changed Join Us button to a Link */}
        <Link 
          to="/login" 
          className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-4 py-2 rounded-lg text-sm font-semibold hover:opacity-90 transition duration-200"
        >
          Join us
        </Link>
      </div>

      {/* Mobile Menu Button (Optional - Add later if needed) */}
      {/* <div className="md:hidden"> ... </div> */}
    </nav>
  );
};

export default AiafNavbar;
