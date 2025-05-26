import React from 'react';
import AiafNavbar from '../components/AiafNavbar';
import AiafVideoSection from '../components/AiafVideoSection';
import AiafExistsSection from '../components/AiafExistsSection';
import AiafMonetizeSection from '../components/AiafMonetizeSection'; // Import the monetize section
// We will create these components next

// Get background image URL
const backgroundImageUrl = new URL('../assets/aiaf/bgc.jpeg', import.meta.url).href;

const AiafLandingPage = () => {
  return (
    <div
      className="min-h-screen text-white font-poppins overflow-hidden bg-cover bg-center bg-fixed" // Removed bg color, added bg image classes
      style={{ backgroundImage: `url(${backgroundImageUrl})` }}
    >
      <AiafNavbar />
      <main className="relative z-10 pt-24 md:pt-28"> {/* Added top padding for fixed navbar */}
        <AiafVideoSection />
        <AiafExistsSection />
        <AiafMonetizeSection /> {/* Render the monetize section */}
        {/* Feature cards will be positioned within AiafVideoSection or relatively */}
      </main>
      {/* Add Footer if needed */}
    </div>
  );
};

export default AiafLandingPage;
