import React from 'react';

// Assuming robot3-video.mp4 is placed in the assets/aiaf folder
const videoUrl = new URL('../assets/aiaf/robot3-video.mp4', import.meta.url).href;

const AiafMonetizeSection = () => {
  return (
    <div className="min-h-screen flex items-center py-16 px-4 md:px-8 lg:px-12">
      <div className="grid grid-cols-1 md:grid-cols-[2fr_3fr] gap-16 items-center w-full max-w-screen-xl mx-auto"> {/* Added max-w and mx-auto */}

        {/* Left Column: Text Content */}
        <div className="text-white">
          <h2 className="text-5xl md:text-6xl lg:text-7xl font-bold leading-tight"> {/* Adjusted text size and leading */}
            Create, own, and monetize AI agents
          </h2>
          <p className="text-lg text-gray-300 mt-6">
            Our platform combines enterprise-grade AI infrastructure with blockchain technology, creating a new paradigm for AI democratization.
          </p>
          <div className="flex space-x-4 mt-10">
            <button className="bg-[#FF006B] hover:bg-[#E0005C] text-white font-semibold px-8 py-3 rounded-full transition duration-300">
              Learn more
            </button>
            <button className="bg-white hover:bg-gray-200 text-gray-900 font-semibold px-8 py-3 rounded-full transition duration-300">
              Join us
            </button>
          </div>
        </div>

        {/* Right Column: Video */}
        <div className="flex justify-center items-center"> {/* Centering the video */}
          <video
            src={videoUrl}
            autoPlay
            loop
            muted
            playsInline
            className="w-full h-auto rounded-lg max-w-full" // Ensure video scales correctly
            style={{ aspectRatio: '1 / 1' }} // Maintain 1:1 aspect ratio
          />
        </div>

      </div>
    </div>
  );
};

export default AiafMonetizeSection;
