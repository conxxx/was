import React from 'react';

const AiafExistsSection = () => {
  const videoUrl = new URL('../assets/aiaf/robot2-video.mp4', import.meta.url).href;

  const listItems = [
    { number: '01', text: 'Enabling Direct Ownership True ownership of AI agents through NFT technology' },
    { number: '02', text: 'Democratizing Access Enterprise-grade infrastructure accessible to all' },
    { number: '03', text: 'Fair Value Distribution Direct monetization for AI creators' },
    { number: '04', text: 'Community Governance Decentralized decision-making and control' },
  ];

  return (
    <div className="min-h-screen flex items-center py-16 px-4 md:px-8 lg:px-12"> {/* Reduced horizontal padding */}
      <div className="grid grid-cols-1 md:grid-cols-[3fr_2fr] gap-16 items-center w-full"> {/* Changed to 3fr 2fr split */}
        {/* Left Column: Video */}
        <div className="flex justify-center">
          <video
            src={videoUrl}
            autoPlay
            loop
            muted
            playsInline
            className="w-full h-auto rounded-lg" // Removed max-w constraints
          />
        </div>

        {/* Right Column: Text Content */}
        <div>
          <h2 className="text-5xl md:text-6xl font-bold text-white mb-6"> {/* Increased text size and margin */}
            Why AIAF Exists?
          </h2>
          <p className="text-lg text-gray-300 mb-10"> {/* Increased text size and margin */}
            The AI revolution is creating unprecedented value, but this value remains concentrated in the hands of a few large organizations. We're changing this paradigm by
          </p>
          <div className="space-y-6"> {/* Increased spacing */}
            {listItems.map((item) => (
              <div key={item.number} className="flex items-start">
                <span className="text-3xl font-bold text-pink-500 mr-6 w-10 flex-shrink-0"> {/* Increased number size, margin, width */}
                  {item.number}
                </span>
                <p className="text-lg text-white">{item.text}</p> {/* Increased text size */}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AiafExistsSection;
