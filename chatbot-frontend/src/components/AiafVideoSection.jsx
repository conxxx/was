import React from 'react';
import AiafFeatureCard from './AiafFeatureCard';

// Use new URL pattern for Vite asset handling
const backgroundImageUrl = new URL('../assets/aiaf/bgc.jpeg', import.meta.url).href;
const mainVideoUrl = new URL('../assets/aiaf/robot-video.mp4', import.meta.url).href;

const AiafVideoSection = () => {
  // Adjusted rotation angles for 3D tilt (rotateZ and rotateY) and added glow direction
  const features = [
    { number: '01', title: 'True Ownership', description: 'Full control of your AI agents through NFT ownership', transform: 'rotateZ(8deg) rotateY(25deg)', glowDirection: 'right' }, // Increased rotateY
    { number: '02', title: 'Professional Infrastructure', description: 'Enterprise-grade performance with decentralized benefits', transform: 'rotateZ(-8deg) rotateY(-25deg)', glowDirection: 'left' }, // Increased rotateY
    { number: '03', title: 'Multiple Revenue Streams', description: 'Diverse monetization options for AI creators', transform: 'rotateZ(8deg) rotateY(25deg)', glowDirection: 'right' }, // Increased rotateY
    { number: '04', title: 'Future-Proof Technology', description: 'Built on scalable, secure blockchain infrastructure', transform: 'rotateZ(-8deg) rotateY(-25deg)', glowDirection: 'left' }, // Increased rotateY
  ];

  return (
    <section
      className="relative flex items-center justify-center min-h-screen py-20 px-4 sm:px-6 lg:px-8" // Removed background classes and style
    >
      {/* Overlay to darken the background slightly if needed */}
      {/* <div className="absolute inset-0 bg-black opacity-30"></div> */}

      {/* Added perspective style to the grid container */}
      <div className="relative z-10 container mx-auto grid grid-cols-1 md:grid-cols-3 gap-8 items-center" style={{ perspective: '1000px' }}>

        {/* Left Feature Cards - Reduced vertical spacing - Apply transform style - Increased size */}
        <div className="space-y-4 md:space-y-8 flex flex-col items-center md:items-end">
          {/* Pass glowDirection prop - Add transformOrigin */}
          <AiafFeatureCard {...features[0]} className="w-72 md:w-80" style={{ transform: features[0].transform, transformOrigin: 'right center' }} glowDirection={features[0].glowDirection} />
          <AiafFeatureCard {...features[2]} className="w-72 md:w-80" style={{ transform: features[2].transform, transformOrigin: 'right center' }} glowDirection={features[2].glowDirection} />
        </div>

        {/* Center Video */}
        <div className="flex justify-center items-center order-first md:order-none">
          {/* Aspect ratio container for 9:16 */}
          <div className="relative w-[280px] sm:w-[320px] md:w-[360px]" style={{ paddingBottom: '177.78%' /* 16/9 * 100% */ }}>
             <video
               src={mainVideoUrl} // Use the generated URL
               autoPlay
               loop
               muted
               playsInline
               className="absolute top-0 left-0 w-full h-full object-contain" // Use object-contain to fit 9:16 video
             />
          </div>
        </div>


        {/* Right Content (Text + Cards) - Reduced vertical spacing */}
        <div className="space-y-4 md:space-y-8 flex flex-col items-center md:items-start">
           <h2 className="text-4xl md:text-5xl lg:text-6xl font-bold text-white text-center md:text-left mb-8 md:mb-0">
             Why Choose AIAF?
           </h2>
           {/* Apply transform style - Increased size - Pass glowDirection prop - Add transformOrigin */}
          <AiafFeatureCard {...features[1]} className="w-72 md:w-80" style={{ transform: features[1].transform, transformOrigin: 'left center' }} glowDirection={features[1].glowDirection} />
          <AiafFeatureCard {...features[3]} className="w-72 md:w-80" style={{ transform: features[3].transform, transformOrigin: 'left center' }} glowDirection={features[3].glowDirection} />
        </div>

      </div>
    </section>
  );
};

export default AiafVideoSection;
