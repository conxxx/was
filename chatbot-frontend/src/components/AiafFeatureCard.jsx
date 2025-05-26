import React, { useState } from 'react'; // Import useState

const AiafFeatureCard = ({ number, title, description, className = '', style = {}, glowDirection = 'right' }) => {

  const [isHovered, setIsHovered] = useState(false); // State for hover effect

  // Determine gradient based on glowDirection
  const gradientClass = glowDirection === 'left'
    ? "bg-gradient-to-l from-[rgba(255,255,255,0.1)] via-[rgba(255,255,255,0.05)] to-pink-500/30" // Glow left
    : "bg-gradient-to-r from-[rgba(255,255,255,0.1)] via-[rgba(255,255,255,0.05)] to-pink-500/30"; // Glow right (default)

  // Base styles for the card
  const baseStyles = `${gradientClass} backdrop-blur-sm border border-[rgba(255,255,255,0.2)] rounded-2xl p-6 text-white shadow-lg shadow-pink-500/20`;

  // Mask style for fading edges - Very minimal fade intensity
  const maskStyle = {
    maskImage: 'linear-gradient(to right, transparent, black 2%, black 98%, transparent), linear-gradient(to bottom, transparent, black 2%, black 98%, transparent)', // Very minimal fade
    maskComposite: 'intersect',
    WebkitMaskImage: 'linear-gradient(to right, transparent, black 2%, black 98%, transparent), linear-gradient(to bottom, transparent, black 2%, black 98%, transparent)', // Very minimal fade
    WebkitMaskComposite: 'source-in',
  };

  // Base transform classes + hover effect
  const transformClasses = "transform transition-transform duration-300 hover:scale-105 origin-center"; // Added hover scale

  // Rotation styles - apply based on position later or via className prop
  // Example rotations (adjust as needed):
  // Added transform-origin
  // Card 1 & 3: rotate-[-6deg]
  // Card 2 & 4: rotate-[6deg]

  // Combine base transform from style prop with hover adjustments
  const dynamicTransform = isHovered
    ? `${style.transform || ''} scale(1.05) translateZ(10px)` // Scale up and lift slightly on hover
    : style.transform || ''; // Use base transform when not hovered

  return (
    // Merged base styles, className, transform classes, dynamic transform style, and maskStyle
    // Added mouse enter/leave handlers
    <div
      className={`${baseStyles} ${className} ${transformClasses}`}
      style={{ ...maskStyle, transform: dynamicTransform }} // Apply dynamic transform here, remove from style prop merge
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="text-pink-400 text-3xl font-bold mb-4">{number}</div>
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <p className="text-sm text-gray-300">{description}</p>
    </div>
  );
};

export default AiafFeatureCard;
