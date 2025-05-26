import React from 'react';
import { motion } from 'motion/react';

// Define a palette of creative/modern colors
const defaultColors = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#FED766', '#F0B7A4',
  '#9B5DE5', '#F15BB5', '#FEE440', '#00F5D4', '#00BBF9',
  '#F72585', '#7209B7', '#3A0CA3', '#4361EE', '#4CC9F0',
  '#FFBE0B', '#FB5607', '#FF006E', '#8338EC', '#3A86FF',
  '#EF476F', '#FFD166', '#06D6A0', '#118AB2', '#073B4C',
];

/**
 * An interactive color picker component using motion.
 * Displays a grid of color squares with hover and tap animations.
 *
 * @param {string} selectedColor - The currently selected hex color string.
 * @param {function(string): void} onChange - Callback function when a color is selected.
 * @param {string[]} [colors=defaultColors] - Optional array of hex color strings to display.
 */
function InteractiveColorPicker({ selectedColor, onChange, colors = defaultColors }) {
  return (
    <div className="grid grid-cols-5 gap-2 p-2 rounded-lg bg-gray-100 dark:bg-navy-700 border border-gray-200 dark:border-navy-600 max-w-[180px]">
      {colors.map((color) => (
        <motion.div
          key={color}
          className="w-6 h-6 rounded cursor-pointer border border-gray-300 dark:border-navy-500"
          style={{ backgroundColor: color }}
          onClick={() => onChange(color)}
          whileHover={{ scale: 1.2, zIndex: 1, transition: { duration: 0.1 } }}
          whileTap={{ scale: 0.9 }}
          animate={{
            scale: selectedColor === color ? 1.15 : 1,
            boxShadow: selectedColor === color ? `0 0 0 2px ${color}` : '0 0 0 0px rgba(0,0,0,0)',
            borderWidth: selectedColor === color ? '2px' : '1px',
            borderColor: selectedColor === color ? 'rgba(255, 255, 255, 0.8)' : 'rgba(150, 150, 150, 0.5)',
          }}
          transition={{ type: 'spring', stiffness: 400, damping: 15 }}
        />
      ))}
    </div>
  );
}

export default InteractiveColorPicker;