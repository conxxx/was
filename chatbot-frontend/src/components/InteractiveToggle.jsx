import React from 'react';
import { motion } from 'motion/react';

/**
 * An interactive toggle switch component using motion for animation.
 *
 * @param {boolean} isOn - The current state of the toggle (true for on, false for off).
 * @param {function(): void} handleToggle - Callback function when the toggle is clicked.
 * @param {string} [labelId] - Optional ID to associate with a label.
 */
function InteractiveToggle({ isOn, handleToggle, labelId }) {
  const spring = {
    type: "spring",
    stiffness: 700,
    damping: 30
  };

  return (
    <motion.div
      className={`flex items-center w-12 h-6 rounded-full p-1 cursor-pointer ${isOn ? 'bg-brand-500 justify-end' : 'bg-gray-300 dark:bg-navy-600 justify-start'}`}
      onClick={handleToggle}
      layout // Animate layout changes
      transition={spring}
      aria-checked={isOn}
      role="switch"
      aria-labelledby={labelId}
      style={{
        boxShadow: isOn ? 'inset 0 1px 2px rgba(0,0,0,0.2)' : 'inset 0 1px 2px rgba(0,0,0,0.1)',
      }}
    >
      <motion.div
        className="w-4 h-4 bg-white rounded-full shadow-md"
        layout // Animate the knob's layout
        transition={spring}
      />
    </motion.div>
  );
}

export default InteractiveToggle;