import React from 'react';

// Basic Card component - provides container styling
// Based on common patterns in Horizon UI and Tailwind usage
// Updated to forward refs and apply glass effect
const Card = React.forwardRef((props, ref) => {
  const { variant, extra, children, ...rest } = props;
  return (
    <div
      ref={ref} // Attach the forwarded ref here
      // Increased blur and added subtle border for enhanced glass effect
      className={`!z-5 relative flex flex-col rounded-[20px] border border-white/10 bg-white bg-opacity-30 backdrop-blur-xl bg-clip-border shadow-3xl shadow-shadow-500 dark:!bg-navy-800 dark:border-white/5 dark:bg-opacity-30 dark:text-white dark:shadow-none ${extra}`}
      {...rest}
    >
      {children}
    </div>
  );
});

export default Card;
