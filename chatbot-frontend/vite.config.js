import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
// Removed Tailwind Vite plugin import as it's not needed for v3

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()], // Removed Tailwind Vite plugin
  server: {
    proxy: {
      // Proxy requests starting with /api to the backend server
      '/api': {
        target: 'http://localhost:5001', // Your backend server address
        changeOrigin: true, // Needed for virtual hosted sites
        secure: false,      // Optional: Set to false if backend uses self-signed certificate
        // Optional: rewrite path if needed, e.g., remove /api prefix
        // rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
    // Ensure the dev server uses the correct port if needed (e.g., 5173 or 5174)
    // port: 5174, // Uncomment and set if you need a specific port
  },
})
