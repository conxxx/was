const path = require('path');
const CopyPlugin = require('copy-webpack-plugin'); // Import the plugin

module.exports = {
  entry: './src/index.js', // Entry point of our application
  output: {
    filename: 'widget.js', // The name of the bundled file
    path: path.resolve(__dirname, 'dist'), // Output directory (we'll create a 'dist' folder)
    // Clean the output directory before each build
    clean: true,
  },
  mode: 'production', // Or 'development' for easier debugging
  // Add module rules if needed (e.g., for Babel transpilation, CSS loaders)
  module: {
    rules: [
      // Add rule for CSS files
      {
        test: /\.css$/i, // Regex to match .css files
        use: ['style-loader', 'css-loader'], // Loaders to use
      },
      // Example: Babel loader for older browser compatibility
      // {
      //   test: /\.js$/,
      //   exclude: /node_modules/,
      //   use: {
      //     loader: 'babel-loader',
      //     options: {
      //       presets: ['@babel/preset-env']
      //     }
      //   }
      // }
    ]
  },
  // Add plugins if needed (e.g., HTMLWebpackPlugin, MiniCssExtractPlugin)
  plugins: [
    new CopyPlugin({
      patterns: [
        { from: '../chatbot-frontend/src/vad-processor.js', to: '.' }, // Copy VAD processor from frontend project
      ],
    }),
  ],
  // Configure development server if needed
  // devServer: {
  //   static: './dist',
  // },
  // Control source maps
  devtool: 'source-map', // Or false for production, or other options
};
