const fs = require('fs');

// Read the raw JavaScript files
const mainCode = fs.readFileSync('main.js', 'utf8');
const rendererCode = fs.readFileSync('renderer.js', 'utf8');

// Build the plugin object
const pluginTemplate = {
  id: "chess",
  name: "Chess Move Predictor",
  version: "1.0.0",
  description: "Train a neural network to predict chess moves. Upload UCI game files, then play against the AI for both sides.",
  mainCode: mainCode,
  rendererCode: rendererCode,
  author: "NeuralCabin"
};

// Convert to a properly escaped JSON string and save it
fs.writeFileSync('chess.nbpl', JSON.stringify(pluginTemplate, null, 2));
console.log('Successfully created chess.nbpl!');
