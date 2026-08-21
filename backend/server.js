const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const { execFile } = require('child_process');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 5000;

// Ensure uploads directory exists
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

// Middleware
app.use(cors());
app.use(express.json());
// Serve the uploads folder so the frontend can access generated images
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Set up multer for file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/')
  },
  filename: function (req, file, cb) {
    // Add extension to the uploaded file for OpenCV/PIL to read easily
    const ext = path.extname(file.originalname) || '.jpg';
    cb(null, file.fieldname + '-' + Date.now() + ext)
  }
})
const upload = multer({ storage: storage });

// Basic health check route
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'Backend is running and ready for the model!' });
});

// Endpoint to receive the brain tumor image for prediction
app.post('/api/predict', upload.single('image'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No image uploaded' });
  }

  const imagePath = path.resolve(req.file.path);
  console.log('Received file for prediction:', imagePath);

  // Spawn Python process to run inference
  const pythonScript = path.join(__dirname, 'predict.py');
  
  execFile('python', [pythonScript, '--image_path', imagePath], (error, stdout, stderr) => {
    if (error) {
      console.error('Execution error:', error);
      console.error('stderr:', stderr);
      return res.status(500).json({ error: 'Failed to process image through model' });
    }

    try {
      // Find the JSON part in the stdout (ignoring any other prints from python)
      const lines = stdout.trim().split('\n');
      const jsonResponseStr = lines[lines.length - 1]; // We expect JSON to be the very last line
      const result = JSON.parse(jsonResponseStr);

      if (result.error) {
        return res.status(500).json({ error: result.error });
      }

      // Add the original image path so the frontend can display it if it wants
      result.original = `uploads/${req.file.filename}`;
      result.success = true;

      res.json(result);
    } catch (parseError) {
      console.error('Failed to parse Python output:', stdout);
      res.status(500).json({ error: 'Invalid response from model script' });
    }
  });
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
