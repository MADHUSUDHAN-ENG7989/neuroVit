require('dotenv').config();
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const { execFile } = require('child_process');
const fs = require('fs');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const { DynamoDBDocumentClient, PutCommand, GetCommand, ScanCommand } = require('@aws-sdk/lib-dynamodb');
const { v4: uuidv4 } = (() => { try { return require('uuid'); } catch(e) { return { v4: () => Date.now().toString(36) + Math.random().toString(36).substr(2) }; } })();

const app = express();
const PORT = process.env.PORT || 5000;
const JWT_SECRET = process.env.JWT_SECRET || 'neurosight-secret-key-2026';

// ─── DynamoDB Client ────────────────────────────────────────────────────────
const ddbClient = new DynamoDBClient({
  region: process.env.AWS_REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
  },
});
const docClient = DynamoDBDocumentClient.from(ddbClient);

const USERS_TABLE = 'Users';
const LOGS_TABLE = 'QueryLogs';

// ─── Middleware ──────────────────────────────────────────────────────────────
app.use(cors());
app.use(express.json());

// Ensure uploads directory exists
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// ─── Auth Middleware ─────────────────────────────────────────────────────────
function authRequired(req, res, next) {
  const header = req.headers.authorization;
  if (!header) return res.status(401).json({ error: 'No token provided' });
  const token = header.split(' ')[1];
  try {
    req.user = jwt.verify(token, JWT_SECRET);
    next();
  } catch {
    res.status(401).json({ error: 'Invalid or expired token' });
  }
}

// ─── Multer ──────────────────────────────────────────────────────────────────
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, 'uploads/'),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname) || '.jpg';
    cb(null, file.fieldname + '-' + Date.now() + ext);
  },
});
const upload = multer({ storage });

// ─── Health Check ────────────────────────────────────────────────────────────
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'Backend is running and ready for the model!' });
});

// ─── Register ────────────────────────────────────────────────────────────────
app.post('/api/auth/register', async (req, res) => {
  const { email, password, name } = req.body;
  if (!email || !password || !name)
    return res.status(400).json({ error: 'Name, email and password are required.' });

  try {
    // Check if user already exists
    const existing = await docClient.send(new GetCommand({ TableName: USERS_TABLE, Key: { email } }));
    if (existing.Item) return res.status(409).json({ error: 'Email already registered.' });

    const passwordHash = await bcrypt.hash(password, 12);
    await docClient.send(new PutCommand({
      TableName: USERS_TABLE,
      Item: { email, name, passwordHash, createdAt: new Date().toISOString() },
    }));

    const token = jwt.sign({ email, name }, JWT_SECRET, { expiresIn: '2d' });
    res.status(201).json({ token, user: { email, name } });
  } catch (err) {
    console.error('Register error:', err);
    res.status(500).json({ error: 'Registration failed: ' + err.message });
  }
});

// ─── Login ───────────────────────────────────────────────────────────────────
app.post('/api/auth/login', async (req, res) => {
  const { email, password } = req.body;
  if (!email || !password)
    return res.status(400).json({ error: 'Email and password are required.' });

  try {
    const result = await docClient.send(new GetCommand({ TableName: USERS_TABLE, Key: { email } }));
    if (!result.Item) return res.status(401).json({ error: 'Invalid email or password.' });

    const valid = await bcrypt.compare(password, result.Item.passwordHash);
    if (!valid) return res.status(401).json({ error: 'Invalid email or password.' });

    const token = jwt.sign({ email, name: result.Item.name }, JWT_SECRET, { expiresIn: '2d' });
    res.json({ token, user: { email, name: result.Item.name } });
  } catch (err) {
    console.error('Login error:', err);
    res.status(500).json({ error: 'Login failed: ' + err.message });
  }
});

// ─── Predict ─────────────────────────────────────────────────────────────────
app.post('/api/predict', authRequired, upload.single('image'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No image uploaded' });

  const imagePath = path.resolve(req.file.path);
  console.log('Received file for prediction:', imagePath);

  const pythonScript = path.join(__dirname, 'predict.py');
  execFile('python', [pythonScript, '--image_path', imagePath], (error, stdout, stderr) => {
    if (error) {
      console.error('Execution error:', error);
      console.error('stderr:', stderr);
      return res.status(500).json({ error: 'Failed to process image through model' });
    }

    try {
      const lines = stdout.trim().split('\n');
      const result = JSON.parse(lines[lines.length - 1]);

      if (result.error) return res.status(500).json({ error: result.error });

      result.original = `uploads/${req.file.filename}`;
      result.success = true;

      // Log to DynamoDB
      const logEntry = {
        logId: `${Date.now()}-${Math.random().toString(36).substr(2, 6)}`,
        timestamp: new Date().toISOString(),
        filename: req.file.filename,
        prediction: result.prediction,
        confidence: String(result.confidence),
        userEmail: req.user.email,
      };

      docClient.send(new PutCommand({ TableName: LOGS_TABLE, Item: logEntry }))
        .then(() => console.log('Log saved to DynamoDB'))
        .catch(err => console.error('Failed to log to DynamoDB:', err.message));

      res.json(result);
    } catch (parseError) {
      console.error('Failed to parse Python output:', stdout);
      res.status(500).json({ error: 'Invalid response from model script' });
    }
  });
});

// ─── Get Logs ────────────────────────────────────────────────────────────────
app.get('/api/logs', authRequired, async (req, res) => {
  try {
    const result = await docClient.send(new ScanCommand({ TableName: LOGS_TABLE }));
    const logs = (result.Items || []).sort((a, b) => b.timestamp.localeCompare(a.timestamp));
    res.json({ logs });
  } catch (err) {
    console.error('Failed to fetch logs:', err);
    res.status(500).json({ error: 'Failed to fetch logs: ' + err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
