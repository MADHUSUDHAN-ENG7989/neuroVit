import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation, Navigate } from 'react-router-dom';
import Home from './Home';
import About from './About';
import Login from './Login';
import Register from './Register';
import Logs from './Logs';
import './index.css';

const API = 'http://localhost:5000';

/* ── Navbar ── */
function Navbar({ online, user, onLogout }) {
  const location = useLocation();
  
  if (location.pathname === '/login' || location.pathname === '/register') return null;

  return (
    <nav className="navbar">
      <div className="nav-brand">
        <div className="nav-logo">🧠</div>
        <div>
          <div className="nav-name">NeuroSight AI</div>
          <div className="nav-tag">Brain Tumor Classification</div>
        </div>
      </div>
      
      <div className="nav-links">
        <Link to="/" className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}>Home</Link>
        <Link to="/about" className={`nav-link ${location.pathname === '/about' ? 'active' : ''}`}>Model Efficiency</Link>
        <Link to="/logs" className={`nav-link ${location.pathname === '/logs' ? 'active' : ''}`}>Logs</Link>
      </div>

      <div className="nav-right">
        <span className="pill pill-violet">PDSCNN + ViT + RRELM</span>
        <div className={`pill ${online ? 'pill-green' : 'pill-red'}`}>
          <span className="status-dot" />
          {online === null ? 'Connecting…' : online ? 'Backend Online' : 'Backend Offline'}
        </div>
        {user && (
          <span style={{ fontSize: '0.8rem', color: '#a0a0b0', marginLeft: '0.5rem' }}>
            👤 {user.name}
          </span>
        )}
        {user && (
          <button onClick={onLogout} className="btn btn-outline" style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem', marginLeft: '0.5rem' }}>
            Logout
          </button>
        )}
      </div>
    </nav>
  );
}

/* ── JWT helper ── */
function getStoredAuth() {
  try {
    const token = localStorage.getItem('ns_token');
    const user = localStorage.getItem('ns_user');
    if (!token || !user) return { token: null, user: null };
    // Decode JWT payload to check expiry (no library needed)
    const payload = JSON.parse(atob(token.split('.')[1]));
    if (payload.exp * 1000 < Date.now()) {
      localStorage.removeItem('ns_token');
      localStorage.removeItem('ns_user');
      return { token: null, user: null };
    }
    return { token, user: JSON.parse(user) };
  } catch {
    return { token: null, user: null };
  }
}

export default function App() {
  const [online, setOnline] = useState(null);
  const [auth, setAuth] = useState(getStoredAuth);

  useEffect(() => {
    fetch(`${API}/api/health`)
      .then(r => setOnline(r.ok))
      .catch(() => setOnline(false));
  }, []);

  const handleLogin = (token, user) => {
    localStorage.setItem('ns_token', token);
    localStorage.setItem('ns_user', JSON.stringify(user));
    setAuth({ token, user });
  };

  const handleLogout = () => {
    localStorage.removeItem('ns_token');
    localStorage.removeItem('ns_user');
    setAuth({ token: null, user: null });
  };

  const isLoggedIn = !!auth.token;

  return (
    <Router>
      <div className="bg-mesh"><div className="bg-grid" /></div>
      <Navbar online={online} user={auth.user} onLogout={handleLogout} />
      
      <Routes>
        <Route path="/login" element={<Login onLogin={handleLogin} />} />
        <Route path="/register" element={<Register onLogin={handleLogin} />} />
        <Route 
          path="/" 
          element={isLoggedIn ? <Home API={API} token={auth.token} /> : <Navigate to="/login" />} 
        />
        <Route 
          path="/about" 
          element={isLoggedIn ? <About /> : <Navigate to="/login" />} 
        />
        <Route 
          path="/logs" 
          element={isLoggedIn ? <Logs API={API} token={auth.token} /> : <Navigate to="/login" />} 
        />
      </Routes>

      <footer className="footer">
        <span className="footer-l">NeuroSight AI — For research purposes only. Not a medical device.</span>
        <span className="footer-r">v2.0 · PDSCNN + ViT + RRELM</span>
      </footer>
    </Router>
  );
}
