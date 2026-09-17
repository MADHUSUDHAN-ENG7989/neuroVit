import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import './index.css';

const API = 'http://localhost:5000';

export default function Login({ onLogin }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!email || !password) return setError('Please enter both email and password.');
    
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`${API}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Login failed');
      
      // Store JWT token (2-day expiry is set in token itself)
      localStorage.setItem('ns_token', data.token);
      localStorage.setItem('ns_user', JSON.stringify(data.user));
      onLogin(data.token, data.user);
      navigate('/');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page login-page anim-fade-in">
      <div className="login-container">
        <div className="login-card">
          <div className="login-header">
            <div className="nav-logo" style={{ fontSize: '3rem', marginBottom: '1rem' }}>🧠</div>
            <h1 className="hero-title" style={{ fontSize: '1.8rem', margin: 0 }}>NeuroSight AI</h1>
            <p className="hero-desc" style={{ fontSize: '0.9rem', marginTop: '0.5rem' }}>
              Sign in to access the diagnostic dashboard
            </p>
          </div>

          <form onSubmit={handleSubmit} className="login-form">
            {error && <div className="err-box" style={{ marginBottom: '1rem', padding: '0.75rem' }}>{error}</div>}
            
            <div className="form-group">
              <label>Email Address</label>
              <input 
                type="email" 
                placeholder="doctor@hospital.com" 
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="form-input"
              />
            </div>
            
            <div className="form-group">
              <label>Password</label>
              <input 
                type="password" 
                placeholder="••••••••" 
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="form-input"
              />
            </div>

            <button type="submit" className="btn btn-cta" style={{ width: '100%', marginTop: '1rem' }} disabled={loading}>
              {loading ? 'Signing In...' : 'Sign In'}
            </button>

            <div style={{ textAlign: 'center', marginTop: '1.5rem', fontSize: '0.9rem', color: '#a0a0b0' }}>
              Don't have an account? <Link to="/register" style={{ color: '#a855f7', textDecoration: 'none', fontWeight: 'bold' }}>Register Now</Link>
            </div>
          </form>
          
          <div className="login-footer">
            <p>Secure Medical Portal • Authorized Personnel Only</p>
          </div>
        </div>
      </div>
    </div>
  );
}
