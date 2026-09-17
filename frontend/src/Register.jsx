import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import './index.css';

const API = 'http://localhost:5000';

export default function Register({ onLogin }) {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!name || !email || !password || !confirm) return setError('Please fill in all fields.');
    if (password !== confirm) return setError('Passwords do not match.');
    if (password.length < 6) return setError('Password must be at least 6 characters.');

    setLoading(true);
    setError('');
    try {
      const res = await fetch(`${API}/api/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, email, password }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Registration failed');

      // Store JWT token
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
            <h1 className="hero-title" style={{ fontSize: '1.8rem', margin: 0 }}>Create Account</h1>
            <p className="hero-desc" style={{ fontSize: '0.9rem', marginTop: '0.5rem' }}>
              Register to access NeuroSight AI
            </p>
          </div>

          <form onSubmit={handleSubmit} className="login-form">
            {error && <div className="err-box" style={{ marginBottom: '1rem', padding: '0.75rem' }}>{error}</div>}
            
            <div className="form-group">
              <label>Full Name</label>
              <input 
                type="text" 
                placeholder="Dr. John Doe" 
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="form-input"
              />
            </div>

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
                placeholder="Min. 6 characters" 
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="form-input"
              />
            </div>

            <div className="form-group">
              <label>Confirm Password</label>
              <input 
                type="password" 
                placeholder="••••••••" 
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                className="form-input"
              />
            </div>

            <button type="submit" className="btn btn-cta" style={{ width: '100%', marginTop: '1rem' }} disabled={loading}>
              {loading ? 'Creating Account...' : 'Register'}
            </button>
            
            <div style={{ textAlign: 'center', marginTop: '1.5rem', fontSize: '0.9rem', color: '#a0a0b0' }}>
              Already have an account? <Link to="/login" style={{ color: '#a855f7', textDecoration: 'none', fontWeight: 'bold' }}>Sign In</Link>
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
