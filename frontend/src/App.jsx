import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import CustomerProfile from './pages/CustomerProfile';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        <nav className="bg-white shadow px-6 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-indigo-600">🧠 AI Retention Platform</h1>
          <span className="text-sm text-gray-500">v1.0</span>
        </nav>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/customer/:id" element={<CustomerProfile />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
