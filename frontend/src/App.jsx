import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import CustomerProfile from './pages/CustomerProfile';
import Footer from './components/Footer';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-100 flex flex-col">
        <nav className="bg-white shadow px-6 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-indigo-600">🧠 AI Retention Platform</h1>
          <span className="text-sm text-gray-500">v1.0</span>
        </nav>
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/customer/:id" element={<CustomerProfile />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
