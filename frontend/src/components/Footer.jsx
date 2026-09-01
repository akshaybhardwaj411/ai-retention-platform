import React from 'react';

const Footer = () => {
  return (
    <footer className="bg-white border-t border-gray-200 py-4 px-6 mt-8">
      <div className="max-w-7xl mx-auto flex justify-between items-center text-sm text-gray-500">
        <div>
          🧠 <span className="font-medium">AI Retention Platform</span> v1.0
        </div>
        <div>
          Built with ❤️ using FastAPI, React, and Supabase
        </div>
      </div>
    </footer>
  );
};

export default Footer;
