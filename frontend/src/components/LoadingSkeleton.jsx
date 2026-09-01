import React from 'react';

const LoadingSkeleton = () => {
  return (
    <div className="animate-pulse">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="bg-gray-200 h-24 rounded-lg"></div>
        ))}
      </div>
      <div className="bg-gray-200 h-64 rounded-lg mb-6"></div>
      <div className="bg-gray-200 h-48 rounded-lg"></div>
    </div>
  );
};

export default LoadingSkeleton;
