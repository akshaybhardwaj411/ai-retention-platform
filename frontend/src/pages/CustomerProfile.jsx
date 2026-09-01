import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import axios from '../api/axios';

const CustomerProfile = () => {
  const { id } = useParams();
  const [customer, setCustomer] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch customer details and prediction
    // We'll implement this later when we have API endpoints
    // For now, display a placeholder
    setLoading(false);
  }, [id]);

  if (loading) return <div className="p-6">Loading customer profile...</div>;
  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold">Customer Profile #{id}</h1>
      <p className="text-gray-500">Coming soon – we'll show AI predictions here!</p>
    </div>
  );
};

export default CustomerProfile;
