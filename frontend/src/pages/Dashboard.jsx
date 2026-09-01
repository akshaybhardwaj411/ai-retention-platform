import React, { useEffect, useState } from 'react';
import axios from '../api/axios';
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts';

const Dashboard = () => {
  const [metrics, setMetrics] = useState({
    totalCustomers: 0,
    activeCustomers: 0,
    highRisk: 0,
    retentionRate: 0,
    revenue: 0,
  });
  const [topRisky, setTopRisky] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch metrics and top risky customers from backend
    // For now we'll use mock data until we build the endpoints
    // In a real scenario, we'll call /analytics and /high-risk
    const mockMetrics = {
      totalCustomers: 10450,
      activeCustomers: 8920,
      highRisk: 540,
      retentionRate: 94.2,
      revenue: 1240000,
    };
    const mockRisky = [
      { id: 1, name: 'Rahul Singh', email: 'rahul@email.com', risk: 92, reason: 'Month-to-month contract' },
      { id: 2, name: 'Vikram Mehta', email: 'vikram@email.com', risk: 88, reason: 'High charges' },
      { id: 3, name: 'Deepa Nair', email: 'deepa@email.com', risk: 81, reason: 'No login for 40 days' },
    ];
    setMetrics(mockMetrics);
    setTopRisky(mockRisky);
    setLoading(false);
  }, []);

  // Sample churn trend data
  const churnData = [
    { month: 'Apr', churn: 42 },
    { month: 'May', churn: 38 },
    { month: 'Jun', churn: 45 },
    { month: 'Jul', churn: 51 },
    { month: 'Aug', churn: 49 },
    { month: 'Sep', churn: 44 },
  ];

  if (loading) return <div className="p-10 text-center">Loading dashboard...</div>;

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div className="bg-white p-4 rounded shadow">
          <p className="text-gray-500 text-sm">Total Customers</p>
          <p className="text-2xl font-bold">{metrics.totalCustomers}</p>
        </div>
        <div className="bg-white p-4 rounded shadow">
          <p className="text-gray-500 text-sm">Active</p>
          <p className="text-2xl font-bold text-green-600">{metrics.activeCustomers}</p>
        </div>
        <div className="bg-white p-4 rounded shadow">
          <p className="text-gray-500 text-sm">High Risk</p>
          <p className="text-2xl font-bold text-red-600">{metrics.highRisk}</p>
        </div>
        <div className="bg-white p-4 rounded shadow">
          <p className="text-gray-500 text-sm">Retention Rate</p>
          <p className="text-2xl font-bold text-blue-600">{metrics.retentionRate}%</p>
        </div>
      </div>

      {/* Chart */}
      <div className="bg-white p-4 rounded shadow mb-6">
        <h2 className="font-semibold mb-4">Monthly Churn Trend</h2>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={churnData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="churn" fill="#f97316" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Top High Risk Customers Table */}
      <div className="bg-white p-4 rounded shadow">
        <h2 className="font-semibold mb-4">🔥 Top High Risk Customers</h2>
        <table className="w-full text-left">
          <thead className="bg-gray-50">
            <tr>
              <th className="p-2">Name</th>
              <th className="p-2">Email</th>
              <th className="p-2">Risk</th>
              <th className="p-2">Reason</th>
            </tr>
          </thead>
          <tbody>
            {topRisky.map((c) => (
              <tr key={c.id} className="border-t">
                <td className="p-2 font-medium">{c.name}</td>
                <td className="p-2 text-gray-600">{c.email}</td>
                <td className="p-2">
                  <span className={`px-2 py-1 rounded text-white text-sm ${
                    c.risk > 80 ? 'bg-red-600' : c.risk > 60 ? 'bg-orange-500' : 'bg-yellow-500'
                  }`}>{c.risk}%</span>
                </td>
                <td className="p-2 text-gray-600">{c.reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Dashboard;
