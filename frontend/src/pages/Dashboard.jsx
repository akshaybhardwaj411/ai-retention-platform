import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { getMetrics, getHighRiskCustomers } from '../api/services';
import api from '../api/axios';
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts';

const Dashboard = () => {
  const [metrics, setMetrics] = useState({
    totalCustomers: 0,
    activeCustomers: 0,
    highRisk: 0,
    retentionRate: 0,
    revenue: 0,
    revenueLoss: 0,
    churnTrend: []
  });
  const [topRisky, setTopRisky] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [testCustomer, setTestCustomer] = useState({
    gender: 'Male',
    SeniorCitizen: 0,
    Partner: 'Yes',
    Dependents: 'No',
    tenure: 12,
    PhoneService: 'Yes',
    InternetService: 'Fiber optic',
    Contract: 'Month-to-month',
    PaymentMethod: 'Electronic check',
    MonthlyCharges: 85.5,
    TotalCharges: 1020.5
  });
  const [predictionResult, setPredictionResult] = useState(null);
  const [predicting, setPredicting] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [metricsData, riskyData] = await Promise.all([
          getMetrics(),
          getHighRiskCustomers()
        ]);
        
        if (metricsData) setMetrics(metricsData);
        if (riskyData) setTopRisky(riskyData);
        setError(null);
      } catch (err) {
        console.error('Dashboard error:', err);
        setError('Failed to load dashboard data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const handlePredict = async () => {
    try {
      setPredicting(true);
      const response = await api.post('/predict', testCustomer);
      setPredictionResult(response.data);
    } catch (error) {
      console.error('Prediction error:', error);
      alert('Failed to get prediction. Is your backend running?');
    } finally {
      setPredicting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-gray-500">Loading dashboard...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 text-red-600 bg-red-50 rounded-lg">
        ❌ {error}
        <p className="text-sm mt-2">Please make sure your backend is running on Render.</p>
      </div>
    );
  }

  // Use real churn trend data from analytics, or fallback
  const churnData = metrics.churnTrend && metrics.churnTrend.length > 0 
    ? metrics.churnTrend 
    : [
        { month: 'Apr', churn: 42 },
        { month: 'May', churn: 38 },
        { month: 'Jun', churn: 45 },
        { month: 'Jul', churn: 51 },
        { month: 'Aug', churn: 49 },
        { month: 'Sep', churn: 44 }
      ];

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Alerts Panel (NEW) - Shows Critical Risks */}
      {topRisky.some(c => c.risk >= 80) && (
        <div className="bg-red-50 border-l-4 border-red-600 p-4 mb-6 rounded">
          <div className="flex items-center">
            <span className="text-2xl mr-3">🚨</span>
            <div>
              <p className="font-bold text-red-700">Critical Risk Alert!</p>
              <p className="text-sm text-red-600">
                {topRisky.filter(c => c.risk >= 80).length} customers are at critical risk (80%+ churn probability). 
                Immediate action recommended.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
        <div className="bg-white p-4 rounded-lg shadow-md border-l-4 border-blue-500">
          <p className="text-gray-500 text-sm">Total Customers</p>
          <p className="text-2xl font-bold">{metrics.totalCustomers.toLocaleString()}</p>
        </div>
        <div className="bg-white p-4 rounded-lg shadow-md border-l-4 border-green-500">
          <p className="text-gray-500 text-sm">Active Customers</p>
          <p className="text-2xl font-bold text-green-600">{metrics.activeCustomers.toLocaleString()}</p>
        </div>
        <div className="bg-white p-4 rounded-lg shadow-md border-l-4 border-red-500">
          <p className="text-gray-500 text-sm">High Risk</p>
          <p className="text-2xl font-bold text-red-600">{metrics.highRisk}</p>
        </div>
        <div className="bg-white p-4 rounded-lg shadow-md border-l-4 border-purple-500">
          <p className="text-gray-500 text-sm">Retention Rate</p>
          <p className="text-2xl font-bold text-purple-600">{metrics.retentionRate}%</p>
        </div>
        <div className="bg-white p-4 rounded-lg shadow-md border-l-4 border-orange-500">
          <p className="text-gray-500 text-sm">Revenue Loss (Est.)</p>
          <p className="text-2xl font-bold text-orange-600">₹{metrics.revenueLoss?.toLocaleString() || 0}</p>
        </div>
      </div>

      {/* AI Prediction Tester */}
      <div className="bg-white p-4 rounded-lg shadow-md mb-6">
        <h2 className="font-semibold text-lg mb-4">🧪 Test AI Prediction</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          <input 
            className="border rounded p-2 text-sm"
            placeholder="Tenure"
            type="number"
            value={testCustomer.tenure}
            onChange={(e) => setTestCustomer({...testCustomer, tenure: parseInt(e.target.value) || 0})}
          />
          <input 
            className="border rounded p-2 text-sm"
            placeholder="Monthly Charges"
            type="number"
            value={testCustomer.MonthlyCharges}
            onChange={(e) => setTestCustomer({...testCustomer, MonthlyCharges: parseFloat(e.target.value) || 0})}
          />
          <select 
            className="border rounded p-2 text-sm"
            value={testCustomer.Contract}
            onChange={(e) => setTestCustomer({...testCustomer, Contract: e.target.value})}
          >
            <option value="Month-to-month">Month-to-month</option>
            <option value="One year">One year</option>
            <option value="Two year">Two year</option>
          </select>
          <button 
            onClick={handlePredict}
            disabled={predicting}
            className="bg-indigo-600 text-white px-4 py-2 rounded hover:bg-indigo-700 transition disabled:opacity-50"
          >
            {predicting ? 'Predicting...' : '🔮 Predict Churn'}
          </button>
        </div>
        {predictionResult && (
          <div className={`p-4 rounded ${predictionResult.risk_level === 'Low Risk' ? 'bg-green-50' : predictionResult.risk_level === 'Medium Risk' ? 'bg-yellow-50' : 'bg-red-50'}`}>
            <p><strong>Probability:</strong> {predictionResult.probability}%</p>
            <p><strong>Risk Level:</strong> <span className="font-bold">{predictionResult.risk_color} {predictionResult.risk_level}</span></p>
            <p><strong>Reasons:</strong> {predictionResult.reasons?.join(', ')}</p>
            <p><strong>Recommendation:</strong> {predictionResult.recommendation}</p>
          </div>
        )}
      </div>

      {/* Chart - Now uses real data */}
      <div className="bg-white p-4 rounded-lg shadow-md mb-6">
        <h2 className="font-semibold text-lg mb-4">📈 Monthly Churn Trend</h2>
        {churnData.length === 0 ? (
          <p className="text-gray-500">No churn data available yet.</p>
        ) : (
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={churnData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="churn" fill="#f97316" />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Top High Risk Customers with Clickable Links */}
      <div className="bg-white p-4 rounded-lg shadow-md">
        <h2 className="font-semibold text-lg mb-4">🔥 Top High Risk Customers</h2>
        {topRisky.length === 0 ? (
          <p className="text-gray-500">No high-risk customers detected.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead className="bg-gray-50">
                <tr>
                  <th className="p-3 text-sm font-semibold text-gray-600">Name</th>
                  <th className="p-3 text-sm font-semibold text-gray-600">Email</th>
                  <th className="p-3 text-sm font-semibold text-gray-600">Risk</th>
                  <th className="p-3 text-sm font-semibold text-gray-600">Reason</th>
                </tr>
              </thead>
              <tbody>
                {topRisky.map((c) => (
                  <tr key={c.id} className="border-t hover:bg-gray-50 transition">
                    <td className="p-3 font-medium">
                      <Link to={`/customer/${c.id}`} className="text-indigo-600 hover:underline">
                        {c.name}
                      </Link>
                    </td>
                    <td className="p-3 text-gray-600">{c.email}</td>
                    <td className="p-3">
                      <span className={`px-3 py-1 rounded-full text-white text-sm font-medium ${
                        c.risk > 80 ? 'bg-red-600' : c.risk > 60 ? 'bg-orange-500' : 'bg-yellow-500'
                      }`}>
                        {c.risk}%
                      </span>
                    </td>
                    <td className="p-3 text-gray-600">{c.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
