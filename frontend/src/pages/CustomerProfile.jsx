import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getCustomerById, getCustomerPredictions, predictChurn } from '../api/services';

const CustomerProfile = () => {
  const { id } = useParams();
  const [customer, setCustomer] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [error, setError] = useState(null);

  // State to edit customer data for prediction
  const [editData, setEditData] = useState({
    gender: 'Male',
    SeniorCitizen: 0,
    Partner: 'No',
    Dependents: 'No',
    tenure: 0,
    PhoneService: 'No',
    InternetService: 'No',
    Contract: 'Month-to-month',
    PaymentMethod: 'Electronic check',
    MonthlyCharges: 0,
    TotalCharges: 0
  });

  useEffect(() => {
    const fetchCustomerData = async () => {
      try {
        setLoading(true);
        const customerData = await getCustomerById(id);
        if (customerData) {
          setCustomer(customerData);
          // Pre-fill edit data with customer data
          setEditData({
            gender: customerData.gender || 'Male',
            SeniorCitizen: customerData.senior_citizen || 0,
            Partner: customerData.partner || 'No',
            Dependents: customerData.dependents || 'No',
            tenure: customerData.tenure || 0,
            PhoneService: customerData.phone_service || 'No',
            InternetService: customerData.internet_service || 'No',
            Contract: customerData.contract || 'Month-to-month',
            PaymentMethod: customerData.payment_method || 'Electronic check',
            MonthlyCharges: customerData.monthly_charges || 0,
            TotalCharges: customerData.total_charges || 0
          });
        } else {
          setError('Customer not found');
        }

        const preds = await getCustomerPredictions(id);
        setPredictions(preds);
      } catch (err) {
        console.error('Error fetching profile:', err);
        setError('Failed to load customer profile');
      } finally {
        setLoading(false);
      }
    };

    fetchCustomerData();
  }, [id]);

  const handlePredict = async () => {
    try {
      setPredicting(true);
      setPredictionResult(null);
      const result = await predictChurn(editData);
      setPredictionResult(result);
    } catch (err) {
      console.error('Prediction error:', err);
      alert('Failed to get prediction. Is your backend running?');
    } finally {
      setPredicting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-gray-500">Loading customer profile...</div>
      </div>
    );
  }

  if (error || !customer) {
    return (
      <div className="p-6 text-red-600 bg-red-50 rounded-lg max-w-4xl mx-auto mt-6">
        ❌ {error || 'Customer not found'}
        <div className="mt-4">
          <Link to="/" className="text-indigo-600 hover:underline">← Back to Dashboard</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-6xl mx-auto">
      {/* Header & Back Button */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <Link to="/" className="text-indigo-600 hover:underline text-sm">← Back to Dashboard</Link>
          <h1 className="text-2xl font-bold mt-1">👤 {customer.name || 'Customer Profile'}</h1>
        </div>
        <div className="text-sm text-gray-500">ID: #{customer.id}</div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Customer Details */}
        <div className="lg:col-span-1 bg-white p-4 rounded-lg shadow-md">
          <h2 className="font-semibold text-lg border-b pb-2 mb-4">📋 Details</h2>
          <div className="space-y-2 text-sm">
            <p><span className="font-medium">Email:</span> {customer.email}</p>
            <p><span className="font-medium">Gender:</span> {customer.gender}</p>
            <p><span className="font-medium">Senior Citizen:</span> {customer.senior_citizen ? 'Yes' : 'No'}</p>
            <p><span className="font-medium">Partner:</span> {customer.partner}</p>
            <p><span className="font-medium">Dependents:</span> {customer.dependents}</p>
            <p><span className="font-medium">Tenure:</span> {customer.tenure} months</p>
            <p><span className="font-medium">Contract:</span> {customer.contract}</p>
            <p><span className="font-medium">Payment:</span> {customer.payment_method}</p>
            <p><span className="font-medium">Monthly:</span> ₹{customer.monthly_charges}</p>
            <p><span className="font-medium">Total:</span> ₹{customer.total_charges}</p>
            <p><span className="font-medium">Internet:</span> {customer.internet_service}</p>
          </div>
        </div>

        {/* Right Column: AI Prediction + History */}
        <div className="lg:col-span-2 space-y-6">
          {/* AI Prediction Tester */}
          <div className="bg-white p-4 rounded-lg shadow-md">
            <h2 className="font-semibold text-lg mb-4">🧠 AI Prediction for this Customer</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-4 text-sm">
              <input 
                className="border rounded p-2"
                placeholder="Tenure"
                type="number"
                value={editData.tenure}
                onChange={(e) => setEditData({...editData, tenure: parseInt(e.target.value) || 0})}
              />
              <input 
                className="border rounded p-2"
                placeholder="Monthly Charges"
                type="number"
                value={editData.MonthlyCharges}
                onChange={(e) => setEditData({...editData, MonthlyCharges: parseFloat(e.target.value) || 0})}
              />
              <select 
                className="border rounded p-2"
                value={editData.Contract}
                onChange={(e) => setEditData({...editData, Contract: e.target.value})}
              >
                <option value="Month-to-month">Month-to-month</option>
                <option value="One year">One year</option>
                <option value="Two year">Two year</option>
              </select>
              <select 
                className="border rounded p-2"
                value={editData.InternetService}
                onChange={(e) => setEditData({...editData, InternetService: e.target.value})}
              >
                <option value="No">No Internet</option>
                <option value="DSL">DSL</option>
                <option value="Fiber optic">Fiber optic</option>
              </select>
              <select 
                className="border rounded p-2"
                value={editData.PaymentMethod}
                onChange={(e) => setEditData({...editData, PaymentMethod: e.target.value})}
              >
                <option value="Electronic check">Electronic check</option>
                <option value="Mailed check">Mailed check</option>
                <option value="Bank transfer (automatic)">Bank transfer</option>
                <option value="Credit card (automatic)">Credit card</option>
              </select>
              <button 
                onClick={handlePredict}
                disabled={predicting}
                className="bg-indigo-600 text-white px-4 py-2 rounded hover:bg-indigo-700 transition disabled:opacity-50"
              >
                {predicting ? 'Predicting...' : '🔮 Run Prediction'}
              </button>
            </div>
            
            {/* Prediction Result */}
            {predictionResult && (
              <div className={`p-4 rounded border ${predictionResult.risk_level === 'Low Risk' ? 'bg-green-50 border-green-200' : predictionResult.risk_level === 'Medium Risk' ? 'bg-yellow-50 border-yellow-200' : 'bg-red-50 border-red-200'}`}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-lg font-bold">{predictionResult.risk_color} {predictionResult.risk_level}</span>
                  <span className="text-2xl font-bold">{predictionResult.probability}%</span>
                </div>
                <p><strong>Reasons:</strong> {predictionResult.reasons?.join(', ')}</p>
                <p className="mt-2"><strong>💡 Recommendation:</strong> {predictionResult.recommendation}</p>
              </div>
            )}
          </div>

          {/* Prediction History */}
          <div className="bg-white p-4 rounded-lg shadow-md">
            <h2 className="font-semibold text-lg border-b pb-2 mb-4">📊 Prediction History</h2>
            {predictions.length === 0 ? (
              <p className="text-gray-500 text-sm">No predictions saved for this customer yet.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="p-2">Date</th>
                      <th className="p-2">Probability</th>
                      <th className="p-2">Risk</th>
                      <th className="p-2">Recommendation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {predictions.map((p) => (
                      <tr key={p.id} className="border-t">
                        <td className="p-2">{new Date(p.predicted_at).toLocaleDateString()}</td>
                        <td className="p-2 font-medium">{p.churn_probability}%</td>
                        <td className="p-2">
                          <span className={`px-2 py-1 rounded text-white text-xs ${
                            p.risk_level === 'Low Risk' ? 'bg-green-500' : 
                            p.risk_level === 'Medium Risk' ? 'bg-yellow-500' : 'bg-red-500'
                          }`}>{p.risk_level}</span>
                        </td>
                        <td className="p-2 text-gray-600">{p.recommendation}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CustomerProfile;
