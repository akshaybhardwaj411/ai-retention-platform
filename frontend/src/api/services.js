import api from './axios';

// Fetch real-time analytics from backend
export const getAnalytics = async () => {
  try {
    const response = await api.get('/analytics');
    return response.data;
  } catch (error) {
    console.error('Error fetching analytics:', error);
    return null;
  }
};

// Fetch dashboard metrics (uses analytics now)
export const getMetrics = async () => {
  try {
    const data = await getAnalytics();
    if (data) {
      return {
        totalCustomers: data.totalCustomers || 0,
        activeCustomers: data.activeCustomers || 0,
        highRisk: data.highRisk || 0,
        retentionRate: data.retentionRate || 0,
        revenue: data.revenue || 0,
        revenueLoss: data.revenueLoss || 0,
        churnTrend: data.churnTrend || []
      };
    }
    // Fallback mock data
    return {
      totalCustomers: 10450,
      activeCustomers: 8920,
      highRisk: 540,
      retentionRate: 94.2,
      revenue: 1240000,
      revenueLoss: 25000,
      churnTrend: [
        { month: 'Apr', churn: 42 },
        { month: 'May', churn: 38 },
        { month: 'Jun', churn: 45 },
        { month: 'Jul', churn: 51 },
        { month: 'Aug', churn: 49 },
        { month: 'Sep', churn: 44 }
      ]
    };
  } catch (error) {
    console.error('Error fetching metrics:', error);
    return null;
  }
};

// Fetch all customers from the backend
export const getAllCustomers = async () => {
  try {
    const response = await api.get('/customers');
    return response.data.customers || [];
  } catch (error) {
    console.error('Error fetching customers:', error);
    return [];
  }
};

// Fetch a single customer by ID
export const getCustomerById = async (id) => {
  try {
    const response = await api.get(`/customers/${id}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching customer:', error);
    return null;
  }
};

// Fetch prediction history for a customer
export const getCustomerPredictions = async (id) => {
  try {
    const response = await api.get(`/customers/${id}/predictions`);
    return response.data.predictions || [];
  } catch (error) {
    console.error('Error fetching predictions:', error);
    return [];
  }
};

// Predict churn for a single customer
export const predictChurn = async (customerData) => {
  try {
    const response = await api.post('/predict', customerData);
    return response.data;
  } catch (error) {
    console.error('Error predicting churn:', error);
    throw error;
  }
};

// Predict and save to database
export const predictAndSave = async (customerData, customerName, customerEmail) => {
  try {
    const response = await api.post(
      `/predict-and-save?customer_name=${encodeURIComponent(customerName)}&customer_email=${encodeURIComponent(customerEmail)}`,
      customerData
    );
    return response.data;
  } catch (error) {
    console.error('Error predicting and saving:', error);
    throw error;
  }
};

// Fetch top high-risk customers
export const getHighRiskCustomers = async () => {
  try {
    const customers = await getAllCustomers();
    // Fetch predictions for each customer to calculate risk
    const highRiskList = [];
    for (const c of customers.slice(0, 10)) {
      const preds = await getCustomerPredictions(c.id);
      if (preds.length > 0) {
        const latest = preds[0];
        if (latest.churn_probability >= 60) {
          highRiskList.push({
            id: c.id,
            name: c.name || 'Unknown',
            email: c.email,
            risk: Math.round(latest.churn_probability),
            reason: latest.reasons ? JSON.parse(latest.reasons)[0] || 'AI detected risk' : 'AI detected risk'
          });
        }
      }
    }
    // Sort by risk descending and take top 5
    highRiskList.sort((a, b) => b.risk - a.risk);
    return highRiskList.slice(0, 5);
  } catch (error) {
    console.error('Error fetching high-risk customers:', error);
    // Fallback mock data
    return [
      { id: 1, name: 'Rahul Singh', email: 'rahul@email.com', risk: 92, reason: 'Month-to-month contract' },
      { id: 2, name: 'Vikram Mehta', email: 'vikram@email.com', risk: 88, reason: 'High charges' },
      { id: 3, name: 'Deepa Nair', email: 'deepa@email.com', risk: 81, reason: 'No login for 40 days' },
    ];
  }
};
