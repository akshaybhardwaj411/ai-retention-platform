import api from './axios';

// Fetch dashboard metrics
export const getMetrics = async () => {
  try {
    // For now, we'll fetch from /analytics endpoint
    // We need to build this endpoint in backend later
    // For now, return mock data until we build the /analytics endpoint
    return {
      totalCustomers: 10450,
      activeCustomers: 8920,
      highRisk: 540,
      retentionRate: 94.2,
      revenue: 1240000,
    };
  } catch (error) {
    console.error('Error fetching metrics:', error);
    return null;
  }
};

// Fetch top high-risk customers
export const getHighRiskCustomers = async () => {
  try {
    // We'll build a /high-risk endpoint later
    // For now, return mock data
    return [
      { id: 1, name: 'Rahul Singh', email: 'rahul@email.com', risk: 92, reason: 'Month-to-month contract' },
      { id: 2, name: 'Vikram Mehta', email: 'vikram@email.com', risk: 88, reason: 'High charges' },
      { id: 3, name: 'Deepa Nair', email: 'deepa@email.com', risk: 81, reason: 'No login for 40 days' },
    ];
  } catch (error) {
    console.error('Error fetching high-risk customers:', error);
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

// Fetch customer by ID
export const getCustomerById = async (id) => {
  try {
    const response = await api.get(`/customers/${id}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching customer:', error);
    return null;
  }
};
