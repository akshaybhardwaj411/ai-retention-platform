import api from './axios';

// Fetch dashboard metrics
export const getMetrics = async () => {
  try {
    // TODO: Build /analytics endpoint later; for now return mock data
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
    // For now, fetch all and filter or keep mock data until analytics is built
    const customers = await getAllCustomers();
    // Mock logic for demo: return first 3 with mock risks
    return customers.slice(0, 3).map((c, i) => ({
      id: c.id,
      name: c.name || 'Unknown',
      email: c.email,
      risk: [92, 88, 81][i % 3],
      reason: ['Month-to-month contract', 'High charges', 'No login for 40 days'][i % 3]
    }));
  } catch (error) {
    console.error('Error fetching high-risk customers:', error);
    return [];
  }
};
