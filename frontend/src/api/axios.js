import axios from 'axios';

// Replace with your actual Render backend URL
const BASE_URL = 'https://ai-retention-backend.onrender.com';

const api = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
});

export default api;
