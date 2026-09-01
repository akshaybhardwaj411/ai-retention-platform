import axios from 'axios';

// CORRECT URL: Your Render backend base URL
const BASE_URL = 'https://ai-retention-platform.onrender.com';

const api = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
});

export default api;
