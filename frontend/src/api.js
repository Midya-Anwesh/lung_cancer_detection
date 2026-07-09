import axios from "axios";

// Use a relative base URL so the app works in both local dev and production.
// In production (Render), Django serves the React app AND the API, so /api/
// automatically points to the correct host.
// In local dev, set REACT_APP_API_URL=http://127.0.0.1:8000 in frontend/.env
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || (process.env.NODE_ENV === 'development' ? 'http://127.0.0.1:8000' : ''),
});

export default api;
