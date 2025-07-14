import axios from 'axios';

const apiClient = axios.create({
    baseURL: 'https://your-api-url.com', // Replace with your API base URL
    timeout: 10000,
    headers: {
        'Content-Type': 'application/json',
    },
});

export default apiClient;