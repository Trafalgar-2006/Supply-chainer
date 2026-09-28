import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// When the API requires a key (SUPPLYCHAINER_API_KEY), the dev server adds it to
// proxied requests, so the key never reaches the browser.
const apiKey = process.env.SUPPLYCHAINER_API_KEY

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        headers: apiKey ? { 'X-API-Key': apiKey } : {},
      },
      '/ws': {
        target: 'ws://127.0.0.1:8000',
        ws: true
      }
    }
  }
})
