import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/chat':          'http://localhost:8000',
      '/conversations': 'http://localhost:8000',
      '/health':        'http://localhost:8000',
      '/training':      'http://localhost:8000',   // fine-tuning data management endpoints
      // Held-out evaluation: both arms, scoring, and the promotion gate.
      // Streams SSE, so buffering must stay off -- http-proxy passes the
      // chunks straight through, which is what the step-by-step progress needs.
      '/raft':          'http://localhost:8000',
    }
  }
})
