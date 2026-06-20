import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base: './' keeps asset paths relative so the same build works both at a web
// root and inside the Capacitor Android WebView.
export default defineConfig({
  base: './',
  plugins: [react()],
  server: { port: 5173 },
})
