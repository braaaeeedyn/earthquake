import React from 'react'
import ReactDOM from 'react-dom/client'
import { Capacitor, registerPlugin } from '@capacitor/core'
import App from './App'
import '@fontsource-variable/literata/opsz.css'   // self-hosted serif for headings (no third-party request)
import './index.css'

// Android app: the system font-size setting becomes a whole-page zoom (QuakeNativePlugin turns off the WebView's
// text-only scaling, which overflowed the layout); the CSS breakpoints are container queries, so they follow it.
// ponytail: capped so the layout never gets narrower than 280 CSS px (a 2.0 font scale on a 360 dp phone gets ~1.29).
async function applyFontScale() {
  if (!Capacitor.isPluginAvailable('QuakeNative')) return
  const { fontScale } = await registerPlugin<{ info(): Promise<{ fontScale?: number }> }>('QuakeNative').info()
  if (fontScale) document.documentElement.style.zoom = String(Math.min(fontScale, window.innerWidth / 280))
}

applyFontScale().catch(() => {}).finally(() =>
  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  ),
)
