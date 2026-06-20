// Pre-wired for Phase 6 Android packaging. Inert until Capacitor is installed:
//   cd app && npm i -D @capacitor/cli && npm i @capacitor/core @capacitor/android
//   npm run build && npx cap add android && npx cap sync && npx cap open android
// The web build in dist/ is the exact source for the Android WebView (webDir below).
import type { CapacitorConfig } from '@capacitor/cli'

const config: CapacitorConfig = {
  appId: 'com.earthquake.forecast',
  appName: 'Earthquake Forecast',
  webDir: 'dist',
}

export default config
