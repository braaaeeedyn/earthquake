// Device location — the native Capacitor Geolocation plugin in the app (which shows the real
// Android permission prompt), falling back to the browser API on web. navigator.geolocation
// alone does nothing in the Android webview, so the app needs the plugin path.

import { Capacitor } from '@capacitor/core'
import { Geolocation } from '@capacitor/geolocation'

// Ask for permission (prompting if needed) and resolve the device's {lat, lon}.
// Rejects if permission is denied or the fix can't be read.
export async function getMyLocation(): Promise<{ lat: number; lon: number }> {
  if (Capacitor.isNativePlatform()) {
    let perm = await Geolocation.checkPermissions()
    if (perm.location !== 'granted' && perm.coarseLocation !== 'granted') {
      perm = await Geolocation.requestPermissions({ permissions: ['location', 'coarseLocation'] })
    }
    if (perm.location !== 'granted' && perm.coarseLocation !== 'granted') {
      throw new Error('location permission denied')
    }
    const pos = await Geolocation.getCurrentPosition({ enableHighAccuracy: false, timeout: 15000 })
    return { lat: pos.coords.latitude, lon: pos.coords.longitude }
  }

  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) return reject(new Error('Geolocation unavailable'))
    navigator.geolocation.getCurrentPosition(
      (pos) => resolve({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
      (err) => reject(err),
    )
  })
}
