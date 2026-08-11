// Mobile push registration (Capacitor + FCM). No-op on web — the browser build has no native
// push, so enablePush() returns null there and the email path is the only alert channel.

import { Capacitor, type PluginListenerHandle } from '@capacitor/core'
import { PushNotifications } from '@capacitor/push-notifications'
import { registerPushToken } from './nearme'

export function isNativeApp() {
  return Capacitor.isNativePlatform()
}

// One-shot: register with FCM and resolve the device token (rejects on registrationError).
function awaitToken(): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    let reg: PluginListenerHandle | undefined
    let err: PluginListenerHandle | undefined
    const cleanup = () => { reg?.remove(); err?.remove() }
    PushNotifications.addListener('registration', (t) => { cleanup(); resolve(t.value) })
      .then((h) => { reg = h })
    PushNotifications.addListener('registrationError', (e) => {
      cleanup(); reject(new Error(String(e.error ?? 'registration failed')))
    }).then((h) => { err = h })
    PushNotifications.register().catch(reject)
  })
}

// Request permission, register for push, and send {token, location} to the server.
// Returns a status string, or null on web (where push isn't available).
export async function enablePush(loc: { name: string; lat: number; lon: number }): Promise<string | null> {
  if (!Capacitor.isNativePlatform()) return null

  let perm = await PushNotifications.checkPermissions()
  if (perm.receive !== 'granted') perm = await PushNotifications.requestPermissions()
  if (perm.receive !== 'granted') throw new Error('notification permission denied')

  const token = await awaitToken()
  await registerPushToken({ token, lat: loc.lat, lon: loc.lon, name: loc.name })
  return 'Push alerts enabled on this device.'
}

// Foreground handlers so a notification that arrives while the app is open isn't silently
// dropped. Call once at app start. Safe (no-op) on web.
export async function initPush() {
  if (!Capacitor.isNativePlatform()) return
  await PushNotifications.addListener('pushNotificationReceived', (n) => {
    console.log('[push] received in foreground:', n.title, n.body)
  })
  await PushNotifications.addListener('pushNotificationActionPerformed', (a) => {
    console.log('[push] tapped:', a.notification.title)
  })
}
