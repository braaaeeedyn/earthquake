// Mobile push registration (Capacitor + FCM). Push is the only alert channel — there is no
// email path. No-op on web: the browser build has no native push, so enablePush() returns null
// there and alerts are only available in the installed app.

import { Capacitor, type PluginListenerHandle } from '@capacitor/core'
import { PushNotifications } from '@capacitor/push-notifications'
import { registerPushToken, unregisterPushToken } from './nearme'

const SUBSCRIBED_KEY = 'seismic.push.subscribed'
const TOKEN_KEY = 'seismic.push.token'

export function isNativeApp() {
  return Capacitor.isNativePlatform()
}

// Whether this device is currently registered for push alerts (persisted locally).
export function isSubscribed(): boolean {
  try {
    return localStorage.getItem(SUBSCRIBED_KEY) === '1'
  } catch {
    return false
  }
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

// Request permission, register for push, and send {token, stations} to the server.
// `stations` is the list of sensor codes the user chose to subscribe to (near them).
// Returns a status string, or null on web (where push isn't available).
export async function enablePush(sub: { name: string; stations: string[] }): Promise<string | null> {
  if (!Capacitor.isNativePlatform()) return null
  if (!sub.stations.length) throw new Error('pick at least one station')

  let perm = await PushNotifications.checkPermissions()
  if (perm.receive !== 'granted') perm = await PushNotifications.requestPermissions()
  if (perm.receive !== 'granted') throw new Error('notification permission denied')

  const token = await awaitToken()
  await registerPushToken({ token, stations: sub.stations, name: sub.name })
  try {
    localStorage.setItem(SUBSCRIBED_KEY, '1')
    localStorage.setItem(TOKEN_KEY, token)
  } catch {
    // storage unavailable — state just won't persist across restarts
  }
  return 'Push alerts enabled on this device.'
}

// Unsubscribe this device: remove its token server-side and clear the local subscribed state.
export async function disablePush(): Promise<void> {
  if (!Capacitor.isNativePlatform()) return
  let token = ''
  try {
    token = localStorage.getItem(TOKEN_KEY) ?? ''
  } catch {
    token = ''
  }
  if (!token) token = await awaitToken()   // token is stable per install; re-fetch if not stored
  await unregisterPushToken({ token })
  try {
    localStorage.removeItem(SUBSCRIBED_KEY)
    localStorage.removeItem(TOKEN_KEY)
  } catch {
    // ignore
  }
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
