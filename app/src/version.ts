// App version — bump on every release (and rebuild the APK).
//   MAJOR . MINOR . PATCH  (displayed as x.xx.xx)
//   MAJOR = big release, MINOR = feature update, PATCH = quality-of-life (optional).
// The server publishes { latest, min }. If the installed app is behind `min` on MAJOR or MINOR,
// the app blocks and sends the user to the download page; a PATCH gap is only a soft notice.
export const APP_VERSION = '1.00.00'

type Ver = [number, number, number]

function parse(s: string): Ver {
  const p = String(s).split('.').map((x) => parseInt(x, 10))
  return [p[0] || 0, p[1] || 0, p[2] || 0]
}

// Must the user update? True when `have` is behind `min` on MAJOR or MINOR (PATCH is ignored).
export function mustUpdate(have: string, min: string): boolean {
  const [aMaj, aMin] = parse(have)
  const [mMaj, mMin] = parse(min)
  return aMaj < mMaj || (aMaj === mMaj && aMin < mMin)
}

// Is any newer version available (for a non-blocking "update available" notice)?
export function updateAvailable(have: string, latest: string): boolean {
  const a = parse(have)
  const l = parse(latest)
  for (let i = 0; i < 3; i++) {
    if (a[i] < l[i]) return true
    if (a[i] > l[i]) return false
  }
  return false
}
