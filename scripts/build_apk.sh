#!/usr/bin/env bash
# Rebuild the SeismicSoCal Android APK from the CURRENT web code and stage it for the /app download.
#
# The native app's backend URL is baked in at BUILD TIME from VITE_API_BASE:
#   scripts/build_apk.sh
#       -> no VITE_API_BASE: the app targets http://10.0.2.2:8000 (Android EMULATOR only; a real
#          phone can't reach that). Fine for getting current features into the APK, not for shipping.
#   VITE_API_BASE=https://api.yourhost.com scripts/build_apk.sh
#       -> bakes your hosted backend URL in, so the APK works on real devices. Use HTTPS.
#
# Requires: Node/npm, Capacitor CLI (bundled), Android SDK (ANDROID_HOME), and JDK 21
# (pinned in app/android/gradle.properties via org.gradle.java.home).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/app"

if [ -n "${VITE_API_BASE:-}" ]; then
  export VITE_API_BASE
  echo "==> Backend URL baked into the app: $VITE_API_BASE"
else
  echo "==> No VITE_API_BASE set -> native app will target http://10.0.2.2:8000 (EMULATOR ONLY)."
  echo "    Re-run as: VITE_API_BASE=https://your-host scripts/build_apk.sh  to ship to real phones."
fi

echo "==> [1/4] Building web app (Vite)"
npm run build

echo "==> [2/4] Syncing web build into the Android project (cap sync)"
npx cap sync android

echo "==> [3/4] Assembling debug APK (JDK 21 pinned in gradle.properties)"
( cd android && ./gradlew assembleDebug --no-daemon )

echo "==> [4/4] Staging APK for the /app download page"
APK="$ROOT/app/android/app/build/outputs/apk/debug/app-debug.apk"
DEST="$ROOT/app/public/seismicsocal.apk"
cp "$APK" "$DEST"
SIZE=$(stat -c '%s' "$DEST")
printf '==> Done: app/public/seismicsocal.apk  (%s bytes, %.1f MB)\n' "$SIZE" "$(awk "BEGIN{print $SIZE/1048576}")"
