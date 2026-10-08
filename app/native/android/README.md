# Native Android: personalized alert text (shaking at the user's home)

**What it does.** For a quake push, the phone writes the notification itself, adding a line like "Light shaking
likely at Pasadena (MMI IV, 34 km)". The estimate is computed on the phone from the home saved in the app (Capacitor
Preferences), so the location never leaves the device. A server only sends these data-only pushes to apps that
registered the `local_text` capability. The app registers it only if the `QuakeNativePlugin` below is present, so
older APKs keep getting normal notifications.

These two Java files are the source of truth; `app/android/` is gitignored, so they are copied into it.

## Where the Android project lives (2026-10-07)
The real project (`com.seismicsocal`) was moved to **this PC** from the other device: `app/android/` (gitignored),
`app/android/app/google-services.json` (Firebase) and the signing key `~/.android/debug.keystore` (the debug key
that signed 2.00.00; SHA-256 `7dd3cbc8…df9ff99`). The originals are in `other device/` at the repo root (gitignored;
keep a backup of the keystore elsewhere — losing it means every user must uninstall and reinstall).
Requires Node 22 for the Capacitor CLI; `scripts/build_apk.sh` runs just that step on Node 22 via npx if the
system Node is older, and uses Android Studio's bundled JDK.

## Install into the project (already done here for 2.01.00)
1. Copy `QuakeMessagingService.java` and `QuakeNativePlugin.java` to
   `app/android/app/src/main/java/com/seismicsocal/`.
2. Register the plugin in `MainActivity.java` (`registerPlugin(QuakeNativePlugin.class)` before `super.onCreate`).
3. In `app/android/app/build.gradle` dependencies, add
   `implementation "com.google.firebase:firebase-messaging:25.0.1"` (the push plugin keeps it private; same version).
4. Replace the push plugin's messaging service in `app/android/app/src/main/AndroidManifest.xml`. Add
   `xmlns:tools="http://schemas.android.com/tools"` on `<manifest>`, then inside `<application>`:
   ```xml
   <service android:name="com.capacitorjs.plugins.pushnotifications.MessagingService" tools:node="remove" />
   <service android:name=".QuakeMessagingService" android:exported="false">
       <intent-filter>
           <action android:name="com.google.firebase.MESSAGING_EVENT" />
       </intent-filter>
   </service>
   ```
5. Version: `versionName`/`versionCode` in `app/android/app/build.gradle` and `APP_VERSION` in `app/src/version.ts`.
6. Build: `VITE_API_BASE=https://seismicsocal.duckdns.org bash scripts/build_apk.sh` → `app/public/seismicsocal.apk`.
   Check: `aapt dump badging` shows `com.seismicsocal` + the version, `apksigner verify --print-certs` shows the
   SHA-256 above.

## Test on a phone before shipping
- Install over 2.00.00 (same key, so it updates in place). Open the app, find sensors near you (this saves the
  home), and subscribe.
- In `data/processed/push_tokens.json` on the server, the device should now have `"caps": ["local_text"]`.
- Send a test, or wait for a quake: the notification should show the "shaking likely at …" line, and tapping it
  should open the quake page.

## Ship
scp `app/public/seismicsocal.apk` to the VM (`/opt/seismicsocal/app/dist/`, DEPLOY.md), then raise the server's
`APP_LATEST_VERSION` to 2.01.00 for a soft "update available" notice. Only raise `APP_MIN_VERSION` to force it.
