# Native Android: personalized alert text (shaking at the user's home)

These files go into the real Android project on the **other device** (the one with `google-services.json`).
This PC's `app/android` is a stale template, so they could not be compiled or device-tested here.

**What it does.** For a quake push, the phone writes the notification itself, adding a line like "Light shaking
likely at Pasadena (MMI IV, 34 km)". The estimate is computed on the phone from the home saved in the app (Capacitor
Preferences), so the location never leaves the device. A server only sends these data-only pushes to apps that
registered the `local_text` capability. The app registers it only if the `QuakeNative` plugin below is present, so
older APKs keep getting normal notifications.

## Install (Android project, package `com.seismicsocal`)
1. Copy `QuakeMessagingService.java` and `QuakeNativePlugin.java` to
   `app/android/app/src/main/java/com/seismicsocal/`.
2. Register the plugin in `MainActivity.java`:
   ```java
   public class MainActivity extends BridgeActivity {
       @Override
       public void onCreate(Bundle savedInstanceState) {
           registerPlugin(QuakeNativePlugin.class);
           super.onCreate(savedInstanceState);
       }
   }
   ```
3. Replace the push plugin's messaging service in `app/android/app/src/main/AndroidManifest.xml`. Add
   `xmlns:tools="http://schemas.android.com/tools"` on `<manifest>` if missing, then inside `<application>`:
   ```xml
   <service android:name="com.capacitorjs.plugins.pushnotifications.MessagingService" tools:node="remove" />
   <service android:name=".QuakeMessagingService" android:exported="false">
       <intent-filter>
           <action android:name="com.google.firebase.MESSAGING_EVENT" />
       </intent-filter>
   </service>
   ```
4. `cd app && npm install && npx cap sync android` (this also pulls in `@capacitor/preferences`), set
   versionName `2.01.00` / versionCode +1 in `app/android/app/build.gradle`, set `APP_VERSION = '2.01.00'` in
   `app/src/version.ts`, then build: `VITE_API_BASE=https://seismicsocal.duckdns.org bash scripts/build_apk.sh`.
5. **Test on a phone before shipping:**
   - Open the app, find sensors near you (this saves the home), and subscribe.
   - In `data/processed/push_tokens.json` on the server, the device should now have `"caps": ["local_text"]`.
   - Send a dry test (`PUSH_ENABLED=1` only for the test), or wait for a quake: the notification should show the
     "shaking likely at …" line, and tapping it should open the quake page.
6. Ship it like 2.00.00 (DEPLOY.md). Raising the server's `APP_LATEST_VERSION` to 2.01.00 shows a soft "update
   available" notice. Only raise `APP_MIN_VERSION` if you want to force it.
