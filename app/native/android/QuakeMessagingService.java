package com.seismicsocal;

import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Build;
import androidx.annotation.NonNull;
import androidx.core.app.NotificationCompat;
import com.capacitorjs.plugins.pushnotifications.PushNotificationsPlugin;
import com.google.firebase.messaging.FirebaseMessagingService;
import com.google.firebase.messaging.RemoteMessage;
import java.util.Locale;
import java.util.Map;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 * Replaces the push plugin's MessagingService (see README.md). For data-only quake pushes (sent only to apps that
 * registered the 'local_text' capability) it writes the notification itself, adding the estimated shaking at the
 * user's home -- computed here, on the phone, from the home saved in Capacitor Preferences ("CapacitorStorage":
 * seismic.home = {lat, lon, label, vs30}) and the shaking equations (seismic.shaking_model, from /api/shaking-model).
 * The home location never leaves the device. Everything is still forwarded to the plugin, so the app's JS listeners
 * (foreground handling, tap -> quake page) keep working.
 */
public class QuakeMessagingService extends FirebaseMessagingService {

    private static final String CHANNEL = "quakes";

    @Override
    public void onNewToken(@NonNull String token) {
        super.onNewToken(token);
        PushNotificationsPlugin.onNewToken(token);
    }

    @Override
    public void onMessageReceived(@NonNull RemoteMessage msg) {
        super.onMessageReceived(msg);
        Map<String, String> d = msg.getData();
        if (msg.getNotification() == null && "quake".equals(d.get("type"))) {
            try {
                show(d);
            } catch (Exception e) {
                showPlain(d);                       // never lose an alert because the estimate failed
            }
        }
        PushNotificationsPlugin.sendRemoteMessage(msg);
    }

    private void show(Map<String, String> d) throws Exception {
        String body = d.get("body");
        String line = homeLine(d);
        notify(d, d.get("title"), line == null ? body : line + "\n" + body);
    }

    private void showPlain(Map<String, String> d) {
        notify(d, d.get("title"), d.get("body"));
    }

    /** "Light shaking likely at Pasadena (MMI IV, 34 km)" or null when no home / model is saved. */
    private String homeLine(Map<String, String> d) throws Exception {
        SharedPreferences prefs = getSharedPreferences("CapacitorStorage", Context.MODE_PRIVATE);
        String homeRaw = prefs.getString("seismic.home", null), modelRaw = prefs.getString("seismic.shaking_model", null);
        if (homeRaw == null || modelRaw == null || d.get("mag") == null) return null;
        JSONObject home = new JSONObject(homeRaw), m = new JSONObject(modelRaw);
        double km = haversineKm(home.getDouble("lat"), home.getDouble("lon"),
                Double.parseDouble(d.get("lat")), Double.parseDouble(d.get("lon")));
        double term = d.get("pgv_term") == null ? 0 : Double.parseDouble(d.get("pgv_term"));
        double vs30 = home.isNull("vs30") ? 0 : home.getDouble("vs30");
        double mmi = mmi(m, Double.parseDouble(d.get("mag")), km, vs30, term);
        int level = (int) Math.max(1, Math.min(10, Math.round(mmi)));
        String label = m.getJSONArray("levels").getJSONObject(level - 1).getString("label");
        String[] roman = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"};
        String where = home.optString("label", "your home");
        if (level <= 2) return String.format(Locale.US, "Likely not felt at %s (%.0f km)", where, km);
        return String.format(Locale.US, "%s shaking likely at %s (MMI %s, %.0f km)", label, where, roman[level - 1], km);
    }

    /** Same equations as src/eq/shaking.py and app/src/shaking.ts. */
    static double mmi(JSONObject m, double mag, double km, double vs30, double term) throws Exception {
        JSONObject g = m.getJSONObject("gmpe"), s = m.getJSONObject("site"), w = m.getJSONObject("worden");
        double r = Math.max(km, 1.0);
        double logPgv = g.getDouble("a") + g.getDouble("b") * mag + g.getDouble("c") * Math.log10(r) + g.getDouble("d") * r + term;
        if (vs30 > 0) logPgv += s.getDouble("e") * Math.log10(vs30 / s.getDouble("vs30_ref"));
        double y = logPgv + 2;                       // log10(PGV cm/s)
        JSONArray ab = y > w.getDouble("brk") ? w.getJSONArray("hi") : w.getJSONArray("lo");
        return Math.max(1, Math.min(10, ab.getDouble(0) + ab.getDouble(1) * y + m.optDouble("mmi_offset", 0)));
    }

    static double haversineKm(double aLat, double aLon, double bLat, double bLon) {
        double dLat = Math.toRadians(bLat - aLat), dLon = Math.toRadians(bLon - aLon);
        double s = Math.sin(dLat / 2) * Math.sin(dLat / 2)
                + Math.cos(Math.toRadians(aLat)) * Math.cos(Math.toRadians(bLat)) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
        return 2 * 6371 * Math.asin(Math.sqrt(s));
    }

    private void notify(Map<String, String> d, String title, String body) {
        NotificationManager nm = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O && nm.getNotificationChannel(CHANNEL) == null) {
            nm.createNotificationChannel(new NotificationChannel(CHANNEL, "Earthquake alerts", NotificationManager.IMPORTANCE_HIGH));
        }
        Intent open = getPackageManager().getLaunchIntentForPackage(getPackageName());
        PendingIntent pi = PendingIntent.getActivity(this, 0, open, PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
        NotificationCompat.Builder b = new NotificationCompat.Builder(this, CHANNEL)
                .setSmallIcon(getApplicationInfo().icon)
                .setContentTitle(title)
                .setContentText(body)
                .setStyle(new NotificationCompat.BigTextStyle().bigText(body))
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setAutoCancel(true)
                .setContentIntent(pi);
        String tag = d.get("tag");                   // quake-<event id>: a later stage replaces the earlier one
        nm.notify(tag == null || tag.isEmpty() ? null : tag, 1, b.build());
    }
}
