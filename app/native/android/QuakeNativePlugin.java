package com.seismicsocal;

import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

/**
 * Marker plugin: its presence tells the app's JS (Capacitor.isPluginAvailable('QuakeNative')) that this APK includes
 * QuakeMessagingService, so it can register the 'local_text' capability and receive data-only quake pushes.
 * An APK without it keeps receiving normal notifications -- nothing breaks for older builds.
 * It also hands the system font scale to the page (see load()).
 */
@CapacitorPlugin(name = "QuakeNative")
public class QuakeNativePlugin extends Plugin {
    // The WebView scales only the TEXT with Android's font-size setting, so large fonts overflowed the layout.
    // Turn that off; main.tsx zooms the whole page by fontScale instead, so every part grows together.
    @Override
    public void load() {
        getBridge().getWebView().getSettings().setTextZoom(100);
    }

    @PluginMethod
    public void info(PluginCall call) {
        JSObject r = new JSObject();
        r.put("localText", true);
        r.put("fontScale", getContext().getResources().getConfiguration().fontScale);
        call.resolve(r);
    }
}
