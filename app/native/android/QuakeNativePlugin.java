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
 */
@CapacitorPlugin(name = "QuakeNative")
public class QuakeNativePlugin extends Plugin {
    @PluginMethod
    public void info(PluginCall call) {
        JSObject r = new JSObject();
        r.put("localText", true);
        call.resolve(r);
    }
}
