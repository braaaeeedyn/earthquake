"""Firebase Cloud Messaging (FCM HTTP v1) push sender + device-token store.

The mobile app (SeismicSoCal, Capacitor + @capacitor/push-notifications) registers each
device with FCM, gets a token, and POSTs {token, stations, name} to the server's
/api/register-push — where `stations` is the list of sensor codes the device subscribes to
(the ones nearest the user, chosen at signup). We store the station subscription, NOT the
user's coordinates, so the live watcher can push a device whenever one of its stations fires.

Auth: FCM's legacy server key is retired, so we use the HTTP v1 API with an OAuth2 access
token minted from the service-account key (Firebase console -> Project settings ->
Service accounts -> Generate new private key). Path via env FCM_SERVICE_ACCOUNT, else the
repo-root fcm-service-account.json. Missing creds -> dry-run print (same contract as
mailer.send_email), so the app/server still run without push configured.

  python scripts/push_fcm.py --selftest      # mint a token + validate creds (no send)
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKENS = ROOT / "data" / "processed" / "push_tokens.json"
SA_PATH = Path(os.environ.get("FCM_SERVICE_ACCOUNT", ROOT / "fcm-service-account.json"))
SCOPE = "https://www.googleapis.com/auth/firebase.messaging"

_creds = None            # cached google.oauth2 credentials


def load_tokens():
    if TOKENS.exists():
        try:
            return json.loads(TOKENS.read_text())
        except (ValueError, OSError):
            return []
    return []


def save_token(token, stations, name="", mode="standard", caps=()):
    """Upsert a device by token (a device's FCM token is its identity). `stations` is the list of
    sensor codes this device subscribes to (e.g. ['CCC','MWC']); `mode` is its alert speed for the first
    message ('standard' = most safeguards, 'fast' = earlier, less certain); `caps` = what the app can do
    ('local_text': its native code builds the notification itself, with shaking at the user's home). Returns the list."""
    toks = load_tokens()
    entry = {"token": token, "stations": [str(s) for s in stations], "name": name, "mode": mode, "caps": list(caps)}
    toks = [t for t in toks if t.get("token") != token]     # replace stale subscription for same device
    toks.append(entry)
    TOKENS.parent.mkdir(parents=True, exist_ok=True)
    TOKENS.write_text(json.dumps(toks, indent=2))
    return toks


def remove_token(token):
    """Delete a device by its FCM token (unsubscribe). Returns the remaining list."""
    toks = [t for t in load_tokens() if t.get("token") != token]
    TOKENS.parent.mkdir(parents=True, exist_ok=True)
    TOKENS.write_text(json.dumps(toks, indent=2))
    return toks


def _access_token():
    """OAuth2 bearer token for the FCM v1 API, minted (and cached/refreshed) from the SA key."""
    global _creds
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account
    if _creds is None:
        _creds = service_account.Credentials.from_service_account_file(str(SA_PATH), scopes=[SCOPE])
    if not _creds.valid:
        _creds.refresh(Request())
    return _creds.token, _creds.project_id


def build_message(token, title, body, tag=None, data=None, data_only=False):
    """FCM v1 message. Normal: a notification Android shows itself (+ data for when it's tapped). data_only: no
    notification block -- the app's native QuakeMessagingService writes it (title/body ride along as fallback)."""
    fields = {k: str(v) for k, v in (data or {}).items() if v is not None}
    if data_only:
        return {"message": {"token": token, "data": {**fields, "title": title, "body": body, "tag": tag or ""},
                            "android": {"priority": "high"}}}
    return {"message": {
        "token": token,
        "notification": {"title": title, "body": body},
        **({"data": fields} if fields else {}),
        "android": {"priority": "high", **({"notification": {"tag": tag}} if tag else {})},
    }}


def send_push(token, title, body, dry_run=False, tag=None, data=None, data_only=False):
    """Push one notification to one device token via FCM HTTP v1.

    `tag`: pushes with the same tag REPLACE each other in the Android notification tray -- the live
    daemon tags both stages of an event with its id, so the confirmation (or retraction) overwrites
    the provisional "detected, sizing..." notice instead of stacking under it.
    `data`: string fields delivered to the app (the quake: id, lat, lon, mag, ...), used when a notification is
    tapped and, with data_only=True, by the app's native code to write the notification itself (title/body below
    become its fallback text). Only devices that registered the 'local_text' capability get data_only pushes.
    dry_run or missing service-account key -> print instead of send (returns False so callers
    can count real sends). Mirrors mailer.send_email's fail-soft behaviour.
    """
    if dry_run or not SA_PATH.exists():
        why = "dry-run" if dry_run else f"no {SA_PATH.name}"
        print(f"    [push {why}] -> {token[:16]}...  {title}")
        return False
    import requests
    access, project_id = _access_token()
    url = f"https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"
    msg = build_message(token, title, body, tag, data, data_only)
    r = requests.post(url, headers={"Authorization": f"Bearer {access}",
                                    "Content-Type": "application/json"},
                      data=json.dumps(msg), timeout=15)
    if r.status_code == 200:
        return True
    # A 404/UNREGISTERED means the device uninstalled/rotated its token -> drop it.
    if r.status_code in (404, 400) and "UNREGISTERED" in r.text:
        TOKENS.write_text(json.dumps([t for t in load_tokens() if t.get("token") != token], indent=2))
    print(f"    push failed ({r.status_code}) -> {token[:16]}...: {r.text[:160]}")
    return False


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="validate the service-account creds")
    args = ap.parse_args()
    if args.selftest:
        if not SA_PATH.exists():
            print(f"no service-account key at {SA_PATH} — push will run in dry-run mode")
        else:
            tok, pid = _access_token()
            print(f"FCM v1 ready: project={pid}, access token minted ({len(tok)} chars)")
        print(f"stored device tokens: {len(load_tokens())}")
