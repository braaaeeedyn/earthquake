"""Firebase Cloud Messaging (FCM HTTP v1) push sender + device-token store.

The mobile app (SeismicSoCal, Capacitor + @capacitor/push-notifications) registers each
device with FCM, gets a token, and POSTs {token, lat, lon, name} to the server's
/api/register-push. This module persists those tokens and pushes alerts to them, mirroring
the email path in live_watch.py.

Auth: FCM's legacy server key is retired, so we use the HTTP v1 API with an OAuth2 access
token minted from the service-account key (Firebase console -> Project settings ->
Service accounts -> Generate new private key). Path via env FCM_SERVICE_ACCOUNT, else the
repo-root fcm-service-account.json. Missing creds -> dry-run print (same contract as
nearme_watch.send_email), so the app/server still run without push configured.

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


def save_token(token, lat, lon, name=""):
    """Upsert a device by token (a device's FCM token is its identity). Returns the full list."""
    toks = load_tokens()
    entry = {"token": token, "lat": float(lat), "lon": float(lon), "name": name}
    toks = [t for t in toks if t.get("token") != token]     # replace stale location for same device
    toks.append(entry)
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


def send_push(token, title, body, dry_run=False):
    """Push one notification to one device token via FCM HTTP v1.

    dry_run or missing service-account key -> print instead of send (returns False so callers
    can count real sends). Mirrors nearme_watch.send_email's fail-soft behaviour.
    """
    if dry_run or not SA_PATH.exists():
        why = "dry-run" if dry_run else f"no {SA_PATH.name}"
        print(f"    [push {why}] -> {token[:16]}...  {title}")
        return False
    import requests
    access, project_id = _access_token()
    url = f"https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"
    msg = {"message": {
        "token": token,
        "notification": {"title": title, "body": body},
        "android": {"priority": "high"},
    }}
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
