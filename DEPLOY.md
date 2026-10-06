# Deploying SeismicSoCal to Oracle Cloud "Always Free"

Goal: the backend + live alert daemon run 24/7, the site is served over HTTPS, and the Android
app talks to the live backend. Target host: an Oracle **Ampere A1 (ARM64)** Always-Free VM
running **Ubuntu**, fronted by **Caddy** (automatic HTTPS). Estimated time: ~1 hour.

Placeholders used below: `HOST_IP` (the VM's public IP), `seismicsocal.example` (your hostname).

---

## 1. Create the VM (Oracle Cloud console)

1. Sign up at cloud.oracle.com (free; a credit card is required for identity check — Always-Free
   resources are not charged). Pick a home region close to you.
2. **Compute → Instances → Create instance:**
   - **Image:** Canonical **Ubuntu 22.04** (or 24.04).
   - **Shape:** *Change shape* → **Ampere** → `VM.Standard.A1.Flex`. Set **2 OCPU / 12 GB RAM**
     (well within the 4 OCPU / 24 GB Always-Free ceiling, plenty for the models).
   - If you see **"Out of host capacity"**, that's the well-known A1 shortage — try a different
     Availability Domain, try another region, or retry later (it frees up). It is not a mistake
     on your end.
   - **SSH keys:** upload your public key (or let it generate one and download the private key).
   - **Create.** Note the **public IP** (`HOST_IP`).

## 2. Open the firewall — BOTH layers (the classic Oracle gotcha)

Oracle blocks traffic in two places; you must open **both** for ports 80 and 443.

1. **Cloud firewall (VCN Security List):** Networking → Virtual Cloud Networks → your VCN →
   Security Lists → Default → **Add Ingress Rules**:
   - Source `0.0.0.0/0`, IP Protocol TCP, Destination port **80**
   - Source `0.0.0.0/0`, IP Protocol TCP, Destination port **443**
2. **Instance firewall (iptables on the Ubuntu image):** SSH in (next step) and run:
   ```bash
   sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80  -j ACCEPT
   sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
   sudo netfilter-persistent save
   ```

## 3. Point your hostname at the VM

Caddy needs a resolvable hostname to issue a certificate. Either:
- **A domain you own:** add a DNS **A record** → `HOST_IP`, or
- **Free option — DuckDNS:** create a subdomain at duckdns.org and set it to `HOST_IP`.

Wait until `ping seismicsocal.example` resolves to `HOST_IP` before step 6.

## 4. Get the project + data onto the VM

SSH in: `ssh ubuntu@HOST_IP`. Put the project at `/opt/seismicsocal`:

```bash
sudo mkdir -p /opt/seismicsocal && sudo chown ubuntu:ubuntu /opt/seismicsocal
```

From **your Windows machine**, copy the code and the gitignored files it needs (models,
calibration, the station-network `.npz`, and the secrets). Using `rsync` (Git Bash) or `scp`:

```bash
# code (skip the heavy/local-only stuff)
rsync -av --exclude node_modules --exclude .venv --exclude app/android \
      --exclude app/dist --exclude .git \
      /c/Users/braaa/coding/earthquake/ ubuntu@HOST_IP:/opt/seismicsocal/

# gitignored data + secrets the server/daemon need (not carried by git)
scp data/processed/detector.pt data/processed/magnitude_ensemble.pt \
    data/processed/shaking_calibration.json ubuntu@HOST_IP:/opt/seismicsocal/data/processed/
ssh ubuntu@HOST_IP mkdir -p /opt/seismicsocal/data/processed/v2
scp data/processed/v2/pipeline_config.json data/processed/v2/tt_correction.json \
    data/processed/v2/early_mag.json data/processed/v2/early_mag_T2.json \
    ubuntu@HOST_IP:/opt/seismicsocal/data/processed/v2/
scp fcm-service-account.json .env ubuntu@HOST_IP:/opt/seismicsocal/

# VERIFY the VM runs exactly the models you trained (a mismatch here is how the VM once ran
# months-old models while the site advertised new ones):
sha256sum data/processed/detector.pt data/processed/magnitude_ensemble.pt
ssh ubuntu@HOST_IP 'cd /opt/seismicsocal && sha256sum data/processed/detector.pt data/processed/magnitude_ensemble.pt'
```

The checkpoints carry their station list, threshold and amplitude scale; the daemon refuses to start
if a checkpoint's stations differ from `src/eq/network.py`, so models and code must ship together.

## 5. Install runtime + build the site (on the VM)

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip build-essential nodejs npm caddy

cd /opt/seismicsocal
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip
# On ARM64, plain `pip install torch` already gives the CPU build.
pip install torch numpy scipy scikit-learn matplotlib obspy google-auth requests
#   If obspy fails to build on ARM, install miniforge and `conda install -c conda-forge obspy`.

# sanity: the models + gate load and the pipeline runs offline
python scripts/live_watch.py --selftest      # location, 3-station rule, push floor, targeting
python -c "import sys; sys.path[:0]=['scripts','src']; import live_watch; live_watch.load_detector(); from eq.pipeline import MagnitudeEnsemble; from eq import network; MagnitudeEnsemble(live_watch.MAG_CKPT, network.CODES, network.COORDS); print('models OK')"

# build the web app (the site). Web build needs no VITE_API_BASE (it calls /api relatively).
cd app && npm ci && npm run build && cd ..
```

## 6. Run the backend as a service + start Caddy

```bash
# backend + auto-respawning daemon
sudo cp deploy/seismicsocal.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now seismicsocal
systemctl status seismicsocal            # should be active (running)
curl -s localhost:8000/api/status        # {"live": true, "stations": {...}} once SeedLink connects

# HTTPS + static site + /api proxy
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
sudo sed -i 's/your-domain.duckdns.org/seismicsocal.example/' /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

Visit `https://seismicsocal.example` — the site should load over HTTPS, and `/api/...` calls
should work (same origin, proxied to the backend).

## 7. Rebuild the Android app against the live backend

On your Windows machine (needs the Android SDK), bake the real URL in and re-stage the APK:

```bash
VITE_API_BASE=https://seismicsocal.example bash scripts/build_apk.sh
```

Then redeploy the site so the new APK ships (repeat the `rsync` of `app/public/seismicsocal.apk`,
or re-run step 5's build after copying it). Users download it from the site's `/app` page.

## 8. Verify end to end (the real test)

- **Site:** loads over HTTPS; `/app`, `/privacy`, and a bad URL (custom 404) all work on refresh.
- **Contact form:** send a message → it arrives in the support inbox (needs SMTP creds in `.env`).
- **Push:** install the rebuilt APK on a real phone, subscribe, and confirm the FCM token is
  stored: `python scripts/push_fcm.py --selftest` on the VM should report `stored device tokens: 1`.
- **Alert path:** `python scripts/live_watch.py --selftest` checks location, the 3-station rule, the
  push floor and targeting. For a true live check, let the service run and watch
  `journalctl -u seismicsocal -f` for `[EVENT]` lines when a real SoCal quake occurs.

## Shadow mode, then pushes

Pushes are OFF unless `PUSH_ENABLED=1` is in `/opt/seismicsocal/.env` (the unit loads it). After any
pipeline change (or a model change that did not come through the QuakeOps gate), run in **shadow mode** (pushes off) for at least 7 days, then score it:

```bash
python scripts/crosscheck_events.py --since <shadow start date>   # precision vs a +1 h chance baseline
```

Turn pushes on only if the live scorecard matches what the offline replay harness predicted
(`scripts/replay_archive.py`, see README). Subscriptions store station codes, so if the station network
ever changes, subscribers following a retired station must be moved to a remaining one (the one-off
`migrate_subscriptions.py` used for the 2026-10-05 change is in git history, added in commit ccac5c6).

## Operating it

- **Logs:** `journalctl -u seismicsocal -f` (backend + daemon), `journalctl -u caddy -f` (web).
- **Restart:** `sudo systemctl restart seismicsocal`. It restarts automatically on crash, and the
  daemon is auto-respawned by the backend if its SeedLink stream drops.
- **Update:** rsync the new code (plus the checkpoints and `data/processed/v2/*.json` if retrained,
  verifying the sha256), `sudo systemctl restart seismicsocal`, and rebuild the site
  (`cd app && npm run build`).
- **Station health:** `curl -s localhost:8000/api/status` lists each station's up/latency. A station
  can drop off the public SeedLink relay (SCZ2 did during selection); the pipeline works with those
  that stream.
- **Security:** after launch, rotate the Gmail app password (regenerate in Google, update `.env`,
  restart). Keep `.env` and `fcm-service-account.json` readable only by `ubuntu` (`chmod 600`).

## QuakeOps (MLflow registry, daily champion pull + drift, CI deploy)

How it works: HOW_IT_WORKS.md §12. This is a one-time setup, done in this order:
- `PC$` = Git Bash on the Windows PC at the repo root;
- `VM$` = an SSH session on the VM (`ssh -i ~/.ssh/oracle_seismic ubuntu@167.234.214.169`).

**1. Ship the current code** (no restart needed yet):
```bash
PC$ git archive HEAD | ssh -i ~/.ssh/oracle_seismic ubuntu@167.234.214.169 'tar -x -C /opt/seismicsocal'
```
**2. MLflow server** (its own venv keeps MLflow's pins away from the daemon's):
```bash
VM$ sudo mkdir -p /opt/mlflow && sudo chown ubuntu: /opt/mlflow
VM$ python3 -m venv /opt/mlflow/.venv && /opt/mlflow/.venv/bin/pip install -q mlflow
VM$ sudo cp /opt/seismicsocal/deploy/mlflow.service /etc/systemd/system/
VM$ sudo systemctl daemon-reload && sudo systemctl enable --now mlflow
VM$ sleep 20; curl -s 127.0.0.1:5000/health          # -> OK
```
**3. Password-protected public address** (`mlflow.seismicsocal.duckdns.org` already resolves to the VM):
```bash
VM$ caddy version                                    # 2.8+ spells it basic_auth; older versions: basicauth
VM$ caddy hash-password                              # type a password (keep it), copy the $2a$... hash it prints
VM$ sudo nano /etc/caddy/Caddyfile                   # append the mlflow block from deploy/Caddyfile, with the hash pasted in
VM$ sudo systemctl reload caddy
PC$ curl -u quakeops:YOUR_PASSWORD https://mlflow.seismicsocal.duckdns.org/health    # -> OK
```
**4. Daemon venv packages + the VM `.env`:**
```bash
VM$ /opt/seismicsocal/.venv/bin/pip install -q mlflow-skinny evidently==0.7.23
VM$ printf 'MLFLOW_TRACKING_URI=http://127.0.0.1:5000\nOPS_EMAIL_TO=you@example.com\nQUAKEOPS_AUTO_DEPLOY=0\n' >> /opt/seismicsocal/.env
```
**5. The PC `.env`** (append these lines; `.env` is gitignored, never commit it):
```
MLFLOW_TRACKING_URI=https://mlflow.seismicsocal.duckdns.org
MLFLOW_TRACKING_USERNAME=quakeops
MLFLOW_TRACKING_PASSWORD=YOUR_PASSWORD
```
**6. Register today's models as v1 @champion.** This builds the drift reference on the GPU and uploads about 3 MB:
```bash
PC$ .venv/Scripts/python scripts/tracking.py register-legacy
PC$ .venv/Scripts/python scripts/tracking.py status   # -> detector: champion v1 / magnitude: champion v1
```
**7. Daily job on the VM** (09:30 UTC: record the champion, then the drift check). Install it and run it once now:
```bash
VM$ cd /opt/seismicsocal && sudo cp deploy/seismicsocal-quakeops.{service,timer} /etc/systemd/system/
VM$ sudo systemctl daemon-reload && sudo systemctl enable --now seismicsocal-quakeops.timer
VM$ sudo systemctl start seismicsocal-quakeops.service; journalctl -u seismicsocal-quakeops -n 30 --no-pager
```
- **Expected output:** `installed missing sidecar drift_reference.csv`, then `deployed models are the current
  champions`.
- **If it says `v1 is @champion but vNone is deployed`:** the VM's checkpoints differ from the PC's; compare
  their sha256 (§4).
- **Drift:** reports `insufficient` until a full UTC day of live features exists.
- **The daemon restarts itself once**, about 30 s later, because it noticed `models.json` appear. That's
  expected.
- **Check:** `/api/health` now lists both models, and `/health` on the site shows them.

**8. Automatic deploys from GitHub** (optional). Create a dedicated key with no passphrase, authorise it on the
VM, and add two repo secrets (GitHub → Settings → Secrets and variables → Actions):
```bash
PC$ ssh-keygen -t ed25519 -f ~/.ssh/seismic_ci -N "" -C seismic-ci
PC$ cat ~/.ssh/seismic_ci.pub | ssh -i ~/.ssh/oracle_seismic ubuntu@167.234.214.169 'cat >> ~/.ssh/authorized_keys'
#   secrets: VM_HOST = 167.234.214.169    VM_SSH_KEY = the whole contents of ~/.ssh/seismic_ci
```
From then on, every push to `main` runs lint, tests and the site build. If they pass, it deploys the code and
site and restarts the service. Models still arrive only through the registry.

**9. Monthly retrain on the PC:** run `dagster dev -f scripts/quakeops_dagster.py` and switch the schedule on in
the UI. When the UI isn't running, use Task Scheduler (enable "run as soon as possible after a missed start"):
```
schtasks /Create /TN QuakeOpsRetrain /SC MONTHLY /D 3 /ST 03:00 /TR "C:\Users\brady\desktop\coding\earthquake\.venv\Scripts\python.exe C:\Users\brady\desktop\coding\earthquake\scripts\retrain.py"
```
First run it by hand: `.venv/Scripts/python scripts/retrain.py --month 2026-09 --dry-run` (takes hours, moves
nothing).

**Applying a new champion** (when `QUAKEOPS_AUTO_DEPLOY=0`): run `VM$ .venv/bin/python scripts/tracking.py pull --apply`.
It verifies the sha256, swaps the files in atomically, and the daemon restarts itself within 30 s. To roll back,
run `PC$ python scripts/retrain.py rollback --model detector --to N`, then `pull --apply` again.
