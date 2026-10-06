"""Shared .env loading + plain SMTP email.

Importing this module loads KEY=VALUE lines from the gitignored repo-root .env into os.environ (without
overriding variables already set), so scripts that run outside systemd -- tracking.py, retrain.py,
drift_check.py, server.py started by hand -- see the same settings as the service. send_email() sends the
QuakeOps operator notices (promotions, drift streaks); it prints instead when SMTP isn't configured.
"""
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env():
    envf = ROOT / ".env"
    if not envf.exists():
        return
    for line in envf.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()


def send_email(to_addr, subject, body, dry_run=False):
    if dry_run or not os.environ.get("SMTP_USER"):
        print(f"    [email not sent: {'dry run' if dry_run else 'no SMTP_USER'}] to={to_addr}  {subject}\n      {body}")
        return
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = os.environ["SMTP_USER"], to_addr, subject
    msg.set_content(body)
    with smtplib.SMTP(os.environ.get("SMTP_HOST", "smtp.gmail.com"), int(os.environ.get("SMTP_PORT", "587")),
                      timeout=30) as srv:
        srv.starttls()
        srv.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        srv.send_message(msg)
    print(f"    sent email -> {to_addr}")
