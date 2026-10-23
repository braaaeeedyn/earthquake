"""Build the two MMI review PDFs in the repo root: MMI_summary_2026-10-23.pdf (one page) and MMI_details_2026-10-23.pdf
(one page per point of the summary). Tables and figures are recomputed from the data where cheap; the prose and the other
numbers are typed into mmi_summary_template.html / mmi_details_template.html from the verified runs of mmi_review.py.
HTML -> PDF with headless Edge; fails if a page overflows.

  python scripts/mmi_review.py        # once: DYFI cells -> data/processed/mmi_notes/dyfi_cells_scored.csv
  python scripts/build_mmi_notes.py   # figures + html in data/processed/mmi_notes/, PDFs in the repo root
"""
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).parent
OUT = ROOT / "data/processed/mmi_notes"                      # written by mmi_review.py (gitignored)
sys.path.insert(0, str(ROOT / "src"))
from eq import shaking, stats  # noqa: E402

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1f", "#6b6b6b", "#e4e4e2"

# ------------------------------------------------------------------ data
d = np.load(ROOT / "data/processed/v2/magnitude.npz")
o = np.argsort(d["ev_time"], kind="stable"); n = len(o)
split = np.empty(n, "<U5")
split[o[:int(n * .7)]], split[o[int(n * .7):int(n * .85)]], split[o[int(n * .85):]] = "train", "val", "test"
rs, M, R = split[d["row_ev"]], d["mag"][d["row_ev"]].astype(float), d["row_dist"].astype(float)
df = pd.read_csv(OUT / "dyfi_cells_scored.csv")
fit, ev = df[df.fit], df[~df.fit]
evt = ev[ev.term.notna()]
off_t = float(np.median(fit.cdi - fit.mmi_t))
fmt = lambda x, p=2: f"{x:+.{p}f}".replace("-", "&minus;")
f2 = lambda x: f"{x:.2f}"


# ------------------------------------------------------------------ tables
def mr_table(mask, mbins, rbins, rlabels, mlabels):
    h, _, _ = np.histogram2d(M[mask], R[mask], [mbins, rbins])
    rows = "".join(f"<tr><th>{ml}</th>" + "".join(f"<td>{int(v):,}</td>" for v in r) + f"<td><b>{int(r.sum()):,}</b></td></tr>"
                   for ml, r in zip(mlabels, h))
    tot = "<tr class=tot><th>All</th>" + "".join(f"<td>{int(v):,}</td>" for v in h.sum(0)) + f"<td><b>{int(h.sum()):,}</b></td></tr>"
    head = "<tr><th>M \\ R (km)</th>" + "".join(f"<th>{r}</th>" for r in rlabels) + "<th>All</th></tr>"
    return f"<table class=num>{head}{rows}{tot}</table>"


RB, RL = [0, 5, 10, 20, 50, 100, 200.0001], ["0&ndash;5", "5&ndash;10", "10&ndash;20", "20&ndash;50", "50&ndash;100", "100&ndash;200"]
MB, ML = [2, 3, 4, 5, 6, 7.2], ["2&ndash;3", "3&ndash;4", "4&ndash;5", "5&ndash;6", "6&ndash;7.1"]
t_train = mr_table(rs == "train", MB, RB, RL, ML)
t_test = mr_table(rs == "test", MB[:-1], RB, RL, ML[:-1])


def cell_table(g):
    rb = [0, 5, 10, 20, 50, 100, 200, 400]
    t = pd.crosstab(pd.cut(g.mag, [4, 4.5, 5, 5.5], right=False), pd.cut(g.r, rb, right=False))
    nq = g.groupby("id").mag.first()
    qn = np.histogram(nq, [4, 4.5, 5, 5.5])[0]
    head = "<tr><th>M</th><th>Quakes</th>" + "".join(f"<th>{a}&ndash;{b}</th>" for a, b in zip(rb[:-1], rb[1:])) + "<th>All cells</th></tr>"
    body = "".join(f"<tr><th>{lab}</th><td>{q}</td>" + "".join(f"<td>{v:,}</td>" for v in r) + f"<td><b>{r.sum():,}</b></td></tr>"
                   for lab, q, r in zip(["4.0&ndash;4.5", "4.5&ndash;5.0", "5.0&ndash;5.3"], qn, t.values))
    tot = "<tr class=tot><th>All</th>" + f"<td>{qn.sum()}</td>" + "".join(f"<td>{v:,}</td>" for v in t.values.sum(0)) + f"<td><b>{t.values.sum():,}</b></td></tr>"
    return f"<table class=num>{head}{body}{tot}</table>"


t_cells_fit, t_cells_ev = cell_table(fit), cell_table(ev)

y, ids = evt.cdi.values, evt.id.values
mae = lambda yy, p: float(np.mean(np.abs(np.clip(p, 1, 10) - yy)))
bias = lambda yy, p: float(np.mean(np.clip(p, 1, 10) - yy))
STAGES = [("A", "Equation + site term &rarr; Worden", "0", evt.mmi0.values),
          ("B", "A + event correction", "0", evt.mmi_t.values),
          ("C", "A + offset (published)", "0.76", evt.mmi0.values + .76),
          ("D", "B + refitted offset (used)", f"{off_t:.2f}", evt.mmi_t.values + off_t),
          ("D&prime;", "B + offset 0.76", "0.76", evt.mmi_t.values + .76)]
SHORT = {"A": "Equation + site term &rarr; Worden", "B": "A + event correction", "C": "A + offset 0.76 (published)",
         "D": f"B + offset {off_t:.2f} (used with the correction)", "D&prime;": "B + offset 0.76"}
res, staged_rows, short_rows = {}, "", ""
for k, lab, off, p in STAGES:
    m_, mlo, mhi = stats.bootstrap_ci(mae, y, p, clusters=ids, n_boot=2000)
    b_, blo, bhi = stats.bootstrap_ci(bias, y, p, clusters=ids, n_boot=2000)
    e = np.clip(p, 1, 10) - y
    rmse, w1 = np.sqrt(np.mean(e ** 2)), np.mean(np.abs(e) <= 1)
    mq = pd.Series(np.abs(e)).groupby(ids).mean().mean()
    res[k] = (m_, mlo, mhi, b_, blo, bhi)
    print(f"{k:8s} MAE {m_:.4f} [{mlo:.4f},{mhi:.4f}] bias {b_:+.4f} [{blo:+.4f},{bhi:+.4f}] rmse {rmse:.4f} w1 {w1:.4f} mq {mq:.4f}")
    cls = " class=hl" if k == "D" else ""      # the setup used with the correction
    mae_ci = f"{f2(m_)} <span class=ci>[{f2(mlo)}, {f2(mhi)}]</span>"
    bias_ci = f"{fmt(b_)} <span class=ci>[{fmt(blo)}, {fmt(bhi)}]</span>"
    staged_rows += (f"<tr{cls}><th>{k}</th><td class=l>{lab}</td><td>{off}</td><td>{mae_ci}</td><td>{bias_ci}</td>"
                    f"<td>{f2(rmse)}</td><td>{w1:.1%}</td><td>{f2(mq)}</td></tr>")
    short_rows += f"<tr{cls}><th>{k}</th><td class=l>{SHORT[k]}</td><td>{mae_ci}</td><td>{bias_ci}</td></tr>"
P_ = {k: p for k, _, _, p in STAGES}


def paired(pairs):
    rows = ""
    for a, b, what in pairs:
        dl, lo, hi, *_ = stats.paired_bootstrap(mae, y, P_[a], P_[b], clusters=ids, n_boot=2000)
        print(f"paired {a}-{b}: {dl:+.4f} [{lo:+.4f},{hi:+.4f}]")
        rows += f"<tr><th>{a} &minus; {b}</th><td class=l>{what}</td><td>{fmt(dl)} <span class=ci>[{fmt(lo)}, {fmt(hi)}]</span></td></tr>"
    return rows


paired_live = paired((("D&prime;", "C", "offset 0.76 on top of the correction vs published"),
                      ("D&prime;", "D", "offset 0.76 vs 0.43, both with the correction")))
paired_term = paired((("B", "A", "event correction, no offset"), ("D", "C", "correction with matching offset vs published")))

eC = np.clip(evt.mmi0 + .76, 1, 10) - evt.cdi
dist_rows = ""
for a, b in zip([0, 5, 10, 20, 50, 100, 200], [5, 10, 20, 50, 100, 200, 400]):
    s = (evt.r >= a) & (evt.r < b)
    dist_rows += f"<tr><th>{a}&ndash;{b}</th><td>{s.sum():,}</td><td>{fmt(eC[s].mean())}</td><td>{f2(eC[s].abs().mean())}</td></tr>"
cdi_rows = ""
for a, b, lab in ((1, 2, "1.0&ndash;1.9"), (2, 3, "2.0&ndash;2.9"), (3, 4, "3.0&ndash;3.9"), (4, 5, "4.0&ndash;4.9"), (5, 9, "5.0 and above")):
    s = (evt.cdi >= a) & (evt.cdi < b)
    cdi_rows += f"<tr><th>{lab}</th><td>{s.sum():,}</td><td>{fmt(eC[s].mean())}</td><td>{f2(eC[s].abs().mean())}</td></tr>"

pred_rows = ""
for mag in (3.0, 4.0, 4.5, 5.5, 6.5, 7.1):
    cells = []
    for r in (1, 5, 10, 20, 50):
        pgv = shaking.estimate_pgv(mag, r) * 100
        mmi = shaking.pgv_to_mmi(pgv / 100)
        raw = (3.78 + 1.47 * np.log10(pgv)) if np.log10(pgv) <= .53 else (2.89 + 3.16 * np.log10(pgv))
        note = "*" if raw < 1 or raw > 10 else ""
        cells.append(f"<td>{pgv:,.0f} / {mmi:.1f}{note}</td>" if pgv >= 100 else f"<td>{pgv:.3g} / {mmi:.1f}{note}</td>")
    pred_rows += f"<tr><th>M{mag}</th>{''.join(cells)}</tr>"

# ------------------------------------------------------------------ figures
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.linewidth": .6,
                     "xtick.major.width": .6, "ytick.major.width": .6, "legend.frameon": False})


def style(ax):
    ax.grid(True, color=GRID, lw=.5, which="major"); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.9), dpi=220, gridspec_kw={"wspace": .28})
for msk, c, lab in ((rs == "train", BLUE, f"Train (2000–2019): {(rs == 'train').sum():,} records"),
                    (rs == "test", ORANGE, f"Test (2021–2026): {(rs == 'test').sum():,} records")):
    a1.scatter(R[msk], M[msk], s=1.2, c=c, alpha=.35, lw=0, rasterized=True, label=lab)
a1.set_xscale("log"); a1.set_xlim(.3, 260); a1.set_ylim(1.9, 8.3); a1.set_yticks(range(2, 8))
a1.set_xlabel("Epicentral distance (km)"); a1.set_ylabel("Catalogue magnitude")
a1.set_title("a   Station records used to fit the PGV equation", loc="left", fontsize=8.5, color=INK)
for g, c, lab in ((fit, BLUE, f"Offset fit: 29 older quakes, {len(fit):,} cells"),
                  (ev, ORANGE, f"Scored: 28 newer quakes, {len(ev):,} cells")):
    a2.scatter(g.r, g.mag, s=3, c=c, alpha=.45, lw=0, rasterized=True, label=lab)
a2.set_xscale("log"); a2.set_xlim(1, 420); a2.set_ylim(3.98, 5.62); a2.set_yticks(np.arange(4, 5.3, .2))
a2.set_xlabel("Distance, epicentre to cell centre (km)"); a2.set_ylabel("Catalogue magnitude")
a2.set_title("b   DYFI 10 km cells used to check MMI", loc="left", fontsize=8.5, color=INK)
for ax in (a1, a2):
    style(ax)
    lg = ax.legend(loc="upper left", fontsize=7, markerscale=5, handletextpad=.3)
    for h in lg.legend_handles:
        h.set_alpha(1)
fig.savefig(OUT / "fig1_mr.png", bbox_inches="tight"); plt.close(fig)

fig, (b1, b2) = plt.subplots(1, 2, figsize=(7.0, 2.1), dpi=220, sharey=True, gridspec_kw={"wspace": .08})
ylab = ["A  equation + site", "B  + event correction", "C  + offset 0.76 (published)", f"D  + correction, offset {off_t:.2f} (used)",
        "D′ + correction, offset 0.76"]
keys = [k for k, *_ in STAGES]
for i, k in enumerate(keys):
    m_, mlo, mhi, b_, blo, bhi = res[k]
    c = ORANGE if k == "D" else BLUE
    yy = len(keys) - 1 - i
    b1.plot([mlo, mhi], [yy, yy], color=c, lw=2, solid_capstyle="round"); b1.plot(m_, yy, "o", color=c, ms=6, mec="white", mew=1)
    b1.text(mhi + .015, yy, f"{m_:.2f}", va="center", fontsize=7.5, color=INK)
    b2.plot([blo, bhi], [yy, yy], color=c, lw=2, solid_capstyle="round"); b2.plot(b_, yy, "o", color=c, ms=6, mec="white", mew=1)
    b2.text(bhi + .025, yy, f"{b_:+.2f}".replace("-", "−"), va="center", fontsize=7.5, color=INK)
b1.set_yticks(range(len(keys))[::-1]); b1.set_yticklabels(ylab)
b1.set_xlim(.3, .92); b1.set_xlabel("MAE, MMI levels (lower is better)")
b2.set_xlim(-.9, .62); b2.axvline(0, color=MUTED, lw=.8); b2.set_xlabel("Bias, estimate − DYFI (0 is best)")
for ax in (b1, b2):
    style(ax); ax.tick_params(axis="y", length=0); ax.set_ylim(-.6, 4.6)
fig.savefig(OUT / "fig2_stages.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ documents
SUBS = {"T_TRAIN": t_train, "T_TEST": t_test, "T_CELLS_FIT": t_cells_fit, "T_CELLS_EV": t_cells_ev,
        "STAGED_ROWS": staged_rows, "STAGED_SHORT": short_rows, "PAIRED_LIVE": paired_live, "PAIRED_TERM": paired_term,
        "DIST_ROWS": dist_rows, "CDI_ROWS": cdi_rows, "PRED_ROWS": pred_rows, "OFF_T": f"{off_t:.2f}"}
edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"  # ponytail: Windows-only; any headless Chromium works
for name, pdf_name in (("mmi_summary", "MMI_summary_2026-10-23.pdf"), ("mmi_3page", "MMI_3page_2026-10-23.pdf"),
                       ("mmi_details", "MMI_details_2026-10-23.pdf")):
    html = (HERE / f"{name}_template.html").read_text(encoding="utf-8")
    for k, v in SUBS.items():
        html = html.replace("{{" + k + "}}", v)
    assert "{{" not in html, name
    page, pdf = OUT / f"{name}.html", ROOT / pdf_name
    page.write_text(html, encoding="utf-8")
    subprocess.run([edge, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--print-to-pdf-no-header",
                    f"--print-to-pdf={pdf}", page.as_uri()], check=True, timeout=120)
    pages, want = len(re.findall(rb"/Type\s*/Page\b", pdf.read_bytes())), max(1, html.count("<section"))
    assert pages == want, f"{pdf.name}: {pages} pages, expected {want} (a page overflowed)"
    print("wrote", pdf, pages, "page(s),", pdf.stat().st_size // 1024, "KB")
