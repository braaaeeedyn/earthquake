# Product

## Register

brand

## Users
Three audiences share one homepage, in roughly this order of arrival:
- **Portfolio reviewers** (recruiters, engineers, faculty) judging the ML and engineering work. They skim:
  what does it do, does it work, how do we know.
- **Seismology / ML peers**, who want the method, the baselines and the honest limits, and will open the evidence.
- **Southern California residents**, who want to know whether it covers where they live and how to get alerts
  (alerts live in the Android app).

## Product Purpose
SeismicSoCal detects, locates and sizes Southern California earthquakes from a live 19-station seismic stream
with two deep models (Detect, Size), each shown against the classic seismology baseline on held-out data and
on replayed real days, and pushes two-stage alerts to people who follow nearby sensors. Success = a visitor
leaves understanding what the system does, trusting the numbers because the evidence and caveats are visible,
and (if local) knowing whether they're covered.

## Brand Personality
Editorial and explanatory: a well-made science explainer. Calm, plain-spoken, exact. It walks the reader from
"what is this" to "how do we know" without hype. Every number comes with its baseline, its confidence interval
or its caveat. Voice words: lucid, candid, measured.

## Anti-references
- Hype AI landing pages: gradient text, glowing orbs, "revolutionary" copy, hero-metric templates.
- Doom / disaster imagery: red alarm banners, cracked-earth photos, panic tone. This is not an official warning
  system and must never look like one.
- Generic SaaS dashboards and identical icon-card grids.
- Claims of prediction. The site explains detection after a quake begins.

## Design Principles
1. **Show the evidence next to the claim.** Baselines, CIs and replay results sit beside every headline number.
2. **Explain in order.** The page follows the physical sequence (wave → detect → locate → size → alert) so
   non-experts can follow and experts can check.
3. **Honest limits are content, not fine print.** Coverage gaps, latency and "not a warning system" are stated
   plainly where they matter.
4. **Real data only.** Every number, map and figure comes from the actual models, network and replays.
5. **Calm under load.** Nothing alarmist; motion and emphasis are reserved for what is genuinely live.

## Accessibility & Inclusion
WCAG 2.2 AA: body text ≥ 4.5:1, never color as the only signal (status always has a text label),
`prefers-reduced-motion` honored for every animation (the seismograph trace and carousels included), keyboard
reachable controls, works at 320 px width.
