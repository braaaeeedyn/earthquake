// Seismic results contract — produced from the verified experiment runs (see scripts/).
export type Winner = 'deep' | 'tie' | 'baseline'

export interface Task {
  key: string
  name: string
  metric: string
  deep: number
  baseline: number
  baseline_name: string
  winner: Winner
  desc: string
}

export interface Seismic {
  project: string
  model: string
  generated_at: string
  dataset: {
    events: number
    stations: number
    mag_min: number
    mag_max: number
    region: string
  }
  tasks: Task[]
  stations: string[]
  disclaimer: string
}

// Relative URL (no leading slash) so it resolves under base './' on web and in the
// Capacitor WebView. Cache-busted so refreshed results are picked up.
export async function loadSeismic(): Promise<Seismic> {
  const res = await fetch(`seismic.json?t=${Date.now()}`, { cache: 'no-store' })
  if (!res.ok) throw new Error(`Failed to load results (HTTP ${res.status})`)
  return (await res.json()) as Seismic
}
