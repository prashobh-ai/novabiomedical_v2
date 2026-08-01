// ============================================================================
// Maturity rings — concentric arcs, one per metric
// ============================================================================
//
// The single ring plus five flat bars read as a form, not a finding. This
// renders the same five numbers as nested arcs: outermost is the broadest
// measure, innermost the most specific, each drawing in on a stagger so the
// composition assembles rather than appears.
//
// Why concentric rather than five separate dials:
//   * one shape, so a reviewer reads it in a single glance from across a room —
//     which is the actual viewing condition on a Teams screen share;
//   * relative arc lengths are directly comparable at a glance, whereas five
//     separate rings force the eye to compare in pairs;
//   * it scales to any container width without reflowing, so the phone and the
//     boardroom display get the identical composition.
//
// Pure inline SVG. No chart library, no canvas, ~3 KB, and it animates with CSS
// stroke-dashoffset so there is no per-frame JavaScript.

const RING_COLORS = {
  coverage:     { from: '#5B86FF', to: '#7FD8FF' },
  connectivity: { from: '#7B5BFF', to: '#B388FF' },
  provenance:   { from: '#F6B44C', to: '#FFD98A' },
  extraction:   { from: '#FF6E9C', to: '#FF2E6B' },
  freshness:    { from: '#47D6A6', to: '#8AF0CE' },
};
const FALLBACK = { from: '#5B86FF', to: '#7FD8FF' };

// Arcs run 270° with a gap at the bottom, so the start and end of each arc are
// visible and a 100% value still reads as a complete measure rather than a
// closed circle that could be anything.
const SWEEP = 270;
const START_ANGLE = 135;   // clockwise from 3 o'clock, i.e. bottom-left

function polar(cx, cy, r, deg) {
  const rad = (deg * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function arcPath(cx, cy, r, startDeg, sweepDeg) {
  const a = polar(cx, cy, r, startDeg);
  const b = polar(cx, cy, r, startDeg + sweepDeg);
  const largeArc = sweepDeg > 180 ? 1 : 0;
  return `M ${a.x.toFixed(2)} ${a.y.toFixed(2)} A ${r} ${r} 0 ${largeArc} 1 ${b.x.toFixed(2)} ${b.y.toFixed(2)}`;
}

/**
 * @param {Object} health  from computeHealth()
 * @returns {string} inline SVG
 */
export function renderMaturityRings(health) {
  if (!health || !health.metrics || !health.metrics.length) return '';

  const SIZE = 260;
  const cx = SIZE / 2, cy = SIZE / 2;
  const OUTER = 116, STEP = 17, WIDTH = 11;

  const metrics = health.metrics;
  const rings = metrics.map((m, i) => {
    const r = OUTER - i * STEP;
    const pct = Math.max(0, Math.min(100, m.value)) / 100;
    const len = (2 * Math.PI * r) * (SWEEP / 360);
    const c = RING_COLORS[m.key] || FALLBACK;
    return { m, r, pct, len, c, i, id: `mr-grad-${m.key}` };
  });

  const defs = rings.map(r => `
    <linearGradient id="${r.id}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="${r.c.from}" />
      <stop offset="100%" stop-color="${r.c.to}" />
    </linearGradient>`).join('');

  const tracks = rings.map(r => `
    <path class="mr-track" d="${arcPath(cx, cy, r.r, START_ANGLE, SWEEP)}"
          stroke-width="${WIDTH}" />`).join('');

  const arcs = rings.map(r => `
    <path class="mr-arc" d="${arcPath(cx, cy, r.r, START_ANGLE, SWEEP)}"
          stroke="url(#${r.id})" stroke-width="${WIDTH}"
          stroke-dasharray="${r.len.toFixed(1)}"
          style="--mr-len:${r.len.toFixed(1)};--mr-off:${(r.len * (1 - r.pct)).toFixed(1)};--mr-delay:${240 + r.i * 130}ms">
      <title>${escapeHtml(r.m.plainLabel || r.m.label)}: ${r.m.value} out of 100</title>
    </path>`).join('');

  // End-caps mark where each arc stops — the eye locates five values instantly
  // without having to trace each arc back to its origin.
  const caps = rings.map(r => {
    const p = polar(cx, cy, r.r, START_ANGLE + SWEEP * r.pct);
    return `<circle class="mr-cap" cx="${p.x.toFixed(2)}" cy="${p.y.toFixed(2)}" r="3.2"
              fill="${r.c.to}" style="--mr-delay:${420 + r.i * 130}ms" />`;
  }).join('');

  const legend = rings.map(r => `
    <li class="mr-legend-item" style="--mr-delay:${520 + r.i * 90}ms">
      <span class="mr-swatch" style="background:linear-gradient(135deg, ${r.c.from}, ${r.c.to})"></span>
      <span class="mr-legend-label">${escapeHtml(r.m.plainLabel || r.m.label)}</span>
      <span class="mr-legend-val">${r.m.value}</span>
    </li>`).join('');

  return `
    <div class="maturity-rings">
      <div class="mr-chart">
        <svg viewBox="0 0 ${SIZE} ${SIZE}" role="img"
             aria-label="Knowledge readiness ${health.overall} out of 100, across ${metrics.length} measures">
          <defs>${defs}</defs>
          <g class="mr-tracks">${tracks}</g>
          <g class="mr-arcs">${arcs}</g>
          <g class="mr-caps">${caps}</g>
        </svg>
        <div class="mr-center">
          <span class="mr-score" id="health-score-num">${health.overall}</span>
          <span class="mr-suffix">/100</span>
          <span class="mr-band mr-band-${health.band}">${health.band}</span>
        </div>
      </div>
      <ul class="mr-legend">${legend}</ul>
    </div>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
