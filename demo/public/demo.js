// demo.js — Client-side logic for mojo-addon-examples live demo

// --- Dark mode toggle (from _hyper_min) ---

function toggleDarkMode() {
  const html = document.documentElement;
  const isDark = html.classList.toggle('-dark-mode');
  localStorage.setItem('mojo-demo-dark-mode', isDark ? 'dark' : 'light');
  updateDarkModeIcon();
}

function updateDarkModeIcon() {
  const isDark = document.documentElement.classList.contains('-dark-mode');
  const lightIcon = document.querySelector('.light-icon');
  const darkIcon = document.querySelector('.dark-icon');
  if (lightIcon) lightIcon.style.display = isDark ? 'none' : 'inline-block';
  if (darkIcon) darkIcon.style.display = isDark ? 'inline-block' : 'none';
}

document.getElementById('dark-mode-btn').addEventListener('click', toggleDarkMode);

// Restore saved preference
const savedMode = localStorage.getItem('mojo-demo-dark-mode');
const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
if (savedMode === 'dark' || (!savedMode && prefersDark)) {
  document.documentElement.classList.add('-dark-mode');
}
updateDarkModeIcon();

// Keyboard shortcut: Cmd/Ctrl+D for dark mode
document.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'd') {
    e.preventDefault();
    toggleDarkMode();
  }
});

// --- Code toggle ---

function toggleCode(id) {
  const block = document.getElementById(id);
  if (block) block.classList.toggle('open');
}

// Make toggleCode available globally for onclick handlers
window.toggleCode = toggleCode;

// --- Utility ---

function formatMs(ms) {
  if (ms < 0.01) return '<0.01ms';
  if (ms < 1) return ms.toFixed(2) + 'ms';
  return ms.toFixed(1) + 'ms';
}

function formatSize(bytes) {
  if (bytes >= 1048576) return (bytes / 1048576) + 'MB';
  if (bytes >= 1024) return (bytes / 1024) + 'KB';
  return bytes + 'B';
}

// --- Fetch and render each section ---

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

// 1. Image Processing

async function loadImageDemo() {
  const el = document.getElementById('image-result');
  try {
    const data = await fetchJson('/api/image/demo');
    const t = data.transforms;

    if (data.images) {
      const speedup = (t.jsGrayscaleMs / t.grayscale.ms).toFixed(1);
      el.innerHTML = `
        <div class="image-grid">
          <div class="image-card">
            <img src="data:image/png;base64,${data.images.original}" alt="Original">
            <div class="label">Original</div>
            <div class="timing">${data.width}x${data.height} RGBA</div>
          </div>
          <div class="image-card">
            <img src="data:image/png;base64,${data.images.grayscale}" alt="Grayscale">
            <div class="label">Grayscale</div>
            <div class="timing">${formatMs(t.grayscale.ms)} <span class="timing-badge">${speedup}x vs JS</span></div>
          </div>
          <div class="image-card">
            <img src="data:image/png;base64,${data.images.brightness}" alt="Brightness">
            <div class="label">Brightness 1.5x</div>
            <div class="timing">${formatMs(t.brightness.ms)}</div>
          </div>
          <div class="image-card">
            <img src="data:image/png;base64,${data.images.threshold}" alt="Threshold">
            <div class="label">Threshold 128</div>
            <div class="timing">${formatMs(t.threshold.ms)}</div>
          </div>
          <div class="image-card">
            <img src="data:image/png;base64,${data.images.blur}" alt="Blur">
            <div class="label">Box Blur r=5</div>
            <div class="timing">${formatMs(t.blur.ms)}</div>
          </div>
        </div>`;
    } else {
      // No sharp available — show timing only
      const speedup = (t.jsGrayscaleMs / t.grayscale.ms).toFixed(1);
      el.innerHTML = `
        <div class="comparison">
          <div class="comparison-row">
            <span class="comparison-label">Grayscale</span>
            <span class="timing-badge">${formatMs(t.grayscale.ms)}</span>
          </div>
          <div class="comparison-row">
            <span class="comparison-label">Brightness</span>
            <span class="timing-badge">${formatMs(t.brightness.ms)}</span>
          </div>
          <div class="comparison-row">
            <span class="comparison-label">Threshold</span>
            <span class="timing-badge">${formatMs(t.threshold.ms)}</span>
          </div>
          <div class="comparison-row">
            <span class="comparison-label">Blur r=5</span>
            <span class="timing-badge">${formatMs(t.blur.ms)}</span>
          </div>
        </div>
        <p style="margin-top:var(--space-3);color:var(--color-muted);font-size:var(--text-sm)">
          Grayscale: ${speedup}x faster than JavaScript on ${data.width}x${data.height} image
        </p>`;
    }
  } catch (err) {
    el.innerHTML = `<p style="color:var(--color-muted)">Could not load image demo: ${err.message}</p>`;
  }
}

// 2. Statistics

async function loadStatsDemo() {
  const el = document.getElementById('stats-result');
  try {
    const data = await fetchJson('/api/stats/demo');
    const s = data.stats;
    const speedup = (data.jsMs / data.mojoStatsMs).toFixed(1);

    // Histogram bars
    const maxCount = Math.max(...data.histogram);
    const histBars = data.histogram.map(count => {
      const pct = maxCount > 0 ? (count / maxCount * 100) : 0;
      return `<div class="bar" style="height:${pct}%"></div>`;
    }).join('');

    el.innerHTML = `
      <div class="stats-grid">
        <div class="stat-card"><div class="stat-label">Mean</div><div class="stat-value">${s.mean.toFixed(2)}</div></div>
        <div class="stat-card"><div class="stat-label">Std Dev</div><div class="stat-value">${s.stddev.toFixed(2)}</div></div>
        <div class="stat-card"><div class="stat-label">Min</div><div class="stat-value">${s.min.toFixed(2)}</div></div>
        <div class="stat-card"><div class="stat-label">Max</div><div class="stat-value">${s.max.toFixed(2)}</div></div>
        <div class="stat-card"><div class="stat-label">P50</div><div class="stat-value">${s.p50.toFixed(2)}</div></div>
        <div class="stat-card"><div class="stat-label">P95</div><div class="stat-value">${s.p95.toFixed(2)}</div></div>
        <div class="stat-card"><div class="stat-label">P99</div><div class="stat-value">${s.p99.toFixed(2)}</div></div>
      </div>
      <div class="histogram">${histBars}</div>
      <div class="comparison">
        <div class="comparison-row">
          <span class="comparison-label">Mojo</span>
          <div class="comparison-bar-wrapper">
            <div class="comparison-bar bar-mojo" style="width:${Math.min(100, 100 / parseFloat(speedup))}%">${formatMs(data.mojoStatsMs)}</div>
          </div>
        </div>
        <div class="comparison-row">
          <span class="comparison-label">JavaScript</span>
          <div class="comparison-bar-wrapper">
            <div class="comparison-bar bar-js" style="width:100%">${formatMs(data.jsMs)}</div>
          </div>
        </div>
      </div>
      <p style="margin-top:var(--space-2);font-size:var(--text-sm);color:var(--color-muted)">
        ${(data.size / 1e6).toFixed(0)}M elements — Mojo: ${formatMs(data.mojoStatsMs)} + histogram ${formatMs(data.mojoHistMs)} —
        <span class="timing-badge">${speedup}x faster</span>
      </p>`;
  } catch (err) {
    el.innerHTML = `<p style="color:var(--color-muted)">Could not load stats demo: ${err.message}</p>`;
  }
}

// 3. SIMD Search

async function loadSearchDemo() {
  const el = document.getElementById('search-result');
  try {
    const data = await fetchJson('/api/search/demo');
    const lineSpeedup = (data.jsLinesMs / data.mojoLinesMs).toFixed(1);

    el.innerHTML = `
      <div class="comparison">
        <div class="comparison-row">
          <span class="comparison-label">Mojo lines</span>
          <div class="comparison-bar-wrapper">
            <div class="comparison-bar bar-mojo" style="width:${Math.min(100, 100 / parseFloat(lineSpeedup))}%">${formatMs(data.mojoLinesMs)}</div>
          </div>
        </div>
        <div class="comparison-row">
          <span class="comparison-label">JS lines</span>
          <div class="comparison-bar-wrapper">
            <div class="comparison-bar bar-js" style="width:100%">${formatMs(data.jsLinesMs)}</div>
          </div>
        </div>
      </div>
      <p style="margin-top:var(--space-3);font-size:var(--text-sm);color:var(--color-muted)">
        ${formatSize(data.bufferSize)} buffer:
        ${data.lineCount.toLocaleString()} newlines found in ${formatMs(data.mojoLinesMs)},
        ${data.matchCount.toLocaleString()} "${data.pattern}" matches in ${formatMs(data.mojoSearchMs)} —
        <span class="timing-badge">${lineSpeedup}x faster</span>
      </p>`;
  } catch (err) {
    el.innerHTML = `<p style="color:var(--color-muted)">Could not load search demo: ${err.message}</p>`;
  }
}

// 4. Matrix Multiply

async function loadMatmulDemo() {
  const el = document.getElementById('matmul-result');
  try {
    const data = await fetchJson('/api/matmul/demo');
    const jsMs = data.jsNaiveMs;
    const steps = [
      { name: 'JS Baseline', ms: jsMs, kind: 'js' },
      { name: 'Mojo Naive', ms: data.naiveMs, kind: 'mojo' },
      { name: '+ Vectorized', ms: data.vectorizedMs, kind: 'mojo' },
      { name: '+ Cache Tiled', ms: data.tiledMs, kind: 'mojo' },
      { name: '+ Parallel', ms: data.parallelMs, kind: 'mojo' },
    ];

    const maxMs = steps[0].ms;

    const ladderHtml = steps.map(step => {
      const pct = Math.max(2, (step.ms / maxMs) * 100);
      const speedup = step.kind === 'js' ? '' : `${(jsMs / step.ms).toFixed(1)}x`;
      const barClass = step.kind === 'js' ? 'bar-js' : 'bar-mojo';
      return `
        <div class="ladder-step">
          <span class="step-name">${step.name}</span>
          <div class="step-bar-wrapper">
            <div class="step-bar ${barClass}" style="width:${pct}%">${formatMs(step.ms)}</div>
          </div>
          <span class="step-speedup">${speedup}</span>
        </div>`;
    }).join('');

    el.innerHTML = `
      <div class="ladder">${ladderHtml}</div>
      <p style="margin-top:var(--space-3);font-size:var(--text-sm);color:var(--color-muted)">
        ${data.size}x${data.size} Float64 matrix multiply —
        <span class="timing-badge">${(jsMs / data.parallelMs).toFixed(0)}x total speedup</span>
      </p>`;
  } catch (err) {
    el.innerHTML = `<p style="color:var(--color-muted)">Could not load matmul demo: ${err.message}</p>`;
  }
}

// 5. Wyhash

async function loadHashDemo() {
  const el = document.getElementById('hash-result');
  try {
    const data = await fetchJson('/api/hash/demo');

    const rows = data.results.map(r => `
      <tr>
        <td>${r.sizeLabel}</td>
        <td>${r.mojoGBs.toFixed(2)}</td>
        <td>${r.jsWyhashGBs.toFixed(2)}</td>
        <td>${r.jsFnvGBs.toFixed(2)}</td>
        <td>${r.speedup}x</td>
      </tr>`).join('');

    const last = data.results[data.results.length - 1];
    el.innerHTML = `
      <table class="throughput-table">
        <thead>
          <tr><th>Buffer</th><th>Mojo wyhash</th><th>JS wyhash</th><th>JS FNV-1a</th><th>Speedup</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
      <p style="margin-top:var(--space-3);font-size:var(--text-sm);color:var(--color-muted)">
        All GB/s — speedup compares same algorithm (Mojo wyhash vs JS wyhash BigInt) —
        <span class="timing-badge">${last.speedup}x on ${last.sizeLabel}</span>
      </p>`;
  } catch (err) {
    el.innerHTML = `<p style="color:var(--color-muted)">Could not load hash demo: ${err.message}</p>`;
  }
}

// --- Load all demos on page load ---

document.addEventListener('DOMContentLoaded', () => {
  loadImageDemo();
  loadStatsDemo();
  loadSearchDemo();
  loadMatmulDemo();
  loadHashDemo();
});
