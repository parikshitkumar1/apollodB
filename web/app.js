// API Configuration
let API_BASE = (location.search.match(/api=([^&]+)/) || [])[1] || '';
const DEBUG_API = location.search.includes('debug=1');

// API candidate URLs in order of preference
const apiCandidates = (() => {
  const list = [];
  
  // 1. Explicit API URL from query parameter
  const fromQuery = (location.search.match(/api=([^&]+)/) || [])[1];
  if (fromQuery) list.push(fromQuery);
  
  // 2. Current origin (for production deployment - GCP or Render)
  if (location.origin.includes('.run.app') || location.origin.includes('.onrender.com')) {
    list.push(location.origin);
  }
  
  // 3. Common deployment patterns (Render and GCP)
  // Note: Update this after Render deployment with your actual URL
  list.push('https://apollodb2-833721505155.us-central1.run.app');
  
  // 4. Local development endpoints (last resort)
  list.push('http://localhost:8080');
  list.push('http://127.0.0.1:8080');
  
  return Array.from(new Set(list));
})();

async function probe(url, attempt = 1, maxAttempts = 3, baseDelay = 1000) {
  // Normalize URL
  const baseUrl = url.replace(/\/+$/, '');
  const endpoint = `${baseUrl}/healthz`;
  const delay = baseDelay * Math.pow(2, attempt - 1);
  const debugInfo = [];
  
  const log = (...args) => {
    const message = args.join(' ');
    debugInfo.push(`[${new Date().toISOString()}] ${message}`);
    if (DEBUG_API) console.log(`[API]`, ...args);
  };
  
  log(`Probing ${endpoint} (attempt ${attempt}/${maxAttempts})`);
  
  try {
    const ctl = new AbortController();
    const timeout = setTimeout(() => {
      log(`Request to ${endpoint} timed out after 10s`);
      ctl.abort();
    }, 10000);
    
    const startTime = performance.now();
    const response = await fetch(endpoint, { 
      signal: ctl.signal,
      headers: { 
        'Cache-Control': 'no-cache',
        'X-Request-ID': `frontend-${Date.now()}-${Math.random().toString(36).substr(2, 8)}`
      },
      cache: 'no-store',
      mode: 'cors',
      credentials: 'same-origin'
    });
    const responseTime = Math.round(performance.now() - startTime);
    
    clearTimeout(timeout);
    
    // Log response headers for debugging
    const responseHeaders = {};
    response.headers.forEach((value, key) => {
      responseHeaders[key] = value;
    });
    
    log(`Response from ${endpoint}: ${response.status} (${response.statusText}) in ${responseTime}ms`);
    log('Response headers:', JSON.stringify(responseHeaders, null, 2));
    
    if (response.ok) {
      try {
        const data = await response.json();
        log(`API health check successful:`, data);
        return { success: true, debug: debugInfo.join('\n') };
      } catch (e) {
        const text = await response.text();
        log(`Failed to parse JSON response: ${e.message}, response:`, text);
        throw new Error(`Invalid JSON response: ${text.substring(0, 200)}`);
      }
    } else {
      const errorText = await response.text();
      log(`API error: ${response.status} ${response.statusText}: ${errorText}`);
      
      if (attempt < maxAttempts) {
        log(`Retrying in ${delay}ms...`);
        await new Promise(r => setTimeout(r, delay));
        return probe(url, attempt + 1, maxAttempts, baseDelay);
      }
      
      throw new Error(`API returned ${response.status}: ${errorText}`);
    }
  } catch (error) {
    const errorMessage = error.name === 'AbortError' 
      ? 'Request timed out' 
      : error.message || 'Unknown error';
      
    log(`Error probing ${endpoint}: ${errorMessage}`);
    
    if (attempt < maxAttempts) {
      log(`Retrying in ${delay}ms...`);
      await new Promise(r => setTimeout(r, delay));
      return probe(url, attempt + 1, maxAttempts, baseDelay);
    }
    
    return { 
      success: false, 
      error: errorMessage,
      debug: debugInfo.join('\n')
    };
  }
}

// Debug panel for API connection issues
function showDebugPanel(debugInfo) {
  const debugContainer = document.createElement('div');
  debugContainer.id = 'api-debug';
  debugContainer.style.cssText = `
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: #1e1e1e;
    color: #f0f0f0;
    padding: 15px;
    font-family: monospace;
    max-height: 40vh;
    overflow: auto;
    z-index: 10000;
    border-top: 2px solid #ff4444;
    font-size: 13px;
    line-height: 1.4;
  `;
  
  const debugContent = document.createElement('pre');
  debugContent.textContent = debugInfo;
  debugContent.style.margin = '0';
  debugContent.style.whiteSpace = 'pre-wrap';
  
  const closeBtn = document.createElement('button');
  closeBtn.textContent = '×';
  closeBtn.style.cssText = `
    position: absolute;
    top: 5px;
    right: 5px;
    background: #ff4444;
    color: white;
    border: none;
    border-radius: 3px;
    width: 24px;
    height: 24px;
    cursor: pointer;
    font-size: 16px;
    line-height: 1;
    padding: 0;
  `;
  closeBtn.onclick = () => debugContainer.remove();
  
  debugContainer.appendChild(closeBtn);
  debugContainer.appendChild(debugContent);
  document.body.appendChild(debugContainer);
}

window.apiReady = (async () => {
  const debugLog = [];
  const log = (...args) => {
    const message = args.map(arg => 
      typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
    ).join(' ');
    
    const timestamp = new Date().toISOString();
    debugLog.push(`[${timestamp}] ${message}`);
    console.log(`[API]`, ...args);
  };
  
  log('Starting API detection...');
  log('Location:', window.location.href);
  log('API candidates:', apiCandidates);
  
  // Check if we have a direct API URL in the query string
  const urlParams = new URLSearchParams(window.location.search);
  const apiParam = urlParams.get('api');
  
  // Build the list of candidates to try
  const candidates = [];
  if (apiParam) {
    candidates.push(apiParam);
    log(`Using API from URL parameter: ${apiParam}`);
  }
  
  // Add other candidates, avoiding duplicates
  for (const candidate of apiCandidates) {
    if (!candidates.includes(candidate)) {
      candidates.push(candidate);
    }
  }
  
  log('Testing API endpoints in order:', candidates);
  
  let lastError = null;
  
  for (const base of candidates) {
    try {
      // Basic URL validation
      if (!/^https?:\/\//i.test(base)) {
        log(`Skipping invalid URL: ${base}`);
        continue;
      }
      
      log(`\n--- Testing API endpoint: ${base} ---`);
      
      const result = await probe(base);
      
      if (result.success) {
        API_BASE = base.endsWith('/') ? base.slice(0, -1) : base;
        log(`✅ Successfully connected to API: ${API_BASE}`);
        
        // Store the working API in localStorage for next time
        try {
          localStorage.setItem('lastWorkingApi', API_BASE);
        } catch (e) {
          log('Could not save API to localStorage:', e.message);
        }
        
        setStatus?.(`Connected to API: ${API_BASE}`);
        return true;
      } else {
        log(`❌ API check failed: ${result.error || 'Unknown error'}`);
        lastError = result.error;
      }
    } catch (error) {
      log(`❌ Error checking API at ${base}:`, error.message);
      lastError = error.message;
    }
  }
  
  // If we get here, all API endpoints failed
  const errorDetails = [
    'No backend API could be reached. Please check the following:',
    '',
    '1. Ensure the backend server is running',
    '2. Check your internet connection',
    '3. Try refreshing the page',
    '',
    'You can also:',
    '• Add ?api=YOUR_BACKEND_URL to manually specify the API endpoint',
    '• Add ?debug=1 to enable detailed logging',
    '',
    'Last error:',
    lastError || 'Unknown error',
    '',
    'Debug information:',
    ...debugLog
  ].join('\n');
  
  console.error('API connection failed. Debug info:', errorDetails);
  
  // Show debug panel if in debug mode
  if (DEBUG_API) {
    showDebugPanel(errorDetails);
  }
  
  setStatus?.(`Error: Could not connect to backend API. ${DEBUG_API ? 'See debug console for details.' : 'Add ?debug=1 to the URL for more information.'}`);
  
  // Don't throw an error here, as it might prevent the UI from loading
  // Instead, we'll handle the error gracefully in the UI
  return false;
})();

const qs = (sel, root=document) => root.querySelector(sel);
const qsa = (sel, root=document) => Array.from(root.querySelectorAll(sel));

const state = {
  files: [],  // unified file list - single or multiple
  lastSingle: null,
  lastBatch: null,
  config: null,
  batchZipItems: [], // {name, blob}
};

// Analyze endpoint candidates (backend accepts all)
const ENDPOINTS = [
  '/api/analyze-single',
  '/analyze-single',
  '/api/analyze',
  '/analyze',
];

function withTimeout(promise, ms, onAbort) {
  const ctl = new AbortController();
  const t = setTimeout(() => { ctl.abort(); if (onAbort) try { onAbort(); } catch {} }, ms);
  return {
    signal: ctl.signal,
    wait: (p) => p.finally(() => clearTimeout(t))
  };
}

async function postAnalyze(fd) {
  await window.apiReady.catch((e) => { throw e; });
  let lastErr;
  for (const path of ENDPOINTS) {
    try {
      const { signal, wait } = withTimeout(Promise.resolve(), 60000);
      const res = await wait(fetch(`${API_BASE}${path}`, { method: 'POST', body: fd, signal }));
      if (!res.ok) {
        let msg = `HTTP ${res.status}`;
        try {
          const j = await res.json();
          if (j && (j.error || j.detail)) msg += `: ${j.error || j.detail}`;
        } catch {
          try { msg += `: ${await res.text()}`; } catch {}
        }
        throw new Error(msg);
      }
      return await res.json();
    } catch (e) {
      lastErr = e;
      // try next candidate
    }
  }
  throw lastErr || new Error('Analyze request failed');
}

function setStatus(text) {
  const el = qs('#status');
  if (!el) return;
  el.textContent = text || '';
  el.setAttribute('role', 'status');
  el.setAttribute('aria-live', 'polite');
}

function switchTab(tab) {
  // Tabs
  qsa('.navbar-tab').forEach(a => a.classList.toggle('active', a.dataset.tab === tab));
  // Sections
  const map = {
    analysis: '#section-analysis',
    iem: '#section-iem',
    references: '#section-references',
    about: '#section-about',
  };
  Object.entries(map).forEach(([key, sel]) => {
    const el = qs(sel);
    if (!el) return;
    const isActive = key === tab;
    el.classList.toggle('active', isActive);
    el.setAttribute('aria-hidden', String(!isActive));
    // Ensure hidden attribute reflects visibility so layout/formatting applies
    if (isActive) {
      el.removeAttribute('hidden');
      el.style.display = '';
    } else {
      el.setAttribute('hidden', '');
      el.style.display = 'none';
    }
  });
  // Extra safety: hard-hide analysis section when not active
  const analysis = qs('#section-analysis');
  if (analysis && tab !== 'analysis') { analysis.style.display = 'none'; analysis.setAttribute('hidden',''); }
  // Populate dynamic sections on demand (idempotent)
  if (tab === 'iem') {
    renderIEMDatabase(state.lastBatch?.results?.dominant_emotion || state.lastSingle?.result?.primary_emotion || null);
  }
  if (tab === 'references') {
    renderReferences();
  }
  if (tab === 'about') {
    renderAbout();
  }
  // Focus analysis controls when switching back
  if (tab === 'analysis') {
    const f = qs('#audio-files');
    if (f) f.focus({ preventScroll: true });
  }
  // Reset scroll so hidden sections don't peek through
  window.scrollTo({ top: 0, behavior: 'instant' in window ? 'instant' : 'auto' });
}

// No longer needed - unified upload handles both single and batch

function parseWavelet(eqStr) {
  // "GraphicEQ: 20 0.0; 21 0.1; ..."
  const s = eqStr.replace(/^GraphicEQ:\s*/, '');
  const parts = s.split(/;\s*/).filter(Boolean);
  const freqs = [], gains = [];
  for (const p of parts) {
    const [f, g] = p.split(/\s+/);
    const ff = parseFloat(f), gg = parseFloat(g);
    if (!isNaN(ff) && !isNaN(gg)) { freqs.push(ff); gains.push(gg); }
  }
  return { freqs, gains };
}

function renderVA(el, valence, arousal, emotion=null) {
  // Unified color scheme: cyan primary, pink accent
  const accentCyan = '#00bcd4';
  const accentPink = '#ff6fae';
  const shapes = [
    // Subtle quadrant backgrounds with consistent opacity
    { type: 'rect', x0: 0, y0: 0.5, x1: 0.5, y1: 1, fillcolor: 'rgba(255,111,174,0.08)', line: {color:'rgba(255,111,174,0.2)', width:1} },
    { type: 'rect', x0: 0.5, y0: 0.5, x1: 1, y1: 1, fillcolor: 'rgba(0,188,212,0.08)', line: {color:'rgba(0,188,212,0.2)', width:1} },
    { type: 'rect', x0: 0, y0: 0, x1: 0.5, y1: 0.5, fillcolor: 'rgba(255,255,255,0.03)', line: {color:'rgba(255,255,255,0.1)', width:1} },
    { type: 'rect', x0: 0.5, y0: 0, x1: 1, y1: 0.5, fillcolor: 'rgba(158,233,243,0.06)', line: {color:'rgba(158,233,243,0.15)', width:1} },
  ];
  const ref = { happy:[0.8,0.8], sad:[0.2,0.2], calm:[0.8,0.2], neutral:[0.5,0.5] };
  const refTraces = Object.entries(ref).map(([emo,[x,y]]) => ({
    x:[x], y:[y], mode:'markers+text', text:[emo[0].toUpperCase()+emo.slice(1)], 
    textposition: emo==='happy'?'top right' : emo==='sad'?'bottom left' : emo==='calm'?'bottom right' : 'top left',
    textfont: { color: emo===emotion ? accentPink : 'rgba(255,255,255,0.6)', size: 11 },
    marker:{ size: emo===emotion?14:7, color: emo===emotion? accentPink :'rgba(255,255,255,0.4)', 
             line: { color: 'rgba(255,255,255,0.3)', width: 1 } }, 
    showlegend:false, hoverinfo:'text', hovertext:emo
  }));
  const data = [
    ...refTraces,
    { x:[valence], y:[arousal], mode:'markers', 
      marker:{ size:18, color: accentCyan, symbol:'star', line:{color:'#fff', width:2} }, 
      name:'Your Track', hoverinfo:'text', hovertext:`V: ${valence.toFixed(2)}, A: ${arousal.toFixed(2)}` }
  ];
  const layout = {
    title: { text: 'Valence-Arousal Space', font: { size: 14, color: 'rgba(255,255,255,0.9)' } },
    xaxis: { range:[-0.02,1.02], gridcolor:'rgba(255,255,255,0.08)', zeroline:false,
             title: { text: 'Valence →', font: { size: 11, color: 'rgba(255,255,255,0.6)' } },
             tickfont: { size: 10, color: 'rgba(255,255,255,0.5)' } },
    yaxis: { range:[-0.02,1.02], gridcolor:'rgba(255,255,255,0.08)', zeroline:false,
             title: { text: 'Arousal →', font: { size: 11, color: 'rgba(255,255,255,0.6)' } },
             tickfont: { size: 10, color: 'rgba(255,255,255,0.5)' } },
    paper_bgcolor:'transparent', plot_bgcolor:'rgba(10,14,18,0.6)', 
    font:{ color:'#fff', family:'Inter, system-ui, sans-serif' }, shapes,
    margin: { l: 50, r: 20, t: 40, b: 45 },
    showlegend: false
  };
  Plotly.newPlot(el, data, layout, {responsive:true, displayModeBar:false}).then(() => {
    try { Plotly.Plots.resize(el); } catch {}
  });
}

function renderEQ(elChart, elText, eqData, emotion) {
  const accentCyan = '#00bcd4';
  const accentPink = '#ff6fae';
  const { freqs, gains } = parseWavelet(eqData.wavelet);
  // Parse parametric markers from text if available
  const markers = [];
  if (eqData && typeof eqData.parametric === 'string' && eqData.parametric.trim() && eqData.parametric !== 'Flat') {
    const lines = eqData.parametric.split(/\n+/).filter(Boolean);
    for (const line of lines) {
      const m = line.match(/^(\w+)\s+(\d+)\s*Hz:\s*([+\-]?[0-9.]+)\s*dB.*Q\s*=\s*([0-9.]+)/i);
      if (m) {
        const [, type, fc, gain, q] = m;
        markers.push({ fc: Number(fc), gain: Number(gain), q: Number(q), type });
      }
    }
  }
  const traces = [
    { x: freqs, y: gains, mode:'lines', 
      line:{ color: accentCyan, width: 2.5, shape: 'spline' }, 
      fill: 'tozeroy', fillcolor: 'rgba(0,188,212,0.1)',
      name: `${emotion} EQ`, hoverinfo: 'x+y' }
  ];
  if (markers.length) {
    traces.push({
      x: markers.map(m=>m.fc), y: markers.map(m=>m.gain), mode:'markers',
      marker:{ size:10, color: accentPink, symbol:'diamond', line:{color:'#fff', width:1} }, 
      name:'Bands', hoverinfo:'text', hovertext: markers.map(m=>`${m.type} ${m.fc}Hz`)
    });
  }
  Plotly.newPlot(elChart, traces, {
    title: { text: 'EQ Curve', font: { size: 14, color: 'rgba(255,255,255,0.9)' } },
    xaxis:{ type:'log', gridcolor:'rgba(255,255,255,0.08)', zeroline:false,
            title: { text: 'Frequency (Hz)', font: { size: 11, color: 'rgba(255,255,255,0.6)' } },
            tickfont: { size: 10, color: 'rgba(255,255,255,0.5)' } },
    yaxis:{ gridcolor:'rgba(255,255,255,0.08)', zeroline:true, zerolinecolor:'rgba(255,255,255,0.2)',
            title: { text: 'Gain (dB)', font: { size: 11, color: 'rgba(255,255,255,0.6)' } },
            tickfont: { size: 10, color: 'rgba(255,255,255,0.5)' } },
    paper_bgcolor:'transparent', plot_bgcolor:'rgba(10,14,18,0.6)', 
    font:{ color:'#fff', family:'Inter, system-ui, sans-serif' },
    margin: { l: 55, r: 20, t: 40, b: 45 },
    showlegend: false
  }, {responsive:true, displayModeBar:false}).then(() => {
    try { Plotly.Plots.resize(elChart); } catch {}
  });
  elText.textContent = eqData.wavelet;
}

function updateEqView(style, elChart, elText, eqData, emotion) {
  const accentCyan = '#00bcd4';
  const accentPink = '#ff6fae';
  const val = String(style || 'Wavelet');
  if (val === 'Parametric') {
    elText.textContent = eqData.parametric || 'Flat';
  } else {
    elText.textContent = eqData.wavelet || '';
  }
  const { freqs, gains } = parseWavelet(eqData.wavelet || '');
  const traces = [{ x: freqs, y: gains, mode:'lines', 
    line:{ color: accentCyan, width: 2.5, shape: 'spline' }, 
    fill: 'tozeroy', fillcolor: 'rgba(0,188,212,0.1)',
    name:`${emotion} EQ` }];
  if (val === 'Parametric' && eqData?.parametric?.trim() && eqData.parametric !== 'Flat') {
    const lines = eqData.parametric.split(/\n+/).filter(Boolean);
    const markers = [];
    for (const line of lines) {
      const m = line.match(/^(\w+)\s+(\d+)\s*Hz:\s*([+\-]?[0-9.]+)\s*dB.*Q\s*=\s*([0-9.]+)/i);
      if (m) markers.push({ fc: Number(m[2]), gain: Number(m[3]), type: m[1] });
    }
    if (markers.length) {
      traces.push({ x: markers.map(m=>m.fc), y: markers.map(m=>m.gain), mode:'markers',
        marker:{ size:10, color: accentPink, symbol:'diamond', line:{color:'#fff', width:1} }, 
        name:'Bands' });
    }
  }
  Plotly.newPlot(elChart, traces, {
    title: { text: 'EQ Curve', font: { size: 14, color: 'rgba(255,255,255,0.9)' } },
    xaxis:{ type:'log', gridcolor:'rgba(255,255,255,0.08)', zeroline:false,
            title: { text: 'Frequency (Hz)', font: { size: 11, color: 'rgba(255,255,255,0.6)' } },
            tickfont: { size: 10, color: 'rgba(255,255,255,0.5)' } },
    yaxis:{ gridcolor:'rgba(255,255,255,0.08)', zeroline:true, zerolinecolor:'rgba(255,255,255,0.2)',
            title: { text: 'Gain (dB)', font: { size: 11, color: 'rgba(255,255,255,0.6)' } },
            tickfont: { size: 10, color: 'rgba(255,255,255,0.5)' } },
    paper_bgcolor:'transparent', plot_bgcolor:'rgba(10,14,18,0.6)', 
    font:{ color:'#fff', family:'Inter, system-ui, sans-serif' },
    margin: { l: 55, r: 20, t: 40, b: 45 }, showlegend: false
  }, {responsive:true, displayModeBar:false}).then(() => { try { Plotly.Plots.resize(elChart); } catch {} });
}

function renderDist(el, dist) {
  const labels = Object.keys(dist);
  const values = Object.values(dist);
  // Unified color palette: cyan to pink gradient
  const colors = ['#00bcd4', '#33d1e3', '#9ee9f3', '#ffd1e5', '#ff6fae'];
  Plotly.newPlot(el, [{type:'pie', labels, values, hole:0.45, 
    marker:{ colors: colors, line:{color:'rgba(10,14,18,0.8)', width:2} },
    textfont: { color: '#fff', size: 11 },
    hoverinfo: 'label+percent'
  }], {
    title: { text: 'Emotion Distribution', font: { size: 14, color: 'rgba(255,255,255,0.9)' } },
    paper_bgcolor:'transparent', plot_bgcolor:'transparent', 
    font:{ color:'#fff', family:'Inter, system-ui, sans-serif' },
    margin: { l: 20, r: 20, t: 40, b: 20 },
    showlegend: true, legend: { font: { color: 'rgba(255,255,255,0.7)', size: 10 } }
  }, {responsive:true, displayModeBar:false}).then(() => {
    try { Plotly.Plots.resize(el); } catch {}
  });
}

function downloadText(filename, text) {
  const blob = new Blob([text], {type:'text/plain'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

async function analyzeSingle() {
  const file = state.files[0];
  if (!file) { setStatus('Please select an audio file.'); return; }
  const btn = qs('#analyze-btn');
  if (btn) btn.disabled = true;
  setStatus('Analyzing…');
  try {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('aggression', qs('#agg').value);
    fd.append('eq_style', qs('#eq-style').value);
    const data = await postAnalyze(fd);
    state.lastSingle = data; // contains result and eq_data
    setStatus('');
    renderSingleResult(file, data);
  } catch (err) {
    console.error(err);
    setStatus(`Error analyzing file: ${err.message || err}`);
  } finally {
    if (btn) btn.disabled = false;
  }
}

function renderSingleResult(file, data) {
  const container = qs('#results');
  container.innerHTML = '';
  const tpl = qs('#single-template').content.cloneNode(true);
  container.appendChild(tpl);
  container.style.display = '';

  const r = data.result;
  qs('#emotion').textContent = (r.primary_emotion||'').toUpperCase();
  if (qs('#secondary')) qs('#secondary').textContent = (r.secondary_emotion||'').toUpperCase();
  qs('#valence').textContent = (r.valence||0).toFixed(2);
  qs('#arousal').textContent = (r.arousal||0).toFixed(2);
  qs('#confidence').textContent = (r.confidence||0).toFixed(2);

  renderVA(qs('#va-chart'), r.valence, r.arousal, r.primary_emotion);
  const eqChartEl = qs('#eq-chart');
  const eqTextEl = qs('#eq-text');
  renderEQ(eqChartEl, eqTextEl, data.eq_data, r.primary_emotion);
  // Spacing between EQ text and its download button
  const dlEqBtn = qs('#download-eq-text');
  if (dlEqBtn) dlEqBtn.style.marginTop = '12px';
  if (eqTextEl) eqTextEl.style.marginBottom = '12px';

  // React to EQ style changes without re-analyze
  const styleSel = qs('#eq-style');
  if (styleSel) {
    updateEqView(styleSel.value, eqChartEl, eqTextEl, data.eq_data, r.primary_emotion);
    styleSel.addEventListener('change', () => updateEqView(styleSel.value, eqChartEl, eqTextEl, data.eq_data, r.primary_emotion));
  }

  // Stabilize charts after becoming visible
  requestAnimationFrame(() => {
    qsa('.chart', container).forEach(el => { try { Plotly.Plots.resize(el); } catch {} });
    setTimeout(() => { qsa('.chart', container).forEach(el => { try { Plotly.Plots.resize(el); } catch {} }); }, 200);
  });

  // Download EQ text
  qs('#download-eq-text').addEventListener('click', () => {
    const style = qs('#eq-style')?.value || 'Wavelet';
    let content = data.eq_data.wavelet;
    if (style === 'Parametric' && data.eq_data.parametric) content = data.eq_data.parametric;
    downloadText(`eq_${r.primary_emotion}_${style.toLowerCase()}.txt`, content);
  });

  // EQ export buttons (single)
  const exportApo = qs('#export-apo');
  const exportJson = qs('#export-autoeq-json');
  const exportCsv = qs('#export-autoeq-csv');
  async function doExport(fmt) {
    try {
      const fd = new FormData();
      fd.append('emotion', r.primary_emotion);
      fd.append('aggression', qs('#agg').value);
      fd.append('fmt', fmt);
      if (typeof r.valence === 'number') fd.append('valence', String(r.valence));
      if (typeof r.arousal === 'number') fd.append('arousal', String(r.arousal));
      if (typeof r.confidence === 'number') fd.append('confidence', String(r.confidence));
      if (r.secondary_emotion) fd.append('secondary', String(r.secondary_emotion));
      const resp = await fetch(`${API_BASE}/export-eq`, { method:'POST', body: fd });
      if (!resp.ok) throw new Error('export failed');
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      const base = `eq_${r.primary_emotion}`;
      const name = fmt === 'apo' ? `${base}_apo.txt` : (fmt === 'autoeq_json' ? `${base}_autoeq.json` : `${base}_autoeq.csv`);
      a.href = url; a.download = name; a.click(); URL.revokeObjectURL(url);
    } catch (e) { setStatus('Export failed.'); }
  }
  if (exportApo) exportApo.onclick = () => doExport('apo');
  if (exportJson) exportJson.onclick = () => doExport('autoeq_json');
  if (exportCsv) exportCsv.onclick = () => doExport('autoeq_csv');
}

async function analyzeBatch() {
  const files = state.files;
  if (!files.length) { setStatus('Please select audio files.'); return; }
  const btn = qs('#analyze-btn');
  if (btn) btn.disabled = true;
  setStatus('Analyzing batch…');
  try {
    const fd = new FormData();
    for (const f of files) fd.append('files', f);
    fd.append('aggression', qs('#agg').value);
    fd.append('eq_style', qs('#eq-style').value);
    const { signal, wait } = withTimeout(Promise.resolve(), 120000);
    const res = await wait(fetch(`${API_BASE}/analyze-batch`, { method:'POST', body: fd, signal }));
    if (!res.ok) {
      const txt = await res.text().catch(()=> '');
      throw new Error(`Batch analysis failed: HTTP ${res.status}${txt?': '+txt:''}`);
    }
    const data = await res.json();
    state.lastBatch = data;
    setStatus('');
    renderBatchResult(files, data);
  } catch (err) {
    console.error(err);
    setStatus(err.message || 'Batch analysis failed');
  } finally {
    if (btn) btn.disabled = false;
  }
}

function renderBatchResult(files, data) {
  const container = qs('#results');
  container.innerHTML = '';
  const tpl = qs('#batch-template').content.cloneNode(true);
  container.appendChild(tpl);
  container.style.display = '';

  const r = data.results;
  const elSongs = qs('#songs'); if (elSongs) elSongs.textContent = r.total_songs;
  const elDom = qs('#dom-emotion'); if (elDom) elDom.textContent = (r.dominant_emotion||'').toUpperCase();
  const elVal = qs('#avg-valence'); if (elVal) elVal.textContent = (r.average_valence||0).toFixed(2);
  const elAro = qs('#avg-arousal'); if (elAro) elAro.textContent = (r.average_arousal||0).toFixed(2);

  renderVA(qs('#va-chart'), r.average_valence, r.average_arousal, r.dominant_emotion);
  if (Object.keys(r.emotion_distribution||{}).length > 1) {
    renderDist(qs('#dist-chart'), r.emotion_distribution);
  } else {
    qs('#dist-chart').innerHTML = '<div class="status">All songs have the same emotion classification.</div>';
  }

  const eqChart = qs('#eq-chart');
  const eqText = qs('#eq-text') || document.createElement('div');
  if (eqChart) renderEQ(eqChart, eqText, data.eq_data, r.dominant_emotion);

  // Stabilize charts after becoming visible
  requestAnimationFrame(() => {
    qsa('.chart', container).forEach(el => { try { Plotly.Plots.resize(el); } catch {} });
    setTimeout(() => { qsa('.chart', container).forEach(el => { try { Plotly.Plots.resize(el); } catch {} }); }, 200);
  });
  const dlEqTxt = qs('#download-eq-text');
  if (dlEqTxt) dlEqTxt.addEventListener('click', () => {
    downloadText(`batch_eq_${r.dominant_emotion}_wavelet.txt`, data.eq_data.wavelet);
  });
  // Batch APO export (aggregate emotion)
  const batchApo = qs('#batch-export-apo');
  if (batchApo) {
    batchApo.onclick = async () => {
      try {
        const fd = new FormData();
        fd.append('emotion', r.dominant_emotion);
        fd.append('aggression', qs('#agg').value);
        fd.append('fmt', 'apo');
        if (typeof r.average_valence === 'number') fd.append('valence', String(r.average_valence));
        if (typeof r.average_arousal === 'number') fd.append('arousal', String(r.average_arousal));
        const resp = await fetch(`${API_BASE}/export-eq`, { method:'POST', body: fd });
        if (!resp.ok) throw new Error('export failed');
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = `batch_eq_${r.dominant_emotion}_apo.txt`; a.click(); URL.revokeObjectURL(url);
      } catch (e) { setStatus('Batch APO export failed.'); }
    };
  }
  // Server-side ZIP for batch EQ
  const serverZip = qs('#server-zip');
  if (serverZip) {
    serverZip.onclick = async () => {
      try {
        setStatus('Preparing server ZIP...');
        const fd = new FormData();
        for (const f of files) fd.append('files', f);
        const emotions = (r.individual_results||[]).map(it => it.primary_emotion || r.dominant_emotion);
        emotions.forEach(e => fd.append('emotions', e));
        const confidences = (r.individual_results||[]).map(it => typeof it.confidence === 'number' ? String(it.confidence) : '');
        confidences.forEach(c => fd.append('confidences', c));
        fd.append('aggression', qs('#agg').value);
        fd.append('warmth', String(Number(qs('#warmth')?.value || 0)));
        fd.append('presence', String(Number(qs('#presence')?.value || 0)));
        fd.append('air', String(Number(qs('#air')?.value || 0)));
        fd.append('hq_linear', String(!!qs('#hq-linear')?.checked));
        let resp = await fetch(`${API_BASE}/apply-eq-batch-zip`, { method:'POST', body: fd });
        if (!resp.ok) {
          const t = await resp.text().catch(()=>"\u0000");
          throw new Error(`server zip failed: ${resp.status} ${t}`);
        }
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = `apollodb_batch_eq_${(r.dominant_emotion||'batch')}.zip`;
        a.click(); URL.revokeObjectURL(url);
        setStatus('');
      } catch (e) {
        console.error(e); setStatus('Server ZIP failed. Falling back to client ZIP.');
      }
    };
  }

  // Table
  const tableDiv = qs('#table');
  const rows = (r.individual_results||[]).map((it, i) => (
    `<tr>
      <td class="num">${i+1}</td>
      <td class="song">${it.filename||files[i]?.name||''}</td>
      <td>${(it.primary_emotion||'').toUpperCase()}</td>
      <td>${(it.secondary_emotion||'').toUpperCase()}</td>
      <td class="num">${(it.valence||0).toFixed(2)}</td>
      <td class="num">${(it.arousal||0).toFixed(2)}</td>
      <td class="num">${(it.confidence||0).toFixed(2)}</td>
    </tr>`)).join('');
  if (tableDiv) {
    tableDiv.innerHTML = `<div class="premium-card"><div class="table-wrap">
      <table class="results-table">
        <thead>
          <tr>
            <th class="num">#</th><th class="song">Song</th><th>Primary</th><th>Secondary</th><th class="num">Valence</th><th class="num">Arousal</th><th class="num">Confidence</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div></div>`;
  }

  // Individual Downloads (auto-generate EQed audio for each item) with throttling and ZIP collection
  const downloadsDiv = qs('#individual-downloads');
  downloadsDiv.innerHTML = '';
  state.batchZipItems = [];
  const tasks = (r.individual_results||[]).map((it, i) => ({ it, i }));
  const CONCURRENCY = 3;
  let idx = 0;
  async function worker() {
    while (idx < tasks.length) {
      const cur = idx++;
      const { it, i } = tasks[cur];
      const wrap = document.createElement('div');
      const btn = document.createElement('button');
      const fname = it.filename || files[i]?.name || `song_${i+1}.wav`;
      btn.textContent = `${fname} - ${(it.primary_emotion||'').toUpperCase()} (rendering...)`;
      btn.disabled = true;
      wrap.appendChild(btn);
      downloadsDiv.appendChild(wrap);
      try {
        const fd = new FormData();
        fd.append('file', files[i]);
        fd.append('emotion', it.primary_emotion);
        fd.append('aggression', qs('#agg').value);
        if (typeof it.confidence === 'number') fd.append('confidence', String(it.confidence));
        // Macros
        const warmth = Number(qs('#warmth')?.value || 0);
        const presence = Number(qs('#presence')?.value || 0);
        const air = Number(qs('#air')?.value || 0);
        fd.append('warmth', String(warmth));
        fd.append('presence', String(presence));
        fd.append('air', String(air));
        const hq = !!qs('#hq-linear')?.checked;
        fd.append('hq_linear', String(hq));
        let resp = await fetch(`${API_BASE}/apply-eq`, { method:'POST', body: fd });
        if (!resp.ok) {
          const t = await resp.text().catch(()=>' ');
          console.error('apply-eq failed (batch):', resp.status, t, 'file index', i);
          // retry once
          resp = await fetch(`${API_BASE}/apply-eq`, { method:'POST', body: fd });
          if (!resp.ok) {
            const t2 = await resp.text().catch(()=>' ');
            throw new Error(`apply-eq failed (batch): ${resp.status} ${t2}`);
          }
        }
        const lufs = resp.headers.get('X-LUFS');
        const dbtp = resp.headers.get('X-True-Peak-DBTP');
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const out = `eqed_${it.primary_emotion}_${(fname.split('.')[0])}.wav`;
        // Replace button with anchor for reliable download
        const a = document.createElement('a');
        a.textContent = `${fname} - ${(it.primary_emotion||'').toUpperCase()} (ready)`;
        a.href = url; a.download = out; a.role = 'button';
        a.className = btn.className || '';
        wrap.replaceChild(a, btn);
        if (lufs || dbtp) {
          const stats = document.createElement('span');
          stats.className = 'muted small'; stats.style.marginLeft = '8px';
          const parts = [];
          if (lufs) parts.push(`LUFS ${Number(lufs).toFixed(1)}`);
          if (dbtp) parts.push(`dBTP ${Number(dbtp).toFixed(2)}`);
          stats.textContent = `[${parts.join(', ')}]`;
          a.after(stats);
        }
        // collect for ZIP
        state.batchZipItems.push({ name: out, blob });
      } catch (e) {
        console.error(e);
        btn.disabled = false; btn.textContent = `${fname} - ${(it.primary_emotion||'').toUpperCase()} (failed - retry)`;
        btn.onclick = () => {
          // simple retry via click
          idx = Math.min(idx, cur); // re-try this index
        };
      }
    }
  }
  const workers = Array.from({length: Math.min(CONCURRENCY, tasks.length)}, () => worker());
  Promise.all(workers).catch(()=>{});

  // Batch EQ Summary Report
  const dlReport = qs('#download-report');
  if (dlReport) dlReport.addEventListener('click', () => {
    let report = '';
    report += 'ApollodB Batch EQ Analysis Summary\n';
    report += '=====================================\n\n';
    report += 'Batch Analysis Results:\n';
    report += `- Total Songs: ${r.total_songs}\n`;
    report += `- Dominant Emotion: ${(r.dominant_emotion||'').toUpperCase()}\n`;
    report += `- Average Valence: ${(r.average_valence||0).toFixed(2)}\n`;
    report += `- Average Arousal: ${(r.average_arousal||0).toFixed(2)}\n`;
    report += `- EQ Aggression: ${qs('#agg').value}\n`;
    report += `- EQ Style: ${qs('#eq-style').value}\n\n`;
    report += `Recommended EQ Settings for ${(r.dominant_emotion||'').toUpperCase()} Music:\n`;
    report += '==================================================\n';
    report += `${data.eq_data.wavelet}\n\n`;
    report += 'Individual Song Analysis:\n';
    report += '=========================\n';
    (r.individual_results||[]).forEach((it, i) => {
      const fname = it.filename || files[i]?.name || `song_${i+1}.wav`;
      report += `${i+1}. ${fname}\n`;
      report += `   Primary Emotion: ${(it.primary_emotion||'').toUpperCase()}\n`;
      report += `   Secondary Emotion: ${(it.secondary_emotion||'').toUpperCase()}\n`;
      report += `   Valence: ${(it.valence||0).toFixed(2)}\n`;
      report += `   Arousal: ${(it.arousal||0).toFixed(2)}\n`;
      report += `   Confidence: ${(it.confidence||0).toFixed(2)}\n\n`;
    });
    downloadText(`apollodb_batch_analysis_${r.dominant_emotion||'summary'}.txt`, report);
  });
  // No spotify link in batch template; skip if absent
  const sp = qs('#spotify-link');
  if (sp) {
    const q = encodeURIComponent(`${r.dominant_emotion} mix`);
    sp.href = `https://open.spotify.com/search/${q}`;
  }
}

function wireUI() {
  // Ensure sliders are full-width for better small-window control (after DOM + helpers ready)
  ['#agg', '#warmth', '#presence', '#air'].forEach(sel => { const el = qs(sel); if (el) el.classList.add('wide'); });
  // Top tabs
  qsa('.navbar-tab').forEach(tab => tab.addEventListener('click', (e) => {
    e.preventDefault();
    switchTab(tab.dataset.tab);
  }));
  switchTab('analysis');

  qs('#agg').addEventListener('input', () => qs('#agg-val').textContent = qs('#agg').value);
  // Make macro sliders reactive on input as well
  ['#warmth', '#presence', '#air'].forEach(sel => {
    const el = qs(sel);
    if (el) el.addEventListener('input', () => {
      if (sel === '#warmth') qs('#warmth-val').textContent = Number(el.value).toFixed(1);
      if (sel === '#presence') qs('#presence-val').textContent = Number(el.value).toFixed(1);
      if (sel === '#air') qs('#air-val').textContent = Number(el.value).toFixed(1);
    });
  });

  // Unified file input handler
  function updateFileDisplay(displayEl, files) {
    if (!displayEl) return;
    if (!files || files.length === 0) {
      displayEl.classList.remove('has-file');
      displayEl.textContent = '';
    } else if (files.length === 1) {
      displayEl.classList.add('has-file');
      displayEl.textContent = files[0].name;
    } else {
      displayEl.classList.add('has-file');
      displayEl.textContent = `${files.length} files selected`;
    }
  }
  
  const fileInput = qs('#audio-files');
  if (fileInput) {
    fileInput.addEventListener('change', e => { 
      state.files = Array.from(e.target.files);
      updateFileDisplay(qs('#selected-files'), state.files);
    });
  }

  // Unified dropzone
  const dropzone = qs('#dropzone');
  const display = qs('#selected-files');
  if (dropzone && fileInput) {
    dropzone.addEventListener('click', () => fileInput.click());
    
    ['dragenter','dragover'].forEach(evt => dropzone.addEventListener(evt, (e) => { 
      e.preventDefault(); e.stopPropagation(); 
      dropzone.classList.add('drag-over'); 
    }));
    ['dragleave','drop'].forEach(evt => dropzone.addEventListener(evt, (e) => { 
      e.preventDefault(); e.stopPropagation(); 
      dropzone.classList.remove('drag-over'); 
    }));
    dropzone.addEventListener('drop', (e) => {
      const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('audio/') || /\.(mp3|wav|flac|ogg|m4a|aac)$/i.test(f.name));
      if (!files.length) { setStatus('Please drop audio files.'); return; }
      state.files = files;
      const dt = new DataTransfer(); files.forEach(f => dt.items.add(f)); fileInput.files = dt.files;
      updateFileDisplay(display, files);
    });
  }

  // Persist and restore settings
  const SETTINGS_KEY = 'apollodb_settings';
  function saveSettings() {
    const obj = {
      agg: qs('#agg')?.value,
      eq: qs('#eq-style')?.value,
      warmth: qs('#warmth')?.value,
      presence: qs('#presence')?.value,
      air: qs('#air')?.value,
      hq: !!qs('#hq-linear')?.checked,
    };
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(obj)); } catch {}
  }
  function restoreSettings() {
    try {
      const raw = localStorage.getItem(SETTINGS_KEY);
      if (!raw) return;
      const obj = JSON.parse(raw);
      if (obj.agg != null) { qs('#agg').value = obj.agg; qs('#agg-val').textContent = obj.agg; }
      if (obj.eq) qs('#eq-style').value = obj.eq;
      if (obj.warmth != null) { qs('#warmth').value = obj.warmth; qs('#warmth-val').textContent = Number(obj.warmth).toFixed(1); }
      if (obj.presence != null) { qs('#presence').value = obj.presence; qs('#presence-val').textContent = Number(obj.presence).toFixed(1); }
      if (obj.air != null) { qs('#air').value = obj.air; qs('#air-val').textContent = Number(obj.air).toFixed(1); }
      if (obj.hq != null) qs('#hq-linear').checked = !!obj.hq;
    } catch {}
  }
  restoreSettings();
  ['#agg','#eq-style','#warmth','#presence','#air','#hq-linear'].forEach(sel => {
    const el = qs(sel);
    if (el) el.addEventListener('change', () => { 
      if (sel === '#warmth') qs('#warmth-val').textContent = Number(el.value).toFixed(1);
      if (sel === '#presence') qs('#presence-val').textContent = Number(el.value).toFixed(1);
      if (sel === '#air') qs('#air-val').textContent = Number(el.value).toFixed(1);
      if (sel === '#agg') qs('#agg-val').textContent = el.value;
      saveSettings();
    });
  });

  // Global Export buttons (use last result if available)
  async function exportEq(fmt) {
    try {
      const last = state.lastSingle || state.lastBatch;
      if (!last) { setStatus('Run an analysis first.'); return; }
      let emotion, valence, arousal, confidence, secondary;
      if (state.lastSingle?.result) {
        emotion = state.lastSingle.result.primary_emotion;
        valence = state.lastSingle.result.valence;
        arousal = state.lastSingle.result.arousal;
        confidence = state.lastSingle.result.confidence;
        secondary = state.lastSingle.result.secondary_emotion;
      } else if (state.lastBatch?.results) {
        emotion = state.lastBatch.results.dominant_emotion;
        valence = state.lastBatch.results.average_valence;
        arousal = state.lastBatch.results.average_arousal;
      }
      if (!emotion) { setStatus('No emotion available to export.'); return; }
      const fd = new FormData();
      fd.append('emotion', emotion);
      fd.append('aggression', qs('#agg').value);
      if (typeof valence === 'number') fd.append('valence', String(valence));
      if (typeof arousal === 'number') fd.append('arousal', String(arousal));
      if (typeof confidence === 'number') fd.append('confidence', String(confidence));
      if (secondary) fd.append('secondary', String(secondary));
      fd.append('fmt', fmt);
      setStatus('Preparing EQ export…');
      const resp = await fetch(`${API_BASE}/export-eq`, { method:'POST', body: fd });
      if (!resp.ok) {
        const t = await resp.text().catch(()=>'');
        throw new Error(`Export failed: ${resp.status} ${t}`);
      }
      const blob = await resp.blob();
      const a = document.createElement('a');
      const map = { apo: 'apo.txt', autoeq_json: 'autoeq.json', autoeq_csv: 'autoeq.csv' };
      a.download = `eq_${emotion}_${map[fmt]||'export.txt'}`;
      a.href = URL.createObjectURL(blob);
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 30000);
      setStatus('');
    } catch (e) {
      console.error(e);
      setStatus(e.message || 'Export failed');
    }
  }
  const btnApo = qs('#export-apo');
  if (btnApo) btnApo.addEventListener('click', () => exportEq('apo'));
  const btnJson = qs('#export-autoeq-json');
  if (btnJson) btnJson.addEventListener('click', () => exportEq('autoeq_json'));
  const btnCsv = qs('#export-autoeq-csv');
  if (btnCsv) btnCsv.addEventListener('click', () => exportEq('autoeq_csv'));

  qs('#analyze-btn').addEventListener('click', () => {
    if (!state.files.length) { setStatus('Please select audio files.'); return; }
    if (state.files.length === 1) analyzeSingle();
    else analyzeBatch();
  });

  // Keyboard shortcut: press 'A' to analyze
  document.addEventListener('keydown', (e) => {
    if (e.key.toLowerCase() === 'a' && !e.metaKey && !e.ctrlKey && !e.altKey) {
      const active = qs('#section-analysis');
      if (active && !active.hasAttribute('hidden')) qs('#analyze-btn').click();
    }
  });
}

function initTabs() {
  const tabs = {
    analysis: qs('#analysis-tab'),
    iem: qs('#iem-db-tab'),
    faq: qs('#faq-tab'),
    refs: qs('#references-tab'),
    about: qs('#about-tab'),
  };
  ['analysis','iem','refs','about'].forEach(id => {
    const link = qs(`#nav-${id}`);
    if (link) link.addEventListener('click', () => showTab(id));
  });
  function showTab(id) {
    Object.values(tabs).forEach(el => el && el.setAttribute('hidden',''));
    if (tabs[id]) tabs[id].removeAttribute('hidden');
  }
}

// --- Static sections: IEM DB, FAQ, References ---
async function loadConfig() {
  const candidates = ['config.json', '../config.json', '/config.json'];
  for (const path of candidates) {
    try {
      const resp = await fetch(path, { cache: 'no-cache' });
      if (resp.ok) { state.config = await resp.json(); return; }
    } catch (e) { /* try next */ }
  }
  console.warn('config.json not found via any path; using embedded defaults');
}

async function renderStaticContent() {
  if (!state.config) await loadConfig();
  renderFAQ();
  renderReferences();
  renderIEMDatabase();
  renderAbout();
}

function getIEMData() {
  // Use original IEMs from config.json
  let iems = state.config?.iems;
  if (!iems) {
    // Embedded fallback mirrors original config.json
    iems = {
      // TOTL/Summit-Fi
      "LETSHUOER MYSTIC 8": { price: "$1000", signature: "Vocal benchmark", valence_match: ["happy", "calm"], notes: "Female vocal benchmark under $2000, clean & airy" },
      "Thieaudio Monarch MK4": { price: "$1150", signature: "All-rounder", valence_match: ["neutral", "happy"], notes: "Smooth, clean, balanced with tuning options" },
      // High Value Stars
      "XENNS TOP PRO": { price: "$499", signature: "Technical all-rounder", valence_match: ["neutral", "happy"], notes: "Amazing resolution & detail, direct upgrade from Astrals" },
      "ZIIGAAT Arcanis": { price: "$399", signature: "Vocal-centric", valence_match: ["calm", "sad"], notes: "Best vocals under $500, genre-specific for slower tracks" },
      "ZIIGAAT Luna": { price: "$379", signature: "Warm & dreamy", valence_match: ["calm", "neutral"], notes: "Airy, smooth, lush vibes - great for rock/metal" },
      "ZIIGAAT Odyssey": { price: "$229", signature: "Musical journey", valence_match: ["calm", "sad"], notes: "Mini Subtonic Storm that scales, immersive for indie/ballads" },
      "Softears Volume S": { price: "$319", signature: "Vocal scaling", valence_match: ["happy", "calm"], notes: "Amazing vocal scaling at high volume" },
      "Kiwi Ears x HBB PUNCH": { price: "$449", signature: "Balanced basshead", valence_match: ["happy", "neutral"], notes: "Endgame bass with extended vocals & treble" },
      // Great Value
      "Kiwi Ears Astral": { price: "$299", signature: "Balanced fun", valence_match: ["neutral", "happy"], notes: "Great all-rounder, airy with good sub-bass" },
      "SIMGOT SUPERMIX4": { price: "$150", signature: "Smooth Harman", valence_match: ["neutral", "calm"], notes: "Endgame Harman, one of the smoothest IEMs" },
      "Truthear NOVA": { price: "$150", signature: "Clean Harman", valence_match: ["neutral", "happy"], notes: "Pinnacle of trying not to offend anyone, smooth treble" },
      "ARTTI T10": { price: "$50", signature: "Planar value", valence_match: ["neutral", "happy"], notes: "Insane value, almost identical to S12" },
      // Budget Champions
      "TangZu Xuan Wu Gate": { price: "$650", signature: "Clean technical", valence_match: ["neutral", "calm"], notes: "Neutral tuning done right, very detailed" },
      "HIDIZS MK12": { price: "$129-209", signature: "High-volume scaling", valence_match: ["calm", "sad"], notes: "Insane scaling, warm & non-fatiguing" },
      "LETSHUOER DX1": { price: "$159", signature: "Dynamic vocals", valence_match: ["happy", "calm"], notes: "Vibrant vocals with good balance" },
      "ZIIGAAT LUSH": { price: "$179", signature: "Clean technical", valence_match: ["calm", "neutral"], notes: "Fuller sound with scaling, immersive" },
      "EPZ P50": { price: "$200", signature: "Clean balanced", valence_match: ["neutral", "calm"], notes: "Better tuned MEGA5EST, dynamic" },
      "Punch Audio Martilo": { price: "$329", signature: "Balanced basshead", valence_match: ["happy", "neutral"], notes: "Cheaper HBB Punch with slightly less bass" },
      "EPZ K9": { price: "$300", signature: "All-rounder v-shape", valence_match: ["neutral", "happy"], notes: "Natural vocals, nice mid-bass slam" },
      "CrinEar Meta": { price: "$249", signature: "Bright sparkly", valence_match: ["happy", "neutral"], notes: "Bright all-rounder with sparkly treble" },
      "Simgot EM6L Phoenix": { price: "$109", signature: "Smooth warm", valence_match: ["calm", "neutral"], notes: "Great resolution, slightly warm but lively" },
      "Simgot EA500LM": { price: "$89", signature: "Warm resolving", valence_match: ["calm", "happy"], notes: "Very resolving for price, warmer EA1000" },
      "Hidizs MP145": { price: "$159", signature: "Tame Harman", valence_match: ["neutral", "calm"], notes: "Less sharp HeyDay, solid all-rounder" },
      "TinHifi P1 MAX 2": { price: "$139", signature: "Smooth planar", valence_match: ["neutral", "calm"], notes: "Less bright Nova, good for jpop/kpop" },
      "MYER SLA3": { price: "$100", signature: "Dynamic engaging", valence_match: ["neutral", "happy"], notes: "Dynamic contrast, more engaging than safe picks" },
      "7hertz Sonus": { price: "$59", signature: "Neutral balanced", valence_match: ["neutral", "calm"], notes: "Hexa with more air, solid neutral set" },
      "EPZ Q1 PRO": { price: "$30", signature: "Clean balanced", valence_match: ["neutral", "calm"], notes: "OG 7hz Zero with better driver" },
      "ARTTI R2": { price: "$35", signature: "Warm balanced", valence_match: ["calm", "neutral"], notes: "Budget Tanchjim Origin, very smooth" },
      "Simgot EW200": { price: "$39", signature: "Cheaper Aria 2", valence_match: ["neutral", "calm"], notes: "Same driver as Aria 2, great value" },
      "KZ EDC PRO": { price: "$22", signature: "V-shape smooth", valence_match: ["happy", "neutral"], notes: "Nicely tuned v-shape, but it's KZ" },
      "KBear Rosefinch": { price: "$20", signature: "Budget basshead", valence_match: ["happy", "neutral"], notes: "Best budget basshead, slams like a truck" },
      "Simgot EW100P": { price: "$20", signature: "Clean neutral", valence_match: ["neutral", "calm"], notes: "Another Harman/DF, great starter" },
      "TangZu Waner 2": { price: "$20", signature: "Balanced all-rounder", valence_match: ["neutral", "calm"], notes: "More treble air than OG, great accessories" },
      "Moondrop CHU 2": { price: "$19", signature: "Harman-ish", valence_match: ["neutral", "calm"], notes: "Similar to Tanchjim One but not as smooth" },
      "Tangzu Wan'er": { price: "$20", signature: "Vocal forward", valence_match: ["happy", "calm"], notes: "Most vocal forward of $20 sets" },
      "Truthear Gate": { price: "$20", signature: "All-rounder", valence_match: ["neutral", "calm"], notes: "Less uppermids than EW100P" },
      "ZIIGAAT NUO": { price: "$25", signature: "Clean Harman", valence_match: ["neutral", "calm"], notes: "Similar to G10, heftier note-weight" },
      "TangZu Wan'er S.G": { price: "$21", signature: "Clean vocal", valence_match: ["neutral", "happy"], notes: "Cleaner Waner with more vocal emphasis" },
      "QKZ HBB": { price: "$20", signature: "Warm bassy", valence_match: ["calm", "happy"], notes: "Well tuned warm/bassy set" }
    };
  }
  const out = {};
  for (const [name, data] of Object.entries(iems)) {
    out[name] = {
      signature: data.signature,
      valence_match: data.valence_match || [],
      notes: (data.notes || data.description || ''),
      price: data.price || '',
    };
  }
  return out;
}

function renderIEMDatabase(dominantEmotion=null) {
  const iems = getIEMData();
  const featured = qs('#iem-featured');
  const tbody = qs('#iem-tbody');
  if (!featured || !tbody) return;
  featured.innerHTML = '';
  tbody.innerHTML = '';

  // Featured cards: pick top matches for dominant emotion if present
  let entries = Object.entries(iems);
  if (dominantEmotion) {
    entries = entries.sort((a,b)=>{
      const am = a[1].valence_match.includes(dominantEmotion) ? 1 : 0;
      const bm = b[1].valence_match.includes(dominantEmotion) ? 1 : 0;
      return bm - am;
    });
  }
  entries.slice(0,6).forEach(([name, data]) => {
    const card = document.createElement('div');
    card.className = 'iem-card glass';
    card.innerHTML = `
      <div class="iem-name">${name}</div>
      <div class="iem-meta">
        <span class="tag">${data.signature}</span>
        <span class="tag">Best for: ${data.valence_match.map(s=>s).join(', ')}</span>
      </div>
      <div class="iem-price">${data.price||''}</div>
    `;
    featured.appendChild(card);
  });

  // Table rendering (no notes column)
  for (const [name, data] of Object.entries(iems)) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${name}</td>
      <td>${data.signature}</td>
      <td>${(data.valence_match||[]).join(', ')}</td>
      <td>${data.price||'N/A'}</td>
    `;
    tbody.appendChild(tr);
  }

  // Filters
  const priceSel = qs('#iem-price-filter');
  const search = qs('#iem-search');
  if (priceSel && search) {
    const applyFilters = () => {
      const q = (search.value||'').toLowerCase();
      const pf = priceSel.value;
      tbody.innerHTML = '';
      Object.entries(iems).forEach(([name,data])=>{
        const text = `${name} ${data.signature}`.toLowerCase();
        if (q && !text.includes(q)) return;
        const priceNum = parseInt((data.price||'').replace(/[^0-9]/g,''))||0;
        if (pf==='under100' && !(priceNum>0 && priceNum<100)) return;
        if (pf==='100-500' && !(priceNum>=100 && priceNum<=500)) return;
        if (pf==='>500' && !(priceNum>500)) return;
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${name}</td><td>${data.signature}</td><td>${(data.valence_match||[]).join(', ')}</td><td>${data.price||'N/A'}</td>
        `;
        tbody.appendChild(tr);
      });
    };
    priceSel.addEventListener('change', applyFilters);
    search.addEventListener('input', applyFilters);
  }
}

function renderFAQ() {
  const faqs = [
    {
      q: 'How does ApollodB work?',
      a: "ApollodB uses a deep learning model trained on the DEAM dataset to analyze the emotional content of your music. It extracts spectrograms from audio files and predicts emotions based on valence and arousal dimensions. The best part? It's completely open source, so you can peek under the hood and see exactly how the magic happens!"
    },
    {
      q: "How is this better than Spotify's recommendations?",
      a: "While Spotify uses collaborative filtering and basic audio features, ApollodB performs deep emotional analysis of the actual audio content. It provides personalized EQ settings and detailed emotional profiling that goes beyond simple genre categorization. Plus, it doesn't try to sell you anything - it just wants to make your music sound amazing!"
    },
    {
      q: 'What makes this approach innovative?',
      a: "ApollodB combines state-of-the-art music emotion recognition with personalized audio engineering. It's one of the first systems to automatically generate EQ curves based on emotional analysis, bridging the gap between psychology and audio engineering. And did we mention it's open source? Because transparency in AI is everything."
    },
    {
      q: 'What audio formats are supported?',
      a: "ApollodB supports MP3, WAV, M4A, and FLAC audio formats. Files should ideally be of good quality for best analysis results. We're not picky, but your music deserves the best treatment!"
    },
    {
      q: 'How accurate is the emotion detection?',
      a: "The model achieves competitive accuracy on the DEAM dataset across 4 core emotions (neutral, happy, sad, calm). The system employs sophisticated bias handling to ensure accurate emotion detection. It's not perfect, but it's pretty darn good at understanding what makes your heart sing!"
    },
    {
      q: 'What does the aggression slider do?',
      a: 'The aggression slider controls how pronounced the EQ adjustments will be. A higher setting creates more dramatic frequency adjustments, while a lower setting provides subtle corrections. Think of it as the difference between a gentle nudge and a full transformation!'
    },
    {
      q: 'Can I use these EQ settings on any device?',
      a: 'The EQ curves can be exported in multiple formats (Wavelet, Parametric, Graphic) to work with most audio software and hardware equalizers. Whether you\'re using a fancy DAC or just your phone, we\'ve got you covered!'
    },
    {
      q: 'What is valence and arousal?',
      a: 'Valence represents the positivity/negativity of emotions (sad to happy), while arousal represents the energy level (calm to excited). Together, they create a 2D emotional space that captures the complexity of musical emotions. It\'s like a GPS for feelings!'
    },
    {
      q: 'How does the IEM recommendation work?',
      a: "Based on your music's emotional profile, ApollodB suggests IEMs whose frequency response characteristics complement your listening preferences. This is based on acoustic research linking frequency response to emotional perception. Science meets sound, and your ears win!"
    },
    {
      q: 'Is my audio data stored, shared, or used for training?',
      a: 'No. When used on the website, your files are processed transiently to perform the analysis and generate outputs. They are not retained after processing, not shared with third parties, and never used to train models.'
    },
  ];
  const root = qs('#faq-content'); if (!root) return;
  root.innerHTML = '';
  faqs.forEach(item => {
    const el = document.createElement('div');
    el.className = 'faq-item';
    el.innerHTML = `
      <button class="faq-q">${item.q}</button>
      <div class="faq-a" hidden>${item.a}</div>
    `;
    const qbtn = el.querySelector('.faq-q');
    const ans = el.querySelector('.faq-a');
    qbtn.addEventListener('click', () => {
      ans.hidden = !ans.hidden;
    });
    root.appendChild(el);
  });
}

function renderReferences() {
  const refs = [
    { citation: 'Aljanaki, A., Yang, Y.-H., & Soleymani, M. (2017). Developing a benchmark for emotional analysis of music. PLOS ONE, 12(3), e0173392. https://doi.org/10.1371/journal.pone.0173392', description: 'The primary dataset used for training our emotion recognition model.' },
    { citation: "Chen, Y., Ma, Z., Wang, M., & Liu, M. (2024). Advancing music emotion recognition: A transformer encoder-based approach. In Proceedings of the 6th ACM International Conference on Multimedia in Asia (MMAsia '24) (Article 60, pp. 1–5). https://doi.org/10.1145/3696409.3700221", description: 'Modern transformer-based approaches to music emotion recognition.' },
    { citation: 'Ghazarian, S., Wen, N., Galstyan, A., & Peng, N. (2022). DEAM: Dialogue Coherence Evaluation using AMR-based Semantic Manipulations. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL 2022).', description: 'Advanced evaluation methodologies for coherence in dialogue systems.' },
    { citation: 'Kang, J., & Herremans, D. (2025). Towards unified music emotion recognition across dimensional and categorical models. arXiv preprint arXiv:2502.03979v2. https://arxiv.org/abs/2502.03979', description: 'Latest research on unifying different approaches to music emotion recognition.' },
    { citation: 'Soleymani, M., Aljanaki, A., & Yang, Y.-H. (2018). DEAM: MediaEval Database for Emotional Analysis in Music. Swiss Center for Affective Sciences, University of Geneva & Academia Sinica. Presented April 26, 2018.', description: 'Comprehensive overview of the DEAM dataset structure and methodology.' },
    { citation: 'Yang, Y.-H., Aljanaki, A., & Soleymani, M. (2024). Are we there yet? A brief survey of music emotion prediction datasets, models and outstanding challenges. arXiv preprint arXiv:2406.08809. https://arxiv.org/abs/2406.08809', description: 'Current state-of-the-art review of music emotion recognition field and future challenges.' },
    { citation: 'PlusLab NLP. (n.d.). DEAM GitHub Repository. https://github.com/PlusLabNLP/DEAM', description: 'Open source implementation and resources for the DEAM dataset.' },
  ];
  const root = qs('#references-content'); if (!root) return;
  root.innerHTML = '';
  refs.forEach((ref, i) => {
    // Auto-link any URL present in the citation text
    const citationHTML = (ref.citation||'').replace(/(https?:\/\/[^\s)]+)/g, (m)=>`<a href="${m}" target="_blank" rel="noopener">${m}</a>`);
    const card = document.createElement('div');
    card.className = 'ref-card glass';
    card.innerHTML = `
      <div class="ref-index">${i+1}</div>
      <div class="ref-body">
        <div class="ref-citation">${citationHTML}</div>
        <div class="ref-desc">${ref.description}</div>
      </div>
    `;
    root.appendChild(card);
  });
}

function renderAbout() {
  const cfg = state.config?.app_config || {};
  const t = qs('#about-title'); if (t) t.textContent = cfg.title || 'ApollodB';
  const s = qs('#about-subtitle'); if (s) s.textContent = cfg.subtitle || 'AI-Powered Music Emotion Analysis & EQ Optimization';
  const a = qs('#about-author'); if (a) a.textContent = cfg.author || '';
  const l = qs('#about-location'); if (l) l.textContent = cfg.location || '';
  const v = qs('#about-version'); if (v) v.textContent = cfg.version || '';
}

document.addEventListener('DOMContentLoaded', () => {
  wireUI();
  initTabs();
  renderStaticContent();
});
