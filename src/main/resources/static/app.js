// ════════════════════════════════════════════════════════════
//  Your OWN AI — Premium Client Logic Engine (PCA, Vis.js Network, RAG)
// ════════════════════════════════════════════════════════════

const API = 'http://localhost:8080';
const DIMS = 16;

// Cluster color mappings
const COL = { 
  cs: '#00f3ff', 
  math: '#00ff88', 
  food: '#ff9f00', 
  sports: '#d600ff', 
  doc: '#ff0077', 
  default: '#a0a0a0' 
};

const DIM_COL = [
  '#00f3ff','#00f3ff','#00f3ff','#00f3ff',
  '#00ff88','#00ff88','#00ff88','#00ff88',
  '#ff9f00','#ff9f00','#ff9f00','#ff9f00',
  '#d600ff','#d600ff','#d600ff','#d600ff'
];

// Keywords for mock 16D semantic vectors mapping (for demo vectors)
const KW = {
  cs:     ['algorithm','data','tree','graph','array','linked','hash','stack','queue','sort','binary','dynamic','programming','recursion','complexity','pointer','node','search','insert','bfs','dfs','heap','trie'],
  math:   ['calculus','matrix','probability','theorem','integral','derivative','linear','algebra','equation','function','prime','modular','combinatorics','permutation','eigenvalue','statistics','proof'],
  food:   ['food','pizza','sushi','ramen','pasta','recipe','cook','eat','restaurant','dish','ingredient','flavor','spice','noodle','bread','croissant','taco','fish','rice','soup'],
  sports: ['sport','basketball','football','tennis','chess','swim','game','play','score','team','athlete','competition','match','tournament','olympic','dribble','tackle','serve']
};

// ── STATE MANAGEMENT ──
let allItems = [];
let pcaPoints = [];
let hitIds = new Set();
let queryPt = null;
let hoverItem = null;
let pulse = 0;
let selAlgo = 'hnsw';
let searchResults = [];

let activeVisualMode = 'pca'; // 'pca' or 'hnsw'
let selectedHnswLayer = 0;
let hnswInfoData = null;
let visNetworkInstance = null;

// Clemens Wenger 'Vectorial Resonance' States
let activeProjection = 'pca'; // 'pca', 'tsne', or 'umap'
let soundEnabled = false;
let audioCtx = null;
let particles = [];
let sparklineWaves = {
  cosine: new Array(55).fill(0.5),
  euclidean: new Array(55).fill(0.5),
  manhattan: new Array(55).fill(0.5)
};

// ── PCA POWER ITERATION DIMENSIONALITY REDUCTION ──
function pca2D(embs) {
  const n = embs.length, d = embs[0].length;
  if (n < 2) return embs.map(() => [0, 0]);
  
  const mean = new Array(d).fill(0);
  for (const e of embs) {
    for (let i = 0; i < d; i++) mean[i] += e[i] / n;
  }
  
  const X = embs.map(e => e.map((v, i) => v - mean[i]));
  
  function powerIter(X, excl) {
    let v = new Array(d).fill(0).map(() => Math.random() - 0.5);
    if (excl) {
      let dot = v.reduce((s, vi, i) => s + vi * excl[i], 0);
      v = v.map((vi, i) => vi - dot * excl[i]);
    }
    let nrm = Math.sqrt(v.reduce((s, vi) => s + vi * vi, 0));
    v = v.map(vi => vi / nrm);
    
    for (let it = 0; it < 200; it++) {
      const Xv = X.map(xi => xi.reduce((s, xij, j) => s + xij * v[j], 0));
      const nv = new Array(d).fill(0);
      for (let k = 0; k < n; k++) {
        for (let j = 0; j < d; j++) nv[j] += X[k][j] * Xv[k];
      }
      if (excl) {
        let dot = nv.reduce((s, vi, i) => s + vi * excl[i], 0);
        for (let i = 0; i < d; i++) nv[i] -= dot * excl[i];
      }
      nrm = Math.sqrt(nv.reduce((s, vi) => s + vi * vi, 0));
      if (nrm < 1e-10) break;
      const prev = v.slice();
      v = nv.map(vi => vi / nrm);
      if (v.reduce((s, vi, i) => s + (vi - prev[i]) ** 2, 0) < 1e-12) break;
    }
    return v;
  }
  
  const pc1 = powerIter(X, null);
  const pc2 = powerIter(X, pc1);
  return X.map(x => [
    x.reduce((s, v, i) => s + v * pc1[i], 0),
    x.reduce((s, v, i) => s + v * pc2[i], 0)
  ]);
}

// Map text to a custom 16D categorical profile vector (for visualizer database seed)
function textToEmbedding(text) {
  const t = text.toLowerCase(), ws = t.split(/\s+/);
  const s = { cs: 0, math: 0, food: 0, sports: 0 };
  
  for (const w of ws) {
    for (const [cat, kws] of Object.entries(KW)) {
      for (const kw of kws) {
        if (w.includes(kw) || kw.startsWith(w)) { 
          s[cat] += 0.35; 
          break; 
        }
      }
    }
  }
  const mx = Math.max(...Object.values(s), 0.01);
  const n = v => Math.min(v / mx * 0.88, 0.94);
  const jitter = () => (Math.random() - 0.5) * 0.04;
  
  const emb = new Array(16).fill(0.08);
  const fill = (i, score) => {
    if (score < 0.01) return;
    const b = n(score);
    emb[i] = Math.max(0.05, b + jitter());
    emb[i + 1] = Math.max(0.05, b + jitter());
    emb[i + 2] = Math.max(0.05, b * 0.92 + jitter());
    emb[i + 3] = Math.max(0.05, b * 0.87 + jitter());
  };
  fill(0, s.cs);
  fill(4, s.math);
  fill(8, s.food);
  fill(12, s.sports);
  return emb;
}

// ── PCA SCATTER CANVAS RENDERING & RESONANCE ENGINE ──
const sc = document.getElementById('scatter');
const ctx = sc.getContext('2d');
let bounds = { minX: -1, maxX: 1, minY: -1, maxY: 1 };

function resize() {
  const r = sc.parentElement.getBoundingClientRect();
  sc.width = r.width;
  sc.height = r.height;
}
window.addEventListener('resize', resize);

function w2c(wx, wy) {
  const P = 120; // Expanded padding for full screen layout
  const W = sc.width;
  const H = sc.height;
  const rx = bounds.maxX - bounds.minX || 1;
  const ry = bounds.maxY - bounds.minY || 1;
  return [P + ((wx - bounds.minX) / rx) * (W - 2 * P), H - P - ((wy - bounds.minY) / ry) * (H - 2 * P)];
}

// ── PARTICLE SWARM PHYSICS (Inspired by Physics of Beauty) ──
function initParticles() {
  particles = [];
  const count = 350; // Majestic dust trail
  for (let i = 0; i < count; i++) {
    particles.push({
      angle: Math.random() * Math.PI * 2,
      radius: 12 + Math.random() * 45,
      speed: 0.006 + Math.random() * 0.018,
      size: 0.6 + Math.random() * 1.4,
      nodeIndex: 0,
      opacity: 0.12 + Math.random() * 0.55
    });
  }
}

// ── POLYPHONIC RESONANT SYNTHESIZER (Web Audio API) ──
function toggleResonantSound() {
  soundEnabled = !soundEnabled;
  const btn = document.getElementById('toggleSoundBtn');
  const icon = document.getElementById('soundIcon');
  const text = document.getElementById('soundText');
  if (!btn || !icon || !text) return;
  
  if (soundEnabled) {
    text.textContent = 'Audio: Enabled';
    btn.style.color = 'var(--accent-black)';
    btn.style.background = 'var(--accent-white)';
    btn.style.borderColor = 'var(--accent-white)';
    
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    
    const playIntro = () => {
      playChime(330, 0.25, 0.35);
      setTimeout(() => playChime(440, 0.25, 0.35), 100);
      setTimeout(() => playChime(550, 0.35, 0.35), 200);
    };
    
    if (audioCtx.state === 'suspended') {
      audioCtx.resume().then(playIntro).catch(playIntro);
    } else {
      playIntro();
    }
  } else {
    text.textContent = 'Audio: Disabled';
    btn.style.color = 'var(--text-muted)';
    btn.style.background = 'transparent';
    btn.style.borderColor = 'var(--border)';
  }
}

function playChime(frequency, duration, volume) {
  if (!soundEnabled) return;
  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    
    const playNode = () => {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      
      osc.type = 'sine';
      osc.frequency.setValueAtTime(frequency, audioCtx.currentTime);
      
      const now = audioCtx.currentTime;
      gain.gain.setValueAtTime(0.0001, now);
      gain.gain.linearRampToValueAtTime(volume * 0.4, now + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + duration);
      
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      
      osc.start(now);
      osc.stop(now + duration);
    };
    
    if (audioCtx.state === 'suspended') {
      audioCtx.resume().then(playNode);
    } else {
      playNode();
    }
  } catch (err) {
    console.warn("Audio chime play error: ", err);
  }
}

// ── REAL-TIME SPARKLINE WAVE WRITER ──
function pushSparklineData(cosine, euclidean, manhattan) {
  sparklineWaves.cosine.shift(); sparklineWaves.cosine.push(cosine);
  sparklineWaves.euclidean.shift(); sparklineWaves.euclidean.push(euclidean);
  sparklineWaves.manhattan.shift(); sparklineWaves.manhattan.push(manhattan);
  
  document.getElementById('cosineVal').textContent = cosine.toFixed(4);
  document.getElementById('euclideanVal').textContent = euclidean.toFixed(4);
  document.getElementById('manhattanVal').textContent = manhattan.toFixed(4);
  
  drawSparkline('cosineCvs', sparklineWaves.cosine, '#ffffff');
  drawSparkline('euclideanCvs', sparklineWaves.euclidean, '#cccccc');
  drawSparkline('manhattanCvs', sparklineWaves.manhattan, '#888888');
}

function drawSparkline(cvsId, data, color) {
  const cvs = document.getElementById(cvsId);
  if (!cvs) return;
  const W = cvs.clientWidth, H = cvs.clientHeight;
  cvs.width = W; cvs.height = H;
  const vx = cvs.getContext('2d');
  vx.clearRect(0, 0, W, H);
  
  vx.fillStyle = 'rgba(0,0,0,0.18)';
  vx.fillRect(0, 0, W, H);
  
  vx.strokeStyle = color;
  vx.lineWidth = 1.6;
  vx.beginPath();
  const step = W / (data.length - 1);
  const maxVal = Math.max(...data, 0.001);
  const minVal = Math.min(...data, 0);
  const range = maxVal - minVal || 1;
  
  data.forEach((val, i) => {
    const x = i * step;
    const y = H - 3 - ((val - minVal) / range) * (H - 6);
    if (i === 0) vx.moveTo(x, y);
    else vx.lineTo(x, y);
  });
  vx.stroke();
  
  vx.fillStyle = color + '12';
  vx.lineTo(W, H);
  vx.lineTo(0, H);
  vx.closePath();
  vx.fill();
}

function updateParticleBounds() {} // Placeholder hook for slider

function drawFrame() {
  if (activeVisualMode === 'pca') {
    ctx.clearRect(0, 0, sc.width, sc.height);
    ctx.fillStyle = '#0a0a0a'; // Pitch black background
    ctx.fillRect(0, 0, sc.width, sc.height);
    
    // Draw Grid Coordinates ("Latent Dimensions")
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.02)';
    ctx.lineWidth = 1.0;
    for (let i = 1; i <= 7; i++) {
      const tx = (i / 8) * sc.width;
      const ty = (i / 8) * sc.height;
      ctx.beginPath(); ctx.moveTo(tx, 0); ctx.lineTo(tx, sc.height); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, ty); ctx.lineTo(sc.width, ty); ctx.stroke();
    }
    
    // Axis legends (inspired by Clemens Wenger design layout)
    ctx.fillStyle = '#ffffff';
    ctx.font = '600 11.5px Space Grotesk, sans-serif';
    ctx.fillText('Latent Dimension 1 (PC₁ / ProjX)', sc.width / 2 - 90, sc.height - 24);
    
    ctx.save();
    ctx.translate(28, sc.height / 2 + 80);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Latent Dimension 2 (PC₂ / ProjY)', 0, 0);
    ctx.restore();
    
    ctx.fillStyle = '#ffffff';
    ctx.font = '700 14.5px Space Grotesk, sans-serif';
    ctx.fillText('Vectorial Resonance Projections', 32, 38);
    
    // Render swarming particles orbiting nodes
    if (pcaPoints.length > 0) {
      if (particles.length === 0) initParticles();
      
      const threshold = parseFloat(document.getElementById('thresholdSlider')?.value || 0.35);
      
      particles.forEach((p, idx) => {
        p.nodeIndex = idx % pcaPoints.length;
        const node = pcaPoints[p.nodeIndex];
        const [nx, ny] = w2c(node.x, node.y);
        
        p.angle += p.speed;
        // Particle bounds scaling
        const radius = p.radius * threshold * 1.8;
        const px = nx + Math.cos(p.angle) * radius;
        const py = ny + Math.sin(p.angle) * radius;
        
        const col = COL[node.item.category] || COL.default;
        ctx.fillStyle = col;
        ctx.globalAlpha = p.opacity * (soundEnabled ? 1.0 : 0.7);
        ctx.beginPath();
        ctx.arc(px, py, p.size, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.globalAlpha = 1.0;
    }
    
    // Draw organic curving Bezier link from Query Probe
    if (queryPt && hitIds.size > 0) {
      const [qx, qy] = w2c(queryPt.x, queryPt.y);
      for (const pt of pcaPoints) {
        if (!hitIds.has(pt.item.id)) continue;
        const [px, py] = w2c(pt.x, pt.y);
        
        // Fluid, organic Bezier curves (not straight lines)
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.lineWidth = 1.2;
        ctx.setLineDash([3, 5]);
        ctx.beginPath();
        ctx.moveTo(qx, qy);
        
        // Smooth organic S-curving control points
        const cx1 = qx + (px - qx) * 0.25 + Math.sin(pulse) * 12;
        const cy1 = qy + (py - qy) * 0.75 + Math.cos(pulse) * 12;
        ctx.quadraticCurveTo(cx1, cy1, px, py);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    
    // Draw vector cluster nodes
    for (const pt of pcaPoints) {
      const [cx, cy] = w2c(pt.x, pt.y);
      const col = COL[pt.item.category] || COL.default;
      const isHit = hitIds.has(pt.item.id);
      const r = isHit ? 9 : 5.5;
      
      // Orbit concentric geometric rings with neon glow
      if (isHit) {
        const pr = r + 9 + Math.sin(pulse * 1.8) * 4.5;
        ctx.save();
        ctx.shadowColor = col;
        ctx.shadowBlur = 12;
        ctx.beginPath(); ctx.arc(cx, cy, pr, 0, 2 * Math.PI);
        ctx.strokeStyle = col + '88';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.restore();
      }
      
      // Radiant multi-layered glow
      ctx.save();
      const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 3.5);
      grd.addColorStop(0, col + (isHit ? 'bb' : '66'));
      grd.addColorStop(0.4, col + (isHit ? '55' : '22'));
      grd.addColorStop(1, 'transparent');
      ctx.beginPath(); ctx.arc(cx, cy, r * 3.5, 0, 2 * Math.PI);
      ctx.fillStyle = grd;
      ctx.fill();
      ctx.restore();
      
      // Hyper-glowing Core Dot (True neon shadow blur)
      ctx.save();
      ctx.shadowColor = col;
      ctx.shadowBlur = isHit ? 18 : 10;
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2 * Math.PI);
      ctx.fillStyle = col;
      ctx.fill();
      ctx.restore();
      
      if (hoverItem && hoverItem.id === pt.item.id) {
        ctx.save();
        ctx.shadowColor = col;
        ctx.shadowBlur = 12;
        ctx.beginPath(); ctx.arc(cx, cy, r + 5, 0, 2 * Math.PI);
        ctx.strokeStyle = col;
        ctx.lineWidth = 2.0;
        ctx.stroke();
        ctx.restore();
      }
    }
    
    // Draw query marker
    if (queryPt) {
      const [qx, qy] = w2c(queryPt.x, queryPt.y);
      ctx.save();
      ctx.translate(qx, qy);
      ctx.shadowColor = '#fff';
      ctx.shadowBlur = 18;
      ctx.beginPath();
      for (let i = 0; i < 10; i++) {
        const a = (i * Math.PI / 5) - Math.PI / 2;
        const rr = i % 2 === 0 ? 12 : 5;
        if (i === 0) ctx.moveTo(Math.cos(a) * rr, Math.sin(a) * rr);
        else ctx.lineTo(Math.cos(a) * rr, Math.sin(a) * rr);
      }
      ctx.closePath();
      ctx.fillStyle = '#fff';
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.restore();
      
      ctx.fillStyle = '#ffffff';
      ctx.font = '700 11.5px Space Grotesk, sans-serif';
      ctx.fillText('Query Probe', qx + 18, qy + 4);
    }
    
    if (!pcaPoints.length) {
      ctx.fillStyle = '#ffffff';
      ctx.font = '600 13.5px Space Grotesk, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Connecting to VectorDB Server...', sc.width / 2, sc.height / 2);
      ctx.textAlign = 'left';
    }
  }
  pulse += 0.04;
  requestAnimationFrame(drawFrame);
}

// Hover/move event bindings
sc.addEventListener('mousemove', e => {
  if (activeVisualMode !== 'pca') return;
  const rect = sc.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  
  let oldHover = hoverItem;
  hoverItem = null;
  let best = 24;
  for (const pt of pcaPoints) {
    const [cx, cy] = w2c(pt.x, pt.y);
    const d = Math.hypot(mx - cx, my - cy);
    if (d < best) {
      best = d;
      hoverItem = pt.item;
    }
  }
  
  // Resonant Sound Synthesis trigger on hover transitions
  if (hoverItem && (!oldHover || oldHover.id !== hoverItem.id)) {
    // Map node category score index to frequency chime
    let freq = 330; // default (E4)
    if (hoverItem.category === 'cs') freq = 523.25; // C5
    if (hoverItem.category === 'math') freq = 587.33; // D5
    if (hoverItem.category === 'food') freq = 659.25; // E5
    if (hoverItem.category === 'sports') freq = 783.99; // G5
    if (hoverItem.category === 'doc') freq = 880.00; // A5
    
    playChime(freq, 0.45, 0.35);
  }
  
  const tip = document.getElementById('tip');
  if (hoverItem) {
    const col = COL[hoverItem.category] || COL.default;
    tip.style.display = 'block';
    tip.style.left = (e.clientX + 16) + 'px';
    tip.style.top = (e.clientY - 12) + 'px';
    tip.innerHTML = `<span style="font-weight:700;color:${col}">[${hoverItem.category.toUpperCase()}]</span><br>${hoverItem.metadata}`;
  } else {
    tip.style.display = 'none';
  }
});

sc.addEventListener('mouseleave', () => {
  hoverItem = null;
  document.getElementById('tip').style.display = 'none';
});

// ── INTERACTIVE VIS.JS HNSW GRAPH RENDERER ──
function renderHnswNetwork(data, layer) {
  const container = document.getElementById('hnswNetwork');
  if (!data) return;
  
  // Filter nodes & edges on this specific layer
  const filteredNodes = data.nodes.filter(n => n.maxLyr >= layer);
  const filteredEdges = data.edges.filter(e => e.lyr === layer);
  
  const isL = document.body.classList.contains('light-theme');
  
  // Create vis.js datasets
  const nodes = new vis.DataSet(filteredNodes.map(n => {
    const col = COL[n.category] || COL.default;
    const isHit = hitIds.has(n.id);
    return {
      id: n.id,
      label: `#${n.id}`,
      title: `[${n.category.toUpperCase()}] ${n.metadata} (Max Layer: L${n.maxLyr})`,
      color: {
        background: col,
        border: isHit ? (isL ? '#000000' : '#ffffff') : 'rgba(0,0,0,0.5)',
        highlight: { background: col, border: isL ? '#000000' : '#ffffff' }
      },
      shape: 'dot',
      size: isHit ? 30 : 18,
      font: { color: isL ? '#111111' : '#ececf1', face: 'Space Grotesk', size: 12, vadjust: -22 }
    };
  }));
  
  const edges = new vis.DataSet(filteredEdges.map(e => {
    const isHit = searchResults.some(r => r.id === e.src || r.id === e.dst);
    return {
      id: `edge-${e.src}-${e.dst}-${e.lyr}`,
      from: e.src,
      to: e.dst,
      color: {
        color: isHit ? (isL ? '#000000' : '#ffffff') : (isL ? 'rgba(0, 0, 0, 0.12)' : 'rgba(255, 255, 255, 0.12)'),
        highlight: isL ? '#000000' : '#ffffff'
      },
      width: isHit ? 3.5 : 1.2,
      smooth: { type: 'continuous' }
    };
  }));
  
  const visData = { nodes, edges };
  const options = {
    physics: {
      solver: 'forceAtlas2Based',
      forceAtlas2Based: {
        gravitationalConstant: -48,
        centralGravity: 0.006,
        springLength: 180,
        springConstant: 0.06
      },
      stabilization: { iterations: 120, updateInterval: 25 }
    },
    interaction: {
      hover: true,
      tooltipDelay: 100,
      dragNodes: true,
      zoomView: true
    }
  };
  
  visNetworkInstance = new vis.Network(container, visData, options);
  
  // Attach tooltip details on hover
  visNetworkInstance.on("hoverNode", function (params) {
    const nodeId = params.node;
    const nodeVal = data.nodes.find(n => n.id === nodeId);
    if (nodeVal) {
      const tip = document.getElementById('tip');
      const col = COL[nodeVal.category] || COL.default;
      tip.style.display = 'block';
      tip.innerHTML = `<span style="font-weight:700;color:${col}">[${nodeVal.category.toUpperCase()}] ID: #${nodeVal.id}</span><br>${nodeVal.metadata}<br><span style="color:var(--text-muted)">Max Layer: L${nodeVal.maxLyr}</span>`;
      
      const domRect = container.getBoundingClientRect();
      const pointer = visNetworkInstance.canvasToDOM(visNetworkInstance.getBody().nodes[nodeId]);
      tip.style.left = (domRect.left + pointer.x + 18) + 'px';
      tip.style.top = (domRect.top + pointer.y - 12) + 'px';
    }
  });
  
  visNetworkInstance.on("blurNode", function () {
    document.getElementById('tip').style.display = 'none';
  });
}

// Animate the search traversal path inside vis.js Network
function animateHnswPath(path) {
  if (!visNetworkInstance || !path || path.length === 0) return;
  
  let stepIdx = 0;
  
  function nextStep() {
    if (stepIdx >= path.length || !visNetworkInstance) return;
    
    const step = path[stepIdx];
    const edgeId = `edge-${Math.min(step.fromId, step.toId)}-${Math.max(step.fromId, step.toId)}-${step.layer}`;
    
    // Highlight the transitioning nodes and linking edge
    try {
      if (visNetworkInstance.body.data.edges.get(edgeId)) {
        visNetworkInstance.body.data.edges.update({
          id: edgeId,
          color: { color: '#ffffff' },
          width: 5.0
        });
      }
      
      // Update destination node size and glowing border
      const toNode = visNetworkInstance.body.data.nodes.get(step.toId);
      if (toNode) {
        visNetworkInstance.body.data.nodes.update({
          id: step.toId,
          size: 26,
          color: { border: '#ffffff' }
        });
      }
      
      // Focus on the active destination node with an offset
      visNetworkInstance.focus(step.toId, {
        scale: 1.25,
        animation: { duration: 350, easingFunction: 'easeInOutQuad' }
      });
    } catch (e) {
      console.warn("Vis.js edge/node highlight error: " + e.getMessage());
    }
    
    stepIdx++;
    setTimeout(nextStep, 500); // 500ms intervals between greedy jumps
  }
  
  // Clear any existing selections and highlight path
  setTimeout(nextStep, 400);
}

// ── LOADING DATA & DIAGNOSTICS ──
// ── LOADING DATA & DIAGNOSTICS ──
function switchProjection(proj) {
  activeProjection = proj;
  
  // Highlight active projection button on HUD
  const pcas = ['ProjPca', 'ProjTsne', 'ProjUmap'];
  pcas.forEach(p => {
    const btn = document.getElementById('btn' + p);
    if (btn) {
      if (p.toLowerCase().endsWith(proj.toLowerCase())) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    }
  });
  
  reprojectPoints();
}

function reprojectPoints() {
  if (allItems.length < 2) return;
  const pcaCoords = pca2D(allItems.map(v => v.embedding));
  
  const density = parseFloat(document.getElementById('densitySlider')?.value || 0.6);
  
  pcaPoints = allItems.map((item, i) => {
    let x = pcaCoords[i][0];
    let y = pcaCoords[i][1];
    
    if (activeProjection === 'tsne') {
      // Simulate t-SNE concentric groups
      const r = Math.hypot(x, y);
      const theta = Math.atan2(y, x) + 0.65;
      x = r * Math.cos(theta) * 1.15 + Math.sin(y * 4) * 0.06;
      y = r * Math.sin(theta) * 1.15 + Math.cos(x * 4) * 0.06;
    } else if (activeProjection === 'umap') {
      // Simulate UMAP polar manifold clusters
      const scale = 1.35;
      x = x * scale * density + (Math.sin(x * 3.5) * 0.09);
      y = y * scale * density + (Math.cos(y * 3.5) * 0.09);
    }
    return { x, y, item };
  });
  
  // Recalculate bounds
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const p of pcaPoints) {
    x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x);
    y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y);
  }
  const px = (x1 - x0) * 0.22 || 0.1;
  const py = (y1 - y0) * 0.22 || 0.1;
  bounds = { minX: x0 - px, maxX: x1 + px, minY: y0 - py, maxY: y1 + py };
}

async function loadItems() {
  try {
    const r = await fetch(API + '/items');
    allItems = await r.json();
    reprojectPoints();
    const statsLabel = document.getElementById('statsLabel');
    if (statsLabel) statsLabel.textContent = `${allItems.length} vectors · ${DIMS} dims`;
  } catch (_) {}
}

async function loadHnswStats() {
  try {
    const r = await fetch(API + '/hnsw-info');
    hnswInfoData = await r.json();
    
    const maxN = hnswInfoData.nodesPerLayer[0] || 1;
    document.getElementById('layersBreakdown').innerHTML = hnswInfoData.nodesPerLayer.map((cnt, lyr) => {
      const pct = Math.max((cnt / maxN) * 100, 2);
      const edg = hnswInfoData.edgesPerLayer[lyr] || 0;
      return `
        <div class="hnsw-layer-stat-row">
          <div class="hnsw-layer-name">L${lyr}</div>
          <div class="hnsw-layer-bar-track">
            <div class="hnsw-layer-bar-fill" style="width:${pct}%"></div>
          </div>
          <div class="hnsw-layer-counts">${cnt} nodes · ${edg} edges</div>
        </div>`;
    }).reverse().join('');
    
    // Populate layer pills for HNSW visualizer
    const pillsContainer = document.getElementById('layerPillsContainer');
    pillsContainer.innerHTML = hnswInfoData.nodesPerLayer.map((cnt, lyr) => {
      return `<div class="layer-pill ${lyr === selectedHnswLayer ? 'active' : ''}" onclick="selectHnswLayer(${lyr})">L${lyr}</div>`;
    }).reverse().join('');
    
    if (activeVisualMode === 'hnsw') {
      renderHnswNetwork(hnswInfoData, selectedHnswLayer);
    }
  } catch (_) {}
}

function selectHnswLayer(layer) {
  selectedHnswLayer = layer;
  document.querySelectorAll('.layer-pill').forEach((pill, idx) => {
    // Reverse ordered indexing
    const lyrIdx = hnswInfoData.nodesPerLayer.length - 1 - idx;
    pill.classList.toggle('active', lyrIdx === layer);
  });
  if (hnswInfoData) {
    renderHnswNetwork(hnswInfoData, selectedHnswLayer);
  }
}

// ── SEARCH & BENCHMARKING ──
function setAlgo(el) {
  document.querySelectorAll('.algo-pill-btn').forEach(b => b.classList.remove('active'));
  el.classList.add('active');
  selAlgo = el.dataset.algo;
}

async function runSearch() {
  const text = document.getElementById('qInput').value.trim();
  const clearBtn = document.getElementById('clearSearchBtn');
  if (clearBtn) {
    clearBtn.style.display = text ? 'flex' : 'none';
  }
  if (!text) return;
  
  const emb = textToEmbedding(text);
  const kSlider = document.getElementById('kSlider');
  const k = kSlider ? parseInt(kSlider.value) : 5;
  const metric = document.getElementById('metric').value;
  
  const url = `${API}/search?v=${emb.join(',')}&k=${k}&metric=${metric}&algo=${selAlgo}`;
  try {
    const r = await fetch(url);
    const data = await r.json();
    
    searchResults = data.results || [];
    hitIds = new Set(searchResults.map(r => r.id));
    const us = data.latencyUs || 0;
    
    // Safe DOM updating for legacy elements
    const latBig = document.getElementById('latBig');
    if (latBig) latBig.textContent = us < 1000 ? us + ' μs' : (us / 1000).toFixed(2) + ' ms';
    const latSub = document.getElementById('latSub');
    if (latSub) latSub.textContent = `${selAlgo.toUpperCase()}  ·  ${metric.toUpperCase()}  ·  k=${k}`;
    
    // Resonant Audio synthesis chime strike on successful search
    playChime(660, 0.45, 0.45);
    setTimeout(() => playChime(880, 0.6, 0.45), 150);
    
    // Calculate and push dynamic sparkline waves
    let cosDist = 0.95, eucDist = 0.12, manDist = 0.08;
    if (searchResults.length > 0) {
      const firstHit = searchResults[0];
      if (metric === 'cosine') {
        cosDist = 1 - firstHit.distance;
        eucDist = Math.sqrt(2 * firstHit.distance) * 0.45;
        manDist = eucDist * 1.35;
      } else if (metric === 'euclidean') {
        eucDist = firstHit.distance;
        cosDist = 1 - (eucDist * eucDist) * 0.25;
        manDist = eucDist * 1.25;
      } else {
        manDist = firstHit.distance;
        eucDist = manDist * 0.78;
        cosDist = 1 - (eucDist * eucDist) * 0.25;
      }
    }
    pushSparklineData(Math.max(0.0001, cosDist), Math.max(0.0001, eucDist), Math.max(0.0001, manDist));
    
    if (searchResults.length > 0) {
      let sx = 0, sy = 0, sw = 0;
      for (let i = 0; i < Math.min(3, searchResults.length); i++) {
        const pt = pcaPoints.find(p => p.item.id === searchResults[i].id);
        if (pt) {
          const w = 1 / (i + 1);
          sx += pt.x * w; sy += pt.y * w;
          sw += w;
        }
      }
      if (sw > 0) {
        queryPt = { x: sx / sw + (Math.random() - 0.5) * 0.015, y: sy / sw + (Math.random() - 0.5) * 0.015 };
      }
    }
    
    renderResults(searchResults);
    drawVecChart(emb);
    
    // If in HNSW network visual mode, reload layout and animate greedy path
    if (activeVisualMode === 'hnsw' && data.traversalPath && data.traversalPath.length > 0) {
      await loadHnswStats(); // Update hit sizes first
      animateHnswPath(data.traversalPath);
    }
  } catch (_) {
    alert('Cannot reach Server — Is the Spring Boot backend running on port 8080?');
  }
}

document.getElementById('qInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') runSearch();
});

const qInput = document.getElementById('qInput');
const clearSearchBtn = document.getElementById('clearSearchBtn');
if (qInput && clearSearchBtn) {
  qInput.addEventListener('input', () => {
    clearSearchBtn.style.display = qInput.value.trim() ? 'flex' : 'none';
  });
}

function clearSearch() {
  const qInput = document.getElementById('qInput');
  const clearBtn = document.getElementById('clearSearchBtn');
  if (qInput) {
    qInput.value = '';
    qInput.focus();
  }
  if (clearBtn) {
    clearBtn.style.display = 'none';
  }
  
  hitIds = new Set();
  queryPt = null;
  searchResults = [];
  
  const resultsContainer = document.getElementById('results');
  if (resultsContainer) {
    resultsContainer.innerHTML = '<div class="empty-placeholder">Search probe vectors to visualize nearest matching neighbors...</div>';
  }
  
  pushSparklineData(0.0001, 0.0001, 0.0001);
  
  if (activeVisualMode === 'hnsw' && hnswInfoData) {
    renderHnswNetwork(hnswInfoData, selectedHnswLayer);
  }
}

function renderResults(results) {
  const container = document.getElementById('results');
  if (!results || !results.length) {
    container.innerHTML = '<div class="empty-placeholder">No matching nearest neighbors found.</div>';
    return;
  }
  
  container.innerHTML = results.map((r, i) => {
    const col = COL[r.category] || COL.default;
    return `
      <div class="rcard" onmouseenter="hoverItem={id:${r.id}}" onmouseleave="hoverItem=null">
        <div class="rrank">RANK #${i + 1} NEIGHBOR</div>
        <div class="rmeta">${r.metadata}</div>
        <div class="rfoot">
          <span class="rcat cat-${r.category}">${r.category.toUpperCase()}</span>
          <span class="rdist">dist: ${r.distance.toFixed(6)}</span>
          <button class="btn-del" onclick="deleteItem(${r.id})">✕</button>
        </div>
      </div>`;
  }).join('');
}

function drawVecChart(emb) {
  const vc = document.getElementById('vecCvs');
  if (!vc) return;
  const W = vc.parentElement.clientWidth;
  vc.width = W;
  const vx = vc.getContext('2d');
  
  vx.clearRect(0, 0, W, 70);
  vx.fillStyle = '#0a0a0a';
  vx.fillRect(0, 0, W, 70);
  
  const bw = (W - 4) / DIMS;
  for (let i = 0; i < DIMS; i++) {
    const h = emb[i] * 50;
    const x = 2 + i * bw;
    const col = DIM_COL[i];
    vx.shadowColor = col;
    vx.shadowBlur = 0;
    vx.fillStyle = col + 'aa';
    vx.fillRect(x + 1, 55 - h, bw - 2, h);
  }
  
  vx.shadowBlur = 0;
  vx.font = '500 8px monospace';
  vx.textAlign = 'center';
  
  [['CS', 0], ['MATH', 4], ['FOOD', 8], ['SPORT', 12]].forEach(([lbl, gi], i) => {
    vx.fillStyle = Object.values(COL)[i] + 'aa';
    vx.fillText(lbl, 2 + (gi + 1.5) * bw, 65);
  });
  vx.textAlign = 'left';
}

async function runBenchmark() {
  const text = document.getElementById('qInput').value.trim() || 'binary tree search';
  const emb = textToEmbedding(text);
  const metric = document.getElementById('metric').value;
  
  try {
    const r = await fetch(`${API}/benchmark?v=${emb.join(',')}&k=5&metric=${metric}`);
    const d = await r.json();
    
    document.getElementById('benchSec').style.display = 'block';
    const mx = Math.max(d.bruteforceUs, d.kdtreeUs, d.hnswUs, 1);
    
    document.getElementById('benchBars').innerHTML = [
      { lbl: 'Brute Force Search', us: d.bruteforceUs, col: 'var(--red)' },
      { lbl: 'KD-Tree Index', us: d.kdtreeUs, col: 'var(--cs)' },
      { lbl: 'HNSW Graph (O(log N))', us: d.hnswUs, col: 'var(--accent)' },
    ].map(({ lbl, us, col }) => {
      const pct = Math.max((us / mx) * 100, 2);
      const disp = us < 1000 ? us + ' μs' : (us / 1000).toFixed(2) + ' ms';
      return `
        <div class="bench-bar-group">
          <div class="bench-bar-labels">
            <span class="bench-bar-lbl" style="color:${col}">${lbl}</span>
            <span class="bench-bar-val">${disp}</span>
          </div>
          <div class="bench-bar-track">
            <div class="bench-bar-fill" style="width:${pct}%;background:${col}"></div>
          </div>
        </div>`;
    }).join('');
  } catch (_) {}
}

async function addVector() {
  const meta = document.getElementById('addMeta').value.trim();
  const cat = document.getElementById('addCat').value;
  if (!meta) return;
  
  const emb = textToEmbedding(meta + ' ' + cat);
  try {
    await fetch(API + '/insert', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ metadata: meta, category: cat, embedding: emb })
    });
    document.getElementById('addMeta').value = '';
    await loadItems();
    loadHnswStats();
  } catch (_) {}
}

async function deleteItem(id) {
  try {
    await fetch(`${API}/delete/${id}`, { method: 'DELETE' });
    searchResults = searchResults.filter(r => r.id !== id);
    hitIds.delete(id);
    renderResults(searchResults);
    await loadItems();
    loadHnswStats();
  } catch (_) {}
}

// ── DOCUMENT INGESTION & DRAG AND DROP ──
async function checkOllamaStatus() {
  try {
    const r = await fetch(API + '/status');
    const d = await r.json();
    const badge = document.getElementById('ollamaBadge');
    const card = document.getElementById('ollamaStatusCard');
    
    if (d.ollamaAvailable) {
      if (badge) {
        badge.className = 'status-badge status-online';
        badge.textContent = 'Ollama: Online';
      }
      if (card) {
        card.className = 'ollama-status-card status-online';
        card.querySelector('.status-details').innerHTML = `
          ● Local AI Active<br>
          Embed Model: <span style="color:var(--cs)">${d.embedModel}</span><br>
          Llama Model: <span style="color:var(--math)">${d.genModel}</span><br>
          Embedded Vectors: <span style="color:var(--text)">${d.docCount} items</span><br>
          Embedding Dims: <span style="color:var(--gold)">${d.docDims || 'Dynamic'}</span>`;
      }
    } else {
      if (badge) {
        badge.className = 'status-badge status-offline';
        badge.textContent = 'Ollama: Offline';
      }
      if (card) {
        card.className = 'ollama-status-card status-offline';
        card.querySelector('.status-details').innerHTML = `
          ● Local AI Offline<br><br>
          To activate full RAG capabilities:<br>
          1. Download from <u>ollama.com</u><br>
          2. Run: <span style="color:var(--red)">ollama pull nomic-embed-text</span><br>
          3. Run: <span style="color:var(--red)">ollama pull llama3.2</span>`;
      }
    }
  } catch (_) {}
}

// Drag & Drop event bindings
const dropZone = document.getElementById('fileDropZone');
const fileInput = document.getElementById('fileInput');

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

['dragleave', 'dragend'].forEach(evt => {
  dropZone.addEventListener(evt, () => dropZone.classList.remove('dragover'));
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    handleInboundFile(files[0]);
  }
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files.length > 0) {
    handleInboundFile(e.target.files[0]);
  }
});

function handleInboundFile(file) {
  const status = document.getElementById('insertStatus');
  if (!file.name.endsWith('.txt') && !file.name.endsWith('.md')) {
    status.innerHTML = '<span style="color:var(--red)">✗ Format unsupported. Only .txt or .md files.</span>';
    return;
  }
  
  const reader = new FileReader();
  reader.onload = function(e) {
    const textContent = e.target.result;
    document.getElementById('docTitle').value = file.name.replace(/\.[^/.]+$/, ""); // Strip extension
    document.getElementById('docText').value = textContent;
    status.innerHTML = `<span style="color:var(--green)">✓ Loaded ${file.name} successfully! Click Embed to index.</span>`;
  };
  reader.readAsText(file);
}

async function insertDocument() {
  const title = document.getElementById('docTitle').value.trim();
  const text = document.getElementById('docText').value.trim();
  const btn = document.getElementById('insertDocBtn');
  const status = document.getElementById('insertStatus');
  
  if (!title || !text) {
    status.textContent = '⚠ Subject title and body content required.';
    return;
  }
  
  btn.disabled = true;
  btn.textContent = 'Vectorizing & Segmenting...';
  status.innerHTML = '<span style="color:var(--text-muted)">Generating vector coordinates via Ollama embeddings...</span>';
  
  try {
    const r = await fetch(API + '/doc/insert', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, text })
    });
    const d = await r.json();
    
    if (d.error) {
      status.innerHTML = `<span style="color:var(--red)">✗ Ingestion failed: ${d.error}</span>`;
    } else {
      status.innerHTML = `<span style="color:var(--green)">✓ Segmented into ${d.chunks} chunk(s) · Vector: ${d.dims}-D!</span>`;
      document.getElementById('docTitle').value = '';
      document.getElementById('docText').value = '';
      
      // Seed a virtual 16D vector to map onto visual semantic canvas
      const emb16 = textToEmbedding(title + ' ' + text);
      await fetch(API + '/insert', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ metadata: title, category: 'doc', embedding: emb16 })
      });
      
      await loadItems();
      loadHnswStats();
      loadDocList();
      checkOllamaStatus();
    }
  } catch (_) {
    status.innerHTML = '<span style="color:var(--red)">✗ Network / Ingestion failure</span>';
  }
  btn.disabled = false;
  btn.textContent = '⚡ Embed & Index Chunks';
}

async function loadDocList() {
  try {
    const r = await fetch(API + '/doc/list');
    const docs = await r.json();
    document.getElementById('docCountLabel').textContent = docs.length;
    
    const container = document.getElementById('docList');
    if (!docs.length) {
      container.innerHTML = '<div class="empty-placeholder">No document segments indexed.</div>';
      return;
    }
    
    container.innerHTML = docs.map(d => `
      <div class="dcard">
        <div class="dcard-title">${d.title}</div>
        <div class="dcard-preview">${d.preview}</div>
        <div class="dcard-foot">
          <span class="dcard-words">${d.words} words</span>
          <button class="btn-del" onclick="deleteDoc(${d.id})">✕</button>
        </div>
      </div>`).join('');
  } catch (_) {}
}

async function deleteDoc(id) {
  try {
    await fetch(`${API}/doc/delete/${id}`, { method: 'DELETE' });
    loadDocList();
    checkOllamaStatus();
  } catch (_) {}
}

// ── RAG ASK AI (CHATBOT INTERACTIVE TIMELINE) ──
document.getElementById('ragQuestion').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault(); // Prevent standard newline
    askAI();
  }
});

async function askAI() {
  const question = document.getElementById('ragQuestion').value.trim();
  if (!question) return;
  
  const k = parseInt(document.getElementById('ragK').value);
  const btn = document.getElementById('askBtn');
  btn.disabled = true;
  btn.style.opacity = '0.5';
  
  const history = document.getElementById('chatHistory');
  history.innerHTML = ''; // Reset chat log
  
  // Append question bubble
  const qDiv = document.createElement('div');
  qDiv.className = 'chat-q';
  qDiv.innerHTML = `
    <div class="chat-message-inner">
      <div class="chat-avatar">U</div>
      <div class="chat-msg-body">
        <div class="chat-msg-label">You</div>
        <div class="chat-msg-text"></div>
      </div>
    </div>`;
  qDiv.querySelector('.chat-msg-text').textContent = question;
  history.appendChild(qDiv);
  
  // Thinking telemetry indicator
  const thinkDiv = document.createElement('div');
  thinkDiv.className = 'thinking';
  thinkDiv.innerHTML = '<div class="spinner"></div>Segmenting query &amp; retrieving context chunks…';
  history.appendChild(thinkDiv);
  history.scrollTop = history.scrollHeight;
  
  // Highlight PCA visualizer in background
  fetch(API + '/doc/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, k })
  })
  .then(res => res.json())
  .then(data => {
    if (data.contexts && data.contexts.length > 0) {
      hitIds = new Set();
      let sx = 0, sy = 0, sw = 0;
      data.contexts.forEach((ctx, i) => {
        const pt = pcaPoints.find(p => p.item.category === 'doc' && ctx.title.startsWith(p.item.metadata));
        if (pt) {
          hitIds.add(pt.item.id);
          const w = 1 / (i + 1);
          sx += pt.x * w; sy += pt.y * w;
          sw += w;
        }
      });
      if (sw > 0) {
        queryPt = { x: sx / sw + (Math.random() - 0.5) * 0.015, y: sy / sw + (Math.random() - 0.5) * 0.015 };
      }
    } else {
      // Fallback star projection on query keywords 16-D
      hitIds = new Set();
      const emb16 = textToEmbedding(question);
      fetch(`${API}/search?v=${emb16.join(',')}&k=3&metric=cosine&algo=hnsw`)
      .then(res2 => res2.json())
      .then(data2 => {
        if (data2.results && data2.results.length > 0) {
          let sx = 0, sy = 0, sw = 0;
          for (let i = 0; i < Math.min(3, data2.results.length); i++) {
            const pt = pcaPoints.find(p => p.item.id === data2.results[i].id);
            if (pt) {
              const w = 1 / (i + 1);
              sx += pt.x * w; sy += pt.y * w;
              sw += w;
            }
          }
          if (sw > 0) {
            queryPt = { x: sx / sw + (Math.random() - 0.5) * 0.015, y: sy / sw + (Math.random() - 0.5) * 0.015 };
          }
        }
      });
    }
  }).catch(() => {});
  
  try {
    const r = await fetch(API + '/doc/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, k })
    });
    const d = await r.json();
    thinkDiv.remove();
    
    const aDiv = document.createElement('div');
    aDiv.className = 'chat-a';
    
    if (d.error) {
      const isOllamaOffline = d.error.toLowerCase().includes("ollama");
      const errMsg = isOllamaOffline 
        ? `Ollama server is currently offline.\n\nTo enable full local AI capabilities:\n1. Download and install Ollama from **https://ollama.com**\n2. Open your terminal and run:\n   \`ollama pull nomic-embed-text\`\n   \`ollama pull llama3.2\`\n3. Launch the Ollama application and refresh this page!`
        : d.error;
      
      aDiv.innerHTML = `
        <div class="chat-message-inner">
          <div class="chat-avatar">
            <img src="robot_icon.png" class="chat-avatar-img" alt="AI" />
          </div>
          <div class="chat-msg-body">
            <div class="chat-msg-label">System</div>
            <div class="chat-msg-text" style="color:var(--red); white-space:pre-wrap;">${errMsg}</div>
          </div>
        </div>`;
      history.appendChild(aDiv);
    } else {
      aDiv.innerHTML = `
        <div class="chat-message-inner">
          <div class="chat-avatar">
            <img src="robot_icon.png" class="chat-avatar-img" alt="AI" />
          </div>
          <div class="chat-msg-body">
            <div class="chat-msg-label">${d.model || 'Local Model'}</div>
            <div class="chat-msg-text" id="typeTarget"></div>
            <div class="chat-ctx">
              <div class="chat-ctx-label">Retrieved Context References (${d.contexts.length} chunks)</div>
              ${d.contexts.map((c, i) => `
                <span class="ctx-chip" onclick="toggleCtx(${i})">#${i + 1} ${c.title} (dist: ${c.distance.toFixed(3)})</span>
                <div class="ctx-expand" id="ctx-${i}">${c.text}</div>`).join('')}
            </div>
          </div>
        </div>`;
      history.appendChild(aDiv);
      
      // RAG typewriter response stream
      const target = aDiv.querySelector('#typeTarget');
      target.classList.add('typing');
      const fullText = d.answer;
      let textIdx = 0;
      const timer = setInterval(() => {
        if (textIdx >= fullText.length) {
          clearInterval(timer);
          target.classList.remove('typing');
          return;
        }
        const chunk = fullText.slice(textIdx, textIdx + 3);
        target.textContent += chunk;
        textIdx += 3;
        history.scrollTop = history.scrollHeight;
      }, 18);
    }
  } catch (e) {
    thinkDiv.remove();
    const err = document.createElement('div');
    err.className = 'chat-a';
    err.innerHTML = `
      <div class="chat-message-inner">
        <div class="chat-avatar">
          <img src="robot_icon.png" class="chat-avatar-img" alt="AI" />
        </div>
        <div class="chat-msg-body">
          <div class="chat-msg-label">System</div>
          <div class="chat-msg-text" style="color:var(--red)">
            Unable to reach the Spring Boot backend. Please verify your server is running on port 8080!
          </div>
        </div>
      </div>`;
    history.appendChild(err);
  }
  
  document.getElementById('ragQuestion').value = '';
  btn.disabled = false;
  btn.style.opacity = '1';
  history.scrollTop = history.scrollHeight;
}

function toggleCtx(i) {
  const el = document.getElementById('ctx-' + i);
  el.style.display = el.style.display === 'block' ? 'none' : 'block';
}

// ── NAVIGATION & BOOT ──
function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    const isCurrent = btn.id === 'tabBtn-' + name;
    btn.classList.toggle('active', isCurrent);
  });
  
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  
  if (name === 'docs') loadDocList();
  if (name === 'visualizer') {
    setTimeout(() => {
      resize();
      if (activeVisualMode === 'hnsw' && visNetworkInstance) {
        visNetworkInstance.fit();
      }
    }, 50); // Small timeout to allow DOM tab-active displays to register size bounds
  }
}

function switchVisualMode(mode) {
  activeVisualMode = mode;
  
  // Highlight active vertical toolbar icons
  const pcaBtn = document.getElementById('btnToolbarPca');
  const hnswBtn = document.getElementById('btnToolbarHnsw');
  if (pcaBtn) pcaBtn.classList.toggle('active', mode === 'pca');
  if (hnswBtn) hnswBtn.classList.toggle('active', mode === 'hnsw');
  
  document.getElementById('pcaWrapper').style.display = mode === 'pca' ? 'block' : 'none';
  document.getElementById('hnswNetworkWrapper').style.display = mode === 'hnsw' ? 'block' : 'none';
  document.getElementById('hnswLayerSelector').style.display = mode === 'hnsw' ? 'flex' : 'none';
  
  if (mode === 'pca') {
    setTimeout(resize, 40);
  }
  if (mode === 'hnsw' && hnswInfoData) {
    renderHnswNetwork(hnswInfoData, selectedHnswLayer);
  }
}

// ── DRAWER INTERACTION TOGGLES ──
function toggleDocsDrawer() {
  const drawer = document.getElementById('docsDrawer');
  if (!drawer) return;
  drawer.classList.toggle('open');
  const btn = document.getElementById('btnToolbarDocs');
  if (btn) btn.classList.toggle('active', drawer.classList.contains('open'));
  
  // Close settings drawer if open
  const setDrawer = document.getElementById('settingsDrawer');
  if (setDrawer && setDrawer.classList.contains('open') && drawer.classList.contains('open')) {
    toggleSettingsDrawer();
  }
}

function toggleSettingsDrawer() {
  const drawer = document.getElementById('settingsDrawer');
  if (!drawer) return;
  drawer.classList.toggle('open');
  const btn = document.getElementById('btnToolbarSettings');
  if (btn) btn.classList.toggle('active', drawer.classList.contains('open'));
  
  // Close docs drawer if open
  const docDrawer = document.getElementById('docsDrawer');
  if (docDrawer && docDrawer.classList.contains('open') && drawer.classList.contains('open')) {
    toggleDocsDrawer();
  }
}

// Textarea auto expansion
function autoExpandTextarea(el) {
  el.style.height = 'auto';
  el.style.height = el.scrollHeight + 'px';
}

// Metric change event
function onMetricChange() {
  if (document.getElementById('qInput').value.trim()) {
    runSearch();
  }
}

// Monochromatic Theme Toggle
function toggleTheme() {
  const isLight = document.body.classList.toggle('light-theme');
  localStorage.setItem('theme', isLight ? 'light' : 'dark');
  
  if (isLight) {
    COL.cs = '#0070f3';      // Vibrant Blue
    COL.math = '#00a86b';    // Vibrant Green
    COL.food = '#e67e22';    // Vibrant Orange
    COL.sports = '#8e44ad';  // Vibrant Purple
    COL.doc = '#e91e63';     // Vibrant Pink
    COL.default = '#7f8c8d';
    
    // Update DIM_COL for Light theme
    DIM_COL[0] = DIM_COL[1] = DIM_COL[2] = DIM_COL[3] = '#0070f3';
    DIM_COL[4] = DIM_COL[5] = DIM_COL[6] = DIM_COL[7] = '#00a86b';
    DIM_COL[8] = DIM_COL[9] = DIM_COL[10] = DIM_COL[11] = '#e67e22';
    DIM_COL[12] = DIM_COL[13] = DIM_COL[14] = DIM_COL[15] = '#8e44ad';
  } else {
    COL.cs = '#00f3ff';      // Electric Cyan
    COL.math = '#00ff88';    // Neon Green
    COL.food = '#ff9f00';    // Neon Orange
    COL.sports = '#d600ff';  // Neon Magenta
    COL.doc = '#ff0077';     // Neon Coral/Rose
    COL.default = '#a0a0a0';
    
    // Update DIM_COL for Dark theme
    DIM_COL[0] = DIM_COL[1] = DIM_COL[2] = DIM_COL[3] = '#00f3ff';
    DIM_COL[4] = DIM_COL[5] = DIM_COL[6] = DIM_COL[7] = '#00ff88';
    DIM_COL[8] = DIM_COL[9] = DIM_COL[10] = DIM_COL[11] = '#ff9f00';
    DIM_COL[12] = DIM_COL[13] = DIM_COL[14] = DIM_COL[15] = '#d600ff';
  }
  
  // Re-render HNSW graph if in HNSW mode
  if (activeVisualMode === 'hnsw' && hnswInfoData) {
    renderHnswNetwork(hnswInfoData, selectedHnswLayer);
  }
}

// Bootstrapping
const savedTheme = localStorage.getItem('theme');
if (savedTheme === 'light') {
  document.body.classList.add('light-theme');
  COL.cs = '#0070f3';
  COL.math = '#00a86b';
  COL.food = '#e67e22';
  COL.sports = '#8e44ad';
  COL.doc = '#e91e63';
  COL.default = '#7f8c8d';
  
  DIM_COL[0] = DIM_COL[1] = DIM_COL[2] = DIM_COL[3] = '#0070f3';
  DIM_COL[4] = DIM_COL[5] = DIM_COL[6] = DIM_COL[7] = '#00a86b';
  DIM_COL[8] = DIM_COL[9] = DIM_COL[10] = DIM_COL[11] = '#e67e22';
  DIM_COL[12] = DIM_COL[13] = DIM_COL[14] = DIM_COL[15] = '#8e44ad';
} else {
  COL.cs = '#00f3ff';
  COL.math = '#00ff88';
  COL.food = '#ff9f00';
  COL.sports = '#d600ff';
  COL.doc = '#ff0077';
  COL.default = '#a0a0a0';
  
  DIM_COL[0] = DIM_COL[1] = DIM_COL[2] = DIM_COL[3] = '#00f3ff';
  DIM_COL[4] = DIM_COL[5] = DIM_COL[6] = DIM_COL[7] = '#00ff88';
  DIM_COL[8] = DIM_COL[9] = DIM_COL[10] = DIM_COL[11] = '#ff9f00';
  DIM_COL[12] = DIM_COL[13] = DIM_COL[14] = DIM_COL[15] = '#d600ff';
}

resize();
drawFrame();
loadItems().then(loadHnswStats);
checkOllamaStatus();
// Polling to keep track of Ollama status
setInterval(checkOllamaStatus, 8000);
