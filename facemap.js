// facemap.js: MediaPipe face landmarks + a personal "unwrapped face" map that front/left/right photos are warped onto.
"use strict";
const FACEMAP = (() => {
  const SOURCES = [
    { mod: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs", wasm: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm" },
    { mod: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3", wasm: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm" },
  ];
  const MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
  // landmark index groups (MediaPipe face mesh topology)
  const G = {
    lips: [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185],
    leftEye: [263,249,390,373,374,380,381,382,362,466,388,387,386,385,384,398],
    rightEye: [33,7,163,144,145,153,154,155,133,246,161,160,159,158,157,173],
    leftBrow: [276,283,282,295,285,300,293,334,296,336],
    rightBrow: [46,53,52,65,55,70,63,105,66,107],
    nostrils: [64,98,97,2,326,327,294,278,344,440,275,4,45,220,115,48],
    oval: [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109],
  };
  let lm = null, loading = null;

  async function landmarker() {
    if (lm) return lm;
    if (loading) return loading;
    loading = (async () => {
      let lastErr;
      for (const s of SOURCES) {
        try {
          const mod = await import(s.mod);
          const V = mod.FaceLandmarker ? mod : (mod.default || mod);
          const fileset = await V.FilesetResolver.forVisionTasks(s.wasm);
          lm = await V.FaceLandmarker.createFromOptions(fileset, { baseOptions: { modelAssetPath: MODEL, delegate: "CPU" }, runningMode: "IMAGE", numFaces: 1 });
          return lm;
        } catch (e) { lastErr = e; }
      }
      loading = null;
      throw lastErr || new Error("Couldn't load the face model");
    })();
    return loading;
  }

  // Returns a compact flat array [x0,y0,z0,x1,...] (normalized coords ×1e4, ints) or null if no face.
  async function detect(blob) {
    const L = await landmarker();
    const bmp = await createImageBitmap(blob);
    try {
      const r = L.detect(bmp);
      const f = r?.faceLandmarks?.[0];
      if (!f || f.length < 468) return null;
      const out = new Array(f.length * 3);
      f.forEach((p, i) => { out[i * 3] = Math.round(p.x * 1e4); out[i * 3 + 1] = Math.round(p.y * 1e4); out[i * 3 + 2] = Math.round(p.z * 1e4); });
      return out;
    } finally { bmp.close && bmp.close(); }
  }
  const pt = (flat, i, w, h) => [flat[i * 3] / 1e4 * w, flat[i * 3 + 1] / 1e4 * h, flat[i * 3 + 2] / 1e4 * w];

  // ---------- Delaunay (Bowyer–Watson), fine for ~470 points ----------
  function delaunay(P) {
    const n = P.length; let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const [x, y] of P) { minX = Math.min(minX, x); minY = Math.min(minY, y); maxX = Math.max(maxX, x); maxY = Math.max(maxY, y); }
    const d = Math.max(maxX - minX, maxY - minY) * 20, mx = (minX + maxX) / 2, my = (minY + maxY) / 2;
    const pts = P.concat([[mx - d, my - d], [mx, my + d], [mx + d, my - d]]);
    const circ = (a, b, c) => {
      const [ax, ay] = pts[a], [bx, by] = pts[b], [cx, cy] = pts[c], D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
      if (Math.abs(D) < 1e-12) return { x: 0, y: 0, r2: Infinity };
      const ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / D;
      const uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / D;
      return { x: ux, y: uy, r2: (ax - ux) ** 2 + (ay - uy) ** 2 };
    };
    let tris = [{ v: [n, n + 1, n + 2], c: circ(n, n + 1, n + 2) }];
    for (let i = 0; i < n; i++) {
      const [px, py] = pts[i], bad = [], keep = [];
      for (const t of tris) ((px - t.c.x) ** 2 + (py - t.c.y) ** 2 < t.c.r2 ? bad : keep).push(t);
      const edges = new Map();
      for (const t of bad) for (const [a, b] of [[t.v[0], t.v[1]], [t.v[1], t.v[2]], [t.v[2], t.v[0]]]) {
        const k = a < b ? a + "," + b : b + "," + a; edges.set(k, edges.has(k) ? null : [a, b]);
      }
      tris = keep;
      for (const e of edges.values()) if (e) tris.push({ v: [e[0], e[1], i], c: circ(e[0], e[1], i) });
    }
    return tris.filter(t => t.v.every(k => k < n)).map(t => t.v);
  }
  function hull(points) {
    const p = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lo = [], up = [];
    for (const q of p) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
    for (const q of p.reverse()) { while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
    return lo.slice(0, -1).concat(up.slice(0, -1));
  }

  // Build a personal layout from one frontal photo's landmarks: cylindrical unwrap so the sides of the face get room.
  function buildLayout(flat, W, H, target = 1000) {
    const N = 468, P = [];
    for (let i = 0; i < N; i++) P.push(pt(flat, i, W, H));
    const nose = P[1], cx = nose[0];
    const R = Math.hypot(P[454][0] - P[234][0], P[454][1] - P[234][1]) / 2 * 1.08;
    let uv = P.map(([x, y]) => [cx + R * Math.asin(Math.max(-.985, Math.min(.985, (x - cx) / R))), y]);
    let minU = Infinity, minV = Infinity, maxU = -Infinity, maxV = -Infinity;
    for (const [u, v] of uv) { minU = Math.min(minU, u); minV = Math.min(minV, v); maxU = Math.max(maxU, u); maxV = Math.max(maxV, v); }
    const s = target / (maxU - minU), m = 16;
    uv = uv.map(([u, v]) => [(u - minU) * s + m, (v - minV) * s + m]);
    const MW = Math.round((maxU - minU) * s + 2 * m), MH = Math.round((maxV - minV) * s + 2 * m);
    let tris = delaunay(uv);
    // drop long skinny triangles Delaunay adds along the concave outline
    const el = []; for (const [a, b, c] of tris) for (const [p, q] of [[a, b], [b, c], [c, a]]) el.push(Math.hypot(uv[p][0] - uv[q][0], uv[p][1] - uv[q][1]));
    const sorted = el.slice().sort((a, b) => a - b), lim = sorted[Math.floor(sorted.length * .5)] * 4;
    tris = tris.filter(([a, b, c]) => [[a, b], [b, c], [c, a]].every(([p, q]) => Math.hypot(uv[p][0] - uv[q][0], uv[p][1] - uv[q][1]) < lim));
    const groups = {}; for (const [k, idx] of Object.entries(G)) groups[k] = hull(idx.map(i => uv[i]));
    return { v: 1, uv, tris, W: MW, H: MH, groups, created: Date.now() };
  }

  // Exclusion mask (eyes, brows, lips, nostrils) in map space
  function maskCanvas(layout) {
    const c = IMG.canvasOf(layout.W, layout.H), x = c.getContext("2d");
    x.fillStyle = "#fff"; x.strokeStyle = "#fff"; x.lineJoin = "round"; x.lineWidth = layout.W * .03;
    for (const k of ["lips", "leftEye", "rightEye", "leftBrow", "rightBrow", "nostrils"]) {
      const h = layout.groups[k]; x.beginPath(); h.forEach(([u, v], i) => i ? x.lineTo(u, v) : x.moveTo(u, v)); x.closePath(); x.fill(); x.stroke();
    }
    const d = x.getImageData(0, 0, c.width, c.height).data, m = new Uint8Array(c.width * c.height);
    for (let i = 0; i < m.length; i++) m[i] = d[i * 4] > 127 ? 1 : 0;
    return m;
  }

  // Warp several photos (each {canvas, lm}) onto the layout and blend by how face-on each triangle is.
  function render(layout, shots) {
    const { W, H, uv, tris } = layout;
    const acc = new Float32Array(W * H * 3), ws = new Float32Array(W * H);
    const prepared = shots.map(s => {
      const w = s.canvas.width, h = s.canvas.height, P = []; for (let i = 0; i < 468; i++) P.push(pt(s.lm, i, w, h));
      const data = IMG.dataOf(s.canvas).data;
      const ratio = tris.map(([a, b, c]) => {
        const As = ((P[b][0] - P[a][0]) * (P[c][1] - P[a][1]) - (P[c][0] - P[a][0]) * (P[b][1] - P[a][1])) / 2;
        const Am = ((uv[b][0] - uv[a][0]) * (uv[c][1] - uv[a][1]) - (uv[c][0] - uv[a][0]) * (uv[b][1] - uv[a][1])) / 2;
        return Am !== 0 ? As / Am : 0;
      });
      const pos = ratio.filter(r => r > 0).sort((a, b) => a - b), p90 = pos[Math.floor(pos.length * .9)] || 1;
      const triW = ratio.map(r => r > 0 ? Math.min(1, r / p90) ** 2 : 0);
      // smooth per-vertex weights so blends have no hard seams
      const vs = new Float32Array(468), vc = new Float32Array(468);
      tris.forEach((t, k) => t.forEach(v => { vs[v] += triW[k]; vc[v]++; }));
      const vw = Array.from(vs, (s, i) => vc[i] ? s / vc[i] : 0);
      return { w, h, P, data, triW, vw };
    });
    // colour gains so side photos match the front photo where they overlap
    const sample = (S, x, y) => { const X = Math.max(0, Math.min(S.w - 1, x | 0)), Y = Math.max(0, Math.min(S.h - 1, y | 0)), i = (Y * S.w + X) * 4; return [S.data[i], S.data[i + 1], S.data[i + 2]]; };
    const triMean = S => tris.map(([a, b, c]) => { const cx = (S.P[a][0] + S.P[b][0] + S.P[c][0]) / 3, cy = (S.P[a][1] + S.P[b][1] + S.P[c][1]) / 3; return sample(S, cx, cy); });
    const means = prepared.map(triMean), gains = prepared.map(() => [1, 1, 1]);
    for (let s = 1; s < prepared.length; s++) {
      const num = [0, 0, 0], den = [0, 0, 0];
      tris.forEach((t, k) => { const w = Math.min(prepared[0].triW[k], prepared[s].triW[k]); if (w < .25) return;
        for (let c = 0; c < 3; c++) { num[c] += w * means[0][k][c]; den[c] += w * means[s][k][c]; } });
      for (let c = 0; c < 3; c++) gains[s][c] = den[c] > 0 ? num[c] / den[c] : 1;
    }
    prepared.forEach((S, si) => {
      const g = gains[si];
      tris.forEach(([a, b, c], k) => {
        if (S.triW[k] <= 0) return;
        const [u0, v0] = uv[a], [u1, v1] = uv[b], [u2, v2] = uv[c];
        const det = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2); if (Math.abs(det) < 1e-9) return;
        const x0 = Math.max(0, Math.floor(Math.min(u0, u1, u2))), x1 = Math.min(W - 1, Math.ceil(Math.max(u0, u1, u2)));
        const y0 = Math.max(0, Math.floor(Math.min(v0, v1, v2))), y1 = Math.min(H - 1, Math.ceil(Math.max(v0, v1, v2)));
        const [sa, sb, sc] = [S.P[a], S.P[b], S.P[c]], [wa, wb, wc] = [S.vw[a], S.vw[b], S.vw[c]];
        for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
          const px = x + .5, py = y + .5;
          const l0 = ((v1 - v2) * (px - u2) + (u2 - u1) * (py - v2)) / det, l1 = ((v2 - v0) * (px - u2) + (u0 - u2) * (py - v2)) / det, l2 = 1 - l0 - l1;
          if (l0 < -1e-4 || l1 < -1e-4 || l2 < -1e-4) continue;
          const sx = l0 * sa[0] + l1 * sb[0] + l2 * sc[0], sy = l0 * sa[1] + l1 * sb[1] + l2 * sc[1];
          if (sx < 0 || sy < 0 || sx > S.w - 2 || sy > S.h - 2) continue;
          const wt = (l0 * wa + l1 * wb + l2 * wc) * S.triW[k] ** .5; if (wt <= 0) continue;
          const xi = sx | 0, yi = sy | 0, ax = sx - xi, ay = sy - yi, i00 = (yi * S.w + xi) * 4, i10 = i00 + 4, i01 = i00 + S.w * 4, i11 = i01 + 4, d = S.data;
          const p = y * W + x;
          for (let ch = 0; ch < 3; ch++) {
            const v = (d[i00 + ch] * (1 - ax) + d[i10 + ch] * ax) * (1 - ay) + (d[i01 + ch] * (1 - ax) + d[i11 + ch] * ax) * ay;
            acc[p * 3 + ch] += wt * v * g[ch];
          }
          ws[p] += wt;
        }
      });
    });
    const out = IMG.canvasOf(W, H), od = out.getContext("2d").createImageData(W, H), valid = new Uint8Array(W * H);
    for (let p = 0; p < W * H; p++) {
      if (ws[p] > 1e-3) { for (let c = 0; c < 3; c++) od.data[p * 4 + c] = acc[p * 3 + c] / ws[p]; od.data[p * 4 + 3] = 255; valid[p] = 1; }
      else { od.data[p * 4] = 24; od.data[p * 4 + 1] = 20; od.data[p * 4 + 2] = 18; od.data[p * 4 + 3] = 255; }
    }
    out.getContext("2d").putImageData(od, 0, 0);
    return { canvas: out, valid };
  }

  return { landmarker, detect, buildLayout, maskCanvas, render, groups: G };
})();
