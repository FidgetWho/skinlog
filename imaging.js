// imaging.js: image processing for Skin Log (no dependencies).
// Burst stacking, sharpness, frame alignment, relief (photometric stereo), redness maps.
"use strict";
const IMG = (() => {
  const lin = new Float32Array(256);
  for (let i = 0; i < 256; i++) { const x = i / 255; lin[i] = x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4); }

  function canvasOf(w, h) { const c = document.createElement("canvas"); c.width = w; c.height = h; return c; }
  function dataOf(c) { return c.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, c.width, c.height); }

  function gray(d, w, h) {
    const g = new Float32Array(w * h);
    for (let i = 0, j = 0; i < g.length; i++, j += 4) g[i] = .299 * d[j] + .587 * d[j + 1] + .114 * d[j + 2];
    return g;
  }
  function half(g, w, h) {
    const w2 = w >> 1, h2 = h >> 1, o = new Float32Array(w2 * h2);
    for (let y = 0; y < h2; y++) for (let x = 0; x < w2; x++) {
      const i = 2 * y * w + 2 * x; o[y * w2 + x] = (g[i] + g[i + 1] + g[i + w] + g[i + w + 1]) / 4;
    }
    return { g: o, w: w2, h: h2 };
  }
  // separable box blur (sliding window), edge-clamped
  function box(src, w, h, r) {
    const tmp = new Float32Array(w * h), out = new Float32Array(w * h), n = 2 * r + 1;
    for (let y = 0; y < h; y++) {
      const row = y * w; let s = 0;
      for (let k = -r; k <= r; k++) s += src[row + Math.min(w - 1, Math.max(0, k))];
      for (let x = 0; x < w; x++) {
        tmp[row + x] = s / n;
        s += src[row + Math.min(w - 1, x + r + 1)] - src[row + Math.max(0, x - r)];
      }
    }
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = -r; k <= r; k++) s += tmp[Math.min(h - 1, Math.max(0, k)) * w + x];
      for (let y = 0; y < h; y++) {
        out[y * w + x] = s / n;
        s += tmp[Math.min(h - 1, y + r + 1) * w + x] - tmp[Math.max(0, y - r) * w + x];
      }
    }
    return out;
  }
  const blur = (src, w, h, r) => box(box(box(src, w, h, r), w, h, r), w, h, r);   // ≈ gaussian
  function highpass(g, w, h, r) { const b = box(g, w, h, r), o = new Float32Array(g.length); for (let i = 0; i < g.length; i++) o[i] = g[i] - b[i]; return o; }

  // Sharpness: variance of the Laplacian on a small centre crop (higher = sharper)
  function sharpness(src, sw, sh, sx = 0, sy = 0, sWidth, sHeight) {
    const w = 320, cw = sWidth || sw, ch = sHeight || sh, h = Math.max(8, Math.round(w * ch / cw));
    const t = canvasOf(w, h), x = t.getContext("2d", { willReadFrequently: true });
    x.drawImage(src, sx, sy, cw, ch, 0, 0, w, h);
    const g = gray(x.getImageData(0, 0, w, h).data, w, h);
    const x0 = Math.round(w * .2), y0 = Math.round(h * .2), x1 = Math.round(w * .8), y1 = Math.round(h * .8);
    let s = 0, q = 0, n = 0;
    for (let yy = y0; yy < y1; yy++) for (let xx = x0; xx < x1; xx++) {
      const i = yy * w + xx, L = 4 * g[i] - g[i - 1] - g[i + 1] - g[i - w] - g[i + w];
      s += L; q += L * L; n++;
    }
    return q / n - (s / n) ** 2;
  }

  // Tile-wise translation estimate between two high-passed gray images (same size).
  // init: optional {sx,sy} arrays (in this level's pixels). Returns {sx, sy, cost} per tile.
  function tileShifts(A, B, w, h, TX, TY, R, init) {
    const tw = w / TX, th = h / TY, sx = new Float32Array(TX * TY), sy = new Float32Array(TX * TY), cost = new Float32Array(TX * TY);
    for (let ty = 0; ty < TY; ty++) for (let tx = 0; tx < TX; tx++) {
      const t = ty * TX + tx;
      const bx = init ? Math.round(init.sx[t]) : 0, by = init ? Math.round(init.sy[t]) : 0;
      const x0 = Math.floor(tx * tw), y0 = Math.floor(ty * th), x1 = Math.floor((tx + 1) * tw), y1 = Math.floor((ty + 1) * th);
      const S = 2 * R + 1, E = new Float32Array(S * S);
      let best = Infinity, bi = 0;
      for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
        let e = 0, n = 0;
        const ox = bx + dx, oy = by + dy;
        for (let y = y0; y < y1; y += 2) {
          const yb = y + oy; if (yb < 0 || yb >= h) continue;
          for (let x = x0; x < x1; x += 2) {
            const xb = x + ox; if (xb < 0 || xb >= w) continue;
            const d = A[y * w + x] - B[yb * w + xb]; e += d * d; n++;
          }
        }
        const v = n ? e / n : Infinity, k = (dy + R) * S + (dx + R);
        E[k] = v; if (v < best) { best = v; bi = k; }
      }
      const ix = bi % S, iy = Math.floor(bi / S);
      let fx = 0, fy = 0;
      if (ix > 0 && ix < S - 1) { const l = E[iy * S + ix - 1], c = E[bi], r = E[iy * S + ix + 1], den = l - 2 * c + r; if (den > 0) fx = .5 * (l - r) / den; }
      if (iy > 0 && iy < S - 1) { const u = E[(iy - 1) * S + ix], c = E[bi], d = E[(iy + 1) * S + ix], den = u - 2 * c + d; if (den > 0) fy = .5 * (u - d) / den; }
      sx[t] = bx + ix - R + Math.max(-.5, Math.min(.5, fx));
      sy[t] = by + iy - R + Math.max(-.5, Math.min(.5, fy));
      cost[t] = best;
    }
    // 3x3 median filter on the shift field (kills outlier tiles)
    const med = (arr) => { const o = new Float32Array(arr.length);
      for (let ty = 0; ty < TY; ty++) for (let tx = 0; tx < TX; tx++) {
        const v = []; for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) { const X = tx + i, Y = ty + j; if (X >= 0 && Y >= 0 && X < TX && Y < TY) v.push(arr[Y * TX + X]); }
        v.sort((a, b) => a - b); o[ty * TX + tx] = v[v.length >> 1];
      } return o; };
    return { sx: med(sx), sy: med(sy), cost };
  }
  // Two-level (1/4 then 1/2 scale) alignment of B onto A. Returns full-res tile shift field.
  function align(gA, gB, w, h, TX = 6, TY = 8) {
    const a2 = half(gA, w, h), b2 = half(gB, w, h), a4 = half(a2.g, a2.w, a2.h), b4 = half(b2.g, b2.w, b2.h);
    const A4 = highpass(a4.g, a4.w, a4.h, 4), B4 = highpass(b4.g, b4.w, b4.h, 4);
    const s4 = tileShifts(A4, B4, a4.w, a4.h, TX, TY, 8);
    const init = { sx: s4.sx.map(v => v * 2), sy: s4.sy.map(v => v * 2) };
    const A2 = highpass(a2.g, a2.w, a2.h, 6), B2 = highpass(b2.g, b2.w, b2.h, 6);
    const s2 = tileShifts(A2, B2, a2.w, a2.h, TX, TY, 2, init);
    let c = 0; for (const v of s2.cost) c += v;
    return { sx: s2.sx.map(v => v * 2), sy: s2.sy.map(v => v * 2), TX, TY, cost: c / s2.cost.length };
  }
  function shiftAt(f, x, y, w, h) {
    const fx = Math.min(f.TX - 1, Math.max(0, (x + .5) / (w / f.TX) - .5)), fy = Math.min(f.TY - 1, Math.max(0, (y + .5) / (h / f.TY) - .5));
    const x0 = Math.floor(fx), y0 = Math.floor(fy), x1 = Math.min(f.TX - 1, x0 + 1), y1 = Math.min(f.TY - 1, y0 + 1), ax = fx - x0, ay = fy - y0;
    const i00 = y0 * f.TX + x0, i10 = y0 * f.TX + x1, i01 = y1 * f.TX + x0, i11 = y1 * f.TX + x1;
    return [
      (f.sx[i00] * (1 - ax) + f.sx[i10] * ax) * (1 - ay) + (f.sx[i01] * (1 - ax) + f.sx[i11] * ax) * ay,
      (f.sy[i00] * (1 - ax) + f.sy[i10] * ax) * (1 - ay) + (f.sy[i01] * (1 - ax) + f.sy[i11] * ax) * ay,
    ];
  }

  // Merge a burst: align every frame to the sharpest, robust-average, then a gentle unsharp mask.
  function stack(frames, onProgress) {
    const scores = frames.map(f => sharpness(f, f.width, f.height));
    let ri = 0; scores.forEach((s, i) => { if (s > scores[ri]) ri = i; });
    const ref = frames[ri], w = ref.width, h = ref.height;
    const rd = dataOf(ref).data, rg = gray(rd, w, h);
    const acc = new Float32Array(w * h * 3), ws = new Float32Array(w * h);
    for (let i = 0, j = 0; i < w * h; i++, j += 4) { acc[i * 3] = rd[j]; acc[i * 3 + 1] = rd[j + 1]; acc[i * 3 + 2] = rd[j + 2]; ws[i] = 1; }
    const refCost = [];
    let used = 1;
    // only merge frames reasonably close to the best in sharpness
    const order = frames.map((f, i) => i).filter(i => i !== ri && scores[i] > scores[ri] * 0.55);
    for (const [n, fi] of order.entries()) {
      const f = frames[fi]; if (f.width !== w || f.height !== h) continue;
      const fd = dataOf(f).data, fg = gray(fd, w, h);
      const field = align(rg, fg, w, h);
      refCost.push(field.cost);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const [dx, dy] = shiftAt(field, x, y, w, h);
          const X = x + dx, Y = y + dy;
          if (X < 0 || Y < 0 || X > w - 2 || Y > h - 2) continue;
          const x0 = X | 0, y0 = Y | 0, ax = X - x0, ay = Y - y0;
          const i00 = (y0 * w + x0) * 4, i10 = i00 + 4, i01 = i00 + w * 4, i11 = i01 + 4;
          const w00 = (1 - ax) * (1 - ay), w10 = ax * (1 - ay), w01 = (1 - ax) * ay, w11 = ax * ay;
          const r = fd[i00] * w00 + fd[i10] * w10 + fd[i01] * w01 + fd[i11] * w11;
          const g = fd[i00 + 1] * w00 + fd[i10 + 1] * w10 + fd[i01 + 1] * w01 + fd[i11 + 1] * w11;
          const b = fd[i00 + 2] * w00 + fd[i10 + 2] * w10 + fd[i01 + 2] * w01 + fd[i11 + 2] * w11;
          const p = y * w + x, diff = (.299 * r + .587 * g + .114 * b) - rg[p];
          const wt = 1 / (1 + (diff / 30) ** 2);           // reject ghosts / misaligned pixels
          acc[p * 3] += wt * r; acc[p * 3 + 1] += wt * g; acc[p * 3 + 2] += wt * b; ws[p] += wt;
        }
      }
      used++;
      onProgress && onProgress((n + 1) / order.length);
    }
    // normalize + gentle unsharp mask (noise is lower now, so a little sharpening is safe)
    const out = canvasOf(w, h), od = out.getContext("2d").createImageData(w, h);
    const ch = [0, 1, 2].map(c => { const a = new Float32Array(w * h); for (let i = 0; i < w * h; i++) a[i] = acc[i * 3 + c] / ws[i]; return a; });
    const amt = used > 2 ? 0.3 : 0;
    for (let c = 0; c < 3; c++) {
      const a = ch[c], b = amt ? box(a, w, h, 1) : a;
      for (let i = 0; i < w * h; i++) od.data[i * 4 + c] = a[i] + amt * (a[i] - b[i]);
    }
    for (let i = 0; i < w * h; i++) od.data[i * 4 + 3] = 255;
    out.getContext("2d").putImageData(od, 0, 0);
    return { canvas: out, used, sharp: scores[ri], ref: ri };
  }

  // Relief from four directional exposures (photometric stereo, small-baseline).
  // f = {top, bottom, left, right} canvases (same size). Returns a grey relief canvas.
  function relief(f) {
    const w = f.top.width, h = f.top.height;
    const lum = c => { const d = dataOf(c).data, o = new Float32Array(w * h); for (let i = 0, j = 0; i < o.length; i++, j += 4) o[i] = .2126 * lin[d[j]] + .7152 * lin[d[j + 1]] + .0722 * lin[d[j + 2]] + 1e-4; return o; };
    const T = lum(f.top), base = gray(dataOf(f.top).data, w, h);
    const warped = { top: T };
    for (const k of ["bottom", "left", "right"]) {
      const L = lum(f[k]), g = gray(dataOf(f[k]).data, w, h), field = align(base, g, w, h), o = new Float32Array(w * h);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const [dx, dy] = shiftAt(field, x, y, w, h), X = Math.min(w - 2, Math.max(0, x + dx)), Y = Math.min(h - 2, Math.max(0, y + dy));
        const x0 = X | 0, y0 = Y | 0, ax = X - x0, ay = Y - y0, i = y0 * w + x0;
        o[y * w + x] = (L[i] * (1 - ax) + L[i + 1] * ax) * (1 - ay) + (L[i + w] * (1 - ax) + L[i + w + 1] * ax) * ay;
      }
      warped[k] = o;
    }
    const gx = new Float32Array(w * h), gy = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) {
      gx[i] = (warped.left[i] - warped.right[i]) / (warped.left[i] + warped.right[i]);
      gy[i] = (warped.top[i] - warped.bottom[i]) / (warped.top[i] + warped.bottom[i]);
    }
    const r = Math.max(6, Math.round(w / 90));
    const hx = highpass(gx, w, h, r), hy = highpass(gy, w, h, r);
    const s = new Float32Array(w * h); for (let i = 0; i < s.length; i++) s[i] = .7071 * (hx[i] + hy[i]);
    const sm = box(s, w, h, 1);
    const sorted = Float32Array.from(sm).map(Math.abs).sort(), sd = sorted[Math.floor(sorted.length * .68)] || 1e-3;
    const out = canvasOf(w, h), od = out.getContext("2d").createImageData(w, h);
    for (let i = 0; i < w * h; i++) { const v = 128 + 40 * sm[i] / sd; od.data[i * 4] = od.data[i * 4 + 1] = od.data[i * 4 + 2] = v; od.data[i * 4 + 3] = 255; }
    out.getContext("2d").putImageData(od, 0, 0);
    return out;
  }

  // Inferno-ish colour map
  const LUT = (() => { const stops = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
    const o = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) { const t = i / 255 * (stops.length - 1), k = Math.min(stops.length - 2, Math.floor(t)), a = t - k;
      for (let c = 0; c < 3; c++) o[i * 3 + c] = stops[k][c] * (1 - a) + stops[k + 1][c] * a; }
    return o; })();

  // Redness map from a colour image + validity mask (Uint8Array 0/1). Returns canvas.
  function redness(c, valid) {
    const w = c.width, h = c.height, d = dataOf(c).data;
    const E = new Float32Array(w * h), V = new Float32Array(w * h), lumi = new Float32Array(w * h), sat = new Float32Array(w * h);
    for (let i = 0, j = 0; i < w * h; i++, j += 4) {
      const R = lin[d[j]] + 1e-3, G = lin[d[j + 1]] + 1e-3, B = lin[d[j + 2]] + 1e-3;
      E[i] = 100 * Math.log10(R / G);
      lumi[i] = Math.log(.2126 * R + .7152 * G + .0722 * B);
      const mx = Math.max(R, G, B), mn = Math.min(R, G, B); sat[i] = 1 - mn / mx;
    }
    // shine: much brighter than surroundings and desaturated
    const lb = blur(lumi, w, h, Math.max(4, Math.round(w / 120)));
    const sv = []; for (let i = 0; i < w * h; i += 7) if (valid[i]) sv.push(sat[i]); sv.sort((a, b) => a - b);
    const sLow = sv.length ? sv[Math.floor(sv.length * .25)] : 0;
    for (let i = 0; i < w * h; i++) V[i] = valid[i] && !(lumi[i] - lb[i] > .18 && sat[i] < sLow) ? 1 : 0;
    const r = Math.max(6, Math.round(w / 45));
    const EV = new Float32Array(w * h); for (let i = 0; i < w * h; i++) EV[i] = E[i] * V[i];
    const num = blur(EV, w, h, r), den = blur(V, w, h, r);
    const dE = new Float32Array(w * h); for (let i = 0; i < w * h; i++) dE[i] = E[i] - num[i] / Math.max(den[i], 1e-3);
    const sm = box(dE, w, h, 1);
    const out = canvasOf(w, h), od = out.getContext("2d").createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      const v = Math.max(0, Math.min(255, Math.round((sm[i] + 2) / 14 * 255)));
      const k = valid[i] ? (V[i] ? 1 : .55) : .18;
      od.data[i * 4] = LUT[v * 3] * k; od.data[i * 4 + 1] = LUT[v * 3 + 1] * k; od.data[i * 4 + 2] = LUT[v * 3 + 2] * k; od.data[i * 4 + 3] = 255;
    }
    out.getContext("2d").putImageData(od, 0, 0);
    return out;
  }

  return { canvasOf, dataOf, sharpness, stack, relief, redness, lin };
})();
