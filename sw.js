// Cache-first app shell: the app opens from the phone's cache, even offline.
const VERSION = "skinlog-v7";
const SHELL = ["./", "./index.html", "./imaging.js", "./facemap.js", "./manifest.webmanifest", "./apple-touch-icon.png", "./icon-192.png", "./icon-512.png"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(VERSION).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== VERSION && k !== "skinlog-models").map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET") return;
  // face model + its runtime: cache after the first download so maps work offline
  if (url.hostname === "cdn.jsdelivr.net" || url.hostname === "storage.googleapis.com") {
    e.respondWith(caches.open("skinlog-models").then(c => c.match(e.request).then(hit => hit || fetch(e.request).then(res => { if (res.ok) c.put(e.request, res.clone()); return res; }))));
    return;
  }
  if (url.origin !== location.origin) return; // API calls go straight to network
  e.respondWith(
    caches.match(e.request, { ignoreSearch: true }).then(hit => {
      // Serve cached copy instantly; refresh the cache in the background for next launch.
      const net = fetch(e.request).then(res => {
        if (res.ok) { const copy = res.clone(); caches.open(VERSION).then(c => c.put(e.request, copy)); }
        return res;
      }).catch(() => hit);
      return hit || net;
    })
  );
});
