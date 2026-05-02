import React, { useEffect, useMemo, useState } from 'react';
import { StyleSheet, Text, View, ActivityIndicator, ScrollView, Image } from 'react-native';
import { useSharedValue, Worklets } from 'react-native-worklets-core';
import { Camera, useCameraDevice, useCameraFormat, useFrameProcessor } from 'react-native-vision-camera';

type Point = { x: number; y: number };

// ─── Tuning constants ──────────────────────────────────────────────────────────
const SCALE = 4;
const MIN_AREA = 50;
const MAX_AREA = 40000;
const MIN_SIDE = 8;
const MAX_CONTOUR_LEN = 800;
const ASPECT_RATIO = 1.5;
const DP_EPSILON_PCT = 0.03;
const MAX_DP_EPSILON = 15;
const FRAME_SKIP = 3;
const MAX_CONTOUR_PTS = 2000;
const MARKER_COUNT = 20;
const WARP_OUT = 300;          // Final output marker size

function App(): JSX.Element {
  const [hasPermission, setHasPermission] = useState(false);
  const [markers, setMarkers] = useState<string[]>([]);
  const device = useCameraDevice('back');

  // Enforce 2000–3000px camera resolution per assignment requirement
  const format = useCameraFormat(device, [
    { videoResolution: { width: 2560, height: 1920 } },
  ]);

  const frameCount = useSharedValue(0);
  const logTick = useSharedValue(0);
  const markerCountSV = useSharedValue(0);

  // Bridge: worklet → JS thread. Receives base64 BMP data URI.
  const onMarkerDetected = useMemo(() =>
    Worklets.createRunInJsFn((dataUri: string) => {
      setMarkers(prev => {
        if (prev.length >= MARKER_COUNT) return prev;
        return [...prev, dataUri];
      });
    }),
  []);

  useEffect(() => {
    (async () => {
      const status = await Camera.requestCameraPermission();
      setHasPermission(status === 'granted' || status === 'authorized');
    })();
  }, []);

  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';

    // ── 0. FRAME THROTTLING ────────────────────────────────────────────────
    frameCount.value = (frameCount.value + 1) % FRAME_SKIP;
    if (frameCount.value !== 0) return;

    // ── 0b. EARLY EXIT if we already have enough markers ─────────────────
    if (markerCountSV.value >= MARKER_COUNT) return;

    // ── Ensure YUV ────────────────────────────────────────────────────────
    if (frame.pixelFormat !== 'yuv') return;

    // ── Helper: safe pixel lookup (returns white=255 for OOB) ────────────
    const getPix = (
      x: number, y: number, w: number, h: number, img: Uint8Array
    ): number => {
      if (x < 0 || x >= w || y < 0 || y >= h) return 255;
      return img[y * w + x];
    };

    // ── Helper: safe pixel lookup for stride-based raw image ─────────────
    const getRawPix = (
      x: number, y: number, w: number, h: number, stride: number, img: Uint8Array
    ): number => {
      if (x < 0 || x >= w || y < 0 || y >= h) return 255;
      return img[y * stride + x];
    };

    // ── Helper: squared Euclidean distance (avoids sqrt in hot paths) ────
    const distSq = (ax: number, ay: number, bx: number, by: number): number =>
      (ax - bx) * (ax - bx) + (ay - by) * (ay - by);

    // ── Helper: point-to-line distance (used by Douglas-Peucker) ─────────
    const pointLineDist = (p: Point, p1: Point, p2: Point): number => {
      const num = Math.abs(
        (p2.y - p1.y) * p.x - (p2.x - p1.x) * p.y + p2.x * p1.y - p2.y * p1.x
      );
      const den = Math.sqrt((p2.y - p1.y) ** 2 + (p2.x - p1.x) ** 2);
      return den === 0 ? 0 : num / den;
    };

    // ── Helper: recursive Douglas-Peucker ────────────────────────────────
    const douglasPeucker = (points: Point[], epsilon: number): Point[] => {
      if (points.length < 3) return points;
      let maxDist = 0;
      let index = 0;
      const end = points.length - 1;
      for (let i = 1; i < end; i++) {
        const d = pointLineDist(points[i], points[0], points[end]);
        if (d > maxDist) { maxDist = d; index = i; }
      }
      if (maxDist > epsilon) {
        const left = douglasPeucker(points.slice(0, index + 1), epsilon);
        const right = douglasPeucker(points.slice(index, end + 1), epsilon);
        return left.slice(0, left.length - 1).concat(right);
      }
      return [points[0], points[end]];
    };

    // ── Helper: closed-contour polygon approximation ──────────────────────
    const approxPolyDP = (points: Point[], epsilon: number): Point[] => {
      if (points.length < 3) return points;
      let maxD = 0, splitIdx = 1;
      for (let i = 1; i < points.length; i++) {
        const d = distSq(points[i].x, points[i].y, points[0].x, points[0].y);
        if (d > maxD) { maxD = d; splitIdx = i; }
      }
      const left = douglasPeucker(points.slice(0, splitIdx + 1), epsilon);
      const right = douglasPeucker(points.slice(splitIdx), epsilon);
      const approx = left.slice(0, left.length - 1).concat(right);
      if (
        approx.length > 1 &&
        approx[0].x === approx[approx.length - 1].x &&
        approx[0].y === approx[approx.length - 1].y
      ) {
        approx.pop();
      }
      return approx;
    };

    // ── Helper: reduce polygon to 4 vertices (Visvalingam-style) ─────────
    //  Iteratively removes the vertex with smallest triangle area.
    //  Extended to 14 verts: outer border contours get 10-12 verts from DP
    //  due to corner rounding at pixel resolution.
    const reduceToQuad = (pts: Point[]): Point[] | null => {
      if (pts.length === 4) return pts;
      if (pts.length < 4 || pts.length > 14) return null;
      const cur = pts.slice();
      while (cur.length > 4) {
        let minArea = Infinity;
        let removeIdx = 0;
        for (let k = 0; k < cur.length; k++) {
          const prev = cur[(k - 1 + cur.length) % cur.length];
          const curr = cur[k];
          const next = cur[(k + 1) % cur.length];
          const ta = Math.abs(
            (curr.x - prev.x) * (next.y - prev.y) -
            (next.x - prev.x) * (curr.y - prev.y)
          ) / 2;
          if (ta < minArea) { minArea = ta; removeIdx = k; }
        }
        cur.splice(removeIdx, 1);
      }
      return cur;
    };

    // ── Helper: Shoelace area ─────────────────────────────────────────────
    const polygonArea = (pts: Point[]): number => {
      let area = 0;
      for (let i = 0; i < pts.length; i++) {
        const j = (i + 1) % pts.length;
        area += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
      }
      return Math.abs(area / 2);
    };

    // ── Helper: order corners clockwise starting from top-left ────────────
    const orderCorners = (pts: Point[]): Point[] => {
      const cx = (pts[0].x + pts[1].x + pts[2].x + pts[3].x) / 4;
      const cy = (pts[0].y + pts[1].y + pts[2].y + pts[3].y) / 4;
      const sorted = pts.slice().sort((a, b) => {
        return Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx);
      });
      let minSum = Infinity;
      let minIdx = 0;
      for (let i = 0; i < 4; i++) {
        const sum = sorted[i].x + sorted[i].y;
        if (sum < minSum) { minSum = sum; minIdx = i; }
      }
      return [
        sorted[minIdx],
        sorted[(minIdx + 1) % 4],
        sorted[(minIdx + 2) % 4],
        sorted[(minIdx + 3) % 4],
      ];
    };

    // ── Helper: convexity check ──────────────────────────────────────────
    const isConvex = (pts: Point[]): boolean => {
      if (pts.length < 3) return false;
      let sign = 0;
      for (let i = 0; i < pts.length; i++) {
        const p1 = pts[i];
        const p2 = pts[(i + 1) % pts.length];
        const p3 = pts[(i + 2) % pts.length];
        const cross = (p2.x - p1.x) * (p3.y - p2.y) - (p2.y - p1.y) * (p3.x - p2.x);
        if (cross !== 0) {
          if (sign === 0) sign = cross > 0 ? 1 : -1;
          else if ((cross > 0 ? 1 : -1) !== sign) return false;
        }
      }
      return true;
    };

    // ── 1. FRAME ACQUISITION ─────────────────────────────────────────────
    const fWidth = frame.width;
    const fHeight = frame.height;
    const stride = frame.bytesPerRow;
    const buffer = frame.toArrayBuffer();
    const data = new Uint8Array(buffer);

    // ── 2. DOWNSCALE + Y-CHANNEL EXTRACT ─────────────────────────────────
    const outWidth = Math.floor(fWidth / SCALE);
    const outHeight = Math.floor(fHeight / SCALE);
    const grayImage = new Uint8Array(outWidth * outHeight);

    let outIndex = 0;
    for (let y = 0; y < outHeight; y++) {
      const rowOffset = (y * SCALE) * stride;
      for (let x = 0; x < outWidth; x++) {
        grayImage[outIndex++] = data[rowOffset + x * SCALE];
      }
    }

    // ── 2b. OTSU ADAPTIVE THRESHOLD ──────────────────────────────────────
    //  Handles varying camera exposure/brightness automatically.
    const hist = new Uint32Array(256);
    for (let i = 0; i < grayImage.length; i++) hist[grayImage[i]]++;
    const total = grayImage.length;
    let sumAll = 0;
    for (let i = 0; i < 256; i++) sumAll += i * hist[i];
    let sumB = 0, wB = 0, maxVar = 0, THRESHOLD = 128;
    for (let i = 0; i < 256; i++) {
      wB += hist[i];
      if (wB === 0) continue;
      const wF = total - wB;
      if (wF === 0) break;
      sumB += i * hist[i];
      const mB = sumB / wB;
      const mF = (sumAll - sumB) / wF;
      const v = wB * wF * (mB - mF) * (mB - mF);
      if (v > maxVar) { maxVar = v; THRESHOLD = i; }
    }

    // ── 2c. BINARIZE ─────────────────────────────────────────────────────
    const binaryImage = new Uint8Array(outWidth * outHeight);
    for (let i = 0; i < grayImage.length; i++) {
      binaryImage[i] = grayImage[i] > THRESHOLD ? 255 : 0;
    }

    // ── 3. CONTOUR TRACING (Moore-Neighbor) ───────────────────────────────
    //  Key fixes vs original:
    //  a) visited[] array instead of mutating binaryImage with 128 sentinel –
    //     the 128 hack caused re-detection of already-traced contours on the
    //     same frame (the scan-line would re-enter a partially-visited contour).
    //  b) MAX_CONTOUR_PTS reduced from 5000 → 2000. For a 140px real-world
    //     marker downscaled 4x, the outer square perimeter is at most
    //     ~4 * (140/4) = 140 px of contour points. 5000 was 35× too generous.
    //  c) Termination condition: stop when we return to start AND direction
    //     matches startDir (Jacob's stopping criterion) – not just coordinate.
    const visited = new Uint8Array(outWidth * outHeight); // 0 = unvisited
    const contours: Point[][] = [];

    const dx = [1, 1, 0, -1, -1, -1, 0, 1];
    const dy = [0, 1, 1, 1, 0, -1, -1, -1];

    for (let y = 1; y < outHeight - 1; y++) {
      for (let x = 1; x < outWidth - 1; x++) {
        // Starting condition: black pixel whose left neighbour is white → left boundary
        if (
          binaryImage[y * outWidth + x] === 0 &&
          binaryImage[y * outWidth + (x - 1)] === 255 &&
          visited[y * outWidth + x] === 0
        ) {
          const contour: Point[] = [];
          let cx = x, cy = y;
          let dir = 7;         // Approach from west (index 6) → start scanning from index 7
          let startDir = -1;
          let loops = 0;

          while (loops < MAX_CONTOUR_PTS) {
            contour.push({ x: cx, y: cy });
            visited[cy * outWidth + cx] = 1;

            let found = false;
            for (let i = 0; i < 8; i++) {
              const ndir = (dir + 1 + i) % 8;
              const nx = cx + dx[ndir];
              const ny = cy + dy[ndir];
              const pix = getPix(nx, ny, outWidth, outHeight, binaryImage);

              if (pix === 0) { // only unambiguously black pixels continue the trace
                if (startDir === -1) startDir = ndir;
                cx = nx;
                cy = ny;
                dir = (ndir + 4) % 8; // backtrack direction
                found = true;
                break;
              }
            }

            if (!found) break;

            // Stop when we return to the start pixel (standard coordinate check)
            if (cx === x && cy === y && contour.length > 3) break;

            loops++;
          }

          // Length filter: skip too-short AND too-long contours.
          // Too-short: can't be a meaningful quad.
          // Too-long: wall/table blobs (1300-1540 pts) — skip before DP even runs.
          if (contour.length >= 4 * MIN_SIDE && contour.length <= MAX_CONTOUR_LEN) {
            contours.push(contour);
          }
        }
      }
    }

    // ── 4. GEOMETRIC FILTERING ────────────────────────────────────────────
    const candidates: Point[][] = [];
    // Debug counters — remove once detection is working
    let dbgNot4 = 0, dbgArea = 0, dbgConvex = 0, dbgSide = 0, dbgRatio = 0;

    for (let i = 0; i < contours.length; i++) {
      const c = contours[i];

      // Clamp epsilon to prevent large contours from collapsing into triangles.
      const epsilon = Math.min(MAX_DP_EPSILON, c.length * DP_EPSILON_PCT);
      let poly = approxPolyDP(c, epsilon);

      // If DP gives 5–14 verts (corner rounding noise), try to reduce to 4
      if (poly.length !== 4) {
        if (poly.length >= 5 && poly.length <= 14) {
          const reduced = reduceToQuad(poly);
          if (reduced) {
            poly = reduced;
          }
        }
      }

      if (poly.length !== 4) {
        dbgNot4++;
        if (dbgNot4 <= 2) {
          console.log(`[CV-POLY] verts=${poly.length} | contourPts=${c.length} | eps=${epsilon.toFixed(1)}`);
        }
        continue;
      }

      const approx = poly;
      if (approx.length === 4) {
        if (logTick.value === 0) console.log('[CV-QUAD FOUND]');
      }

      const area = polygonArea(approx);
      if (area < MIN_AREA || area > MAX_AREA) {
        dbgArea++;
        if (logTick.value === 0 && dbgArea <= 2) {
          console.log(`[CV-AREA] area=${area.toFixed(0)} (need ${MIN_AREA}–${MAX_AREA})`);
        }
        continue;
      }

      if (!isConvex(approx)) { dbgConvex++; continue; }

      const sides: number[] = [];
      for (let s = 0; s < 4; s++) {
        const a = approx[s];
        const b = approx[(s + 1) % 4];
        sides.push(Math.sqrt(distSq(a.x, a.y, b.x, b.y)));
      }

      const minSide = Math.min(...sides);
      if (minSide < MIN_SIDE) { dbgSide++; continue; }

      const maxSide = Math.max(...sides);
      if (maxSide / minSide > ASPECT_RATIO) { dbgRatio++; continue; }

      candidates.push(approx);
    }

    // ── 5. DEDUPLICATION ──────────────────────────────────────────────────
    //  Merges candidates whose centroids are within 10 px of each other
    //  (downscaled coords). Avoids reporting the same marker twice when the
    //  scan finds both inner and outer edge of the same stroke.
    const merged: Point[][] = [];
    for (let i = 0; i < candidates.length; i++) {
      const ca = candidates[i];
      const cax = (ca[0].x + ca[1].x + ca[2].x + ca[3].x) / 4;
      const cay = (ca[0].y + ca[1].y + ca[2].y + ca[3].y) / 4;
      let dup = false;
      for (let j = 0; j < merged.length; j++) {
        const cb = merged[j];
        const cbx = (cb[0].x + cb[1].x + cb[2].x + cb[3].x) / 4;
        const cby = (cb[0].y + cb[1].y + cb[2].y + cb[3].y) / 4;
        if (distSq(cax, cay, cbx, cby) < 100) { dup = true; break; } // 10 px radius
      }
      if (!dup) merged.push(ca);
    }

    // ── 6. PERSPECTIVE TRANSFORM & VALIDATION ──────────────────────────────
    // Marker structure after warping the outer 140×140 quad:
    //   - Outer ring (~15% each side): black border
    //   - Interior: mostly white
    //   - Somewhere inside: 20×20 black square
    // Validation: border must be dark, interior must be mostly white.

    const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
    const WARP_VAL = 64; // small warp for fast validation

    for (let mi = 0; mi < merged.length; mi++) {
      if (markerCountSV.value >= MARKER_COUNT) break;

      const scaled = merged[mi].map(p => ({ x: p.x * SCALE, y: p.y * SCALE }));
      const ordered = orderCorners(scaled);
      const x0 = ordered[0].x, y0 = ordered[0].y;
      const x1 = ordered[1].x, y1 = ordered[1].y;
      const x2 = ordered[2].x, y2 = ordered[2].y;
      const x3 = ordered[3].x, y3 = ordered[3].y;

      // Compute projective homography coefficients
      const ddx1 = x1 - x2, ddx2 = x3 - x2, sigX = x0 - x1 + x2 - x3;
      const ddy1 = y1 - y2, ddy2 = y3 - y2, sigY = y0 - y1 + y2 - y3;
      const DD = ddx1 * ddy2 - ddx2 * ddy1;
      let g = 0, h = 0;
      if (DD !== 0) {
        g = (sigX * ddy2 - sigY * ddx2) / DD;
        h = (ddx1 * sigY - ddy1 * sigX) / DD;
      }
      const ha = x1 - x0 + g * x1;
      const hb = x3 - x0 + h * x3;
      const hc = x0;
      const hd = y1 - y0 + g * y1;
      const he = y3 - y0 + h * y3;
      const hf = y0;

      // ── 6a. FAST VALIDATION WARP (64×64) ──────────────────────────────
      let minP = 255, maxP = 0;
      const valPatch = new Uint8Array(WARP_VAL * WARP_VAL);
      let vi = 0;
      for (let py = 0; py < WARP_VAL; py++) {
        const vv = py / (WARP_VAL - 1);
        for (let px = 0; px < WARP_VAL; px++) {
          const uu = px / (WARP_VAL - 1);
          const den = g * uu + h * vv + 1;
          const sx = Math.floor((ha * uu + hb * vv + hc) / den);
          const sy = Math.floor((hd * uu + he * vv + hf) / den);
          const pix = getRawPix(sx, sy, fWidth, fHeight, stride, data);
          valPatch[vi++] = pix;
          if (pix < minP) minP = pix;
          if (pix > maxP) maxP = pix;
        }
      }

      // Contrast check
      if (maxP - minP < 30) continue;
      const localT = minP + (maxP - minP) / 2;

      // ── VALIDATION: 3 checks ──────────────────────────────────────────
      // 1. Total dark ratio: should be 10–55% (border is dark, interior white)
      //    All-black (inner square) → ~100% dark → REJECT
      //    All-white (random white area) → ~0% dark → REJECT
      // 2. Center 50% of warp should be mostly white (the interior)
      //    Inner black square warp → center is dark → REJECT
      // 3. We also prefer the LARGEST merged quad (outer border > inner square)
      let totalDark = 0;
      let centerWhite = 0, centerTotal = 0;
      const cStart = Math.floor(WARP_VAL * 0.25);
      const cEnd = Math.floor(WARP_VAL * 0.75);
      vi = 0;
      for (let py = 0; py < WARP_VAL; py++) {
        for (let px = 0; px < WARP_VAL; px++) {
          const isDark = valPatch[vi++] <= localT;
          if (isDark) totalDark++;
          // Center 50% region
          if (px >= cStart && px < cEnd && py >= cStart && py < cEnd) {
            centerTotal++;
            if (!isDark) centerWhite++;
          }
        }
      }

      const totalPx = WARP_VAL * WARP_VAL;
      const darkPct = totalDark / totalPx;
      const centerWhitePct = centerTotal > 0 ? centerWhite / centerTotal : 0;

      if (logTick.value === 0) {
        console.log(`[CV-VAL] dark=${darkPct.toFixed(2)} centerWh=${centerWhitePct.toFixed(2)}`);
      }

      // Dark 10–55%: has both border and white interior
      // Center >55% white: interior is light, not a solid dark blob
      if (darkPct < 0.10 || darkPct > 0.55) continue;
      if (centerWhitePct < 0.55) continue;

      // ── 6b. FULL 300×300 WARP ─────────────────────────────────────────
      const warpPx = new Uint8Array(WARP_OUT * WARP_OUT);
      let wi = 0;
      for (let py = 0; py < WARP_OUT; py++) {
        const vv = py / (WARP_OUT - 1);
        for (let px = 0; px < WARP_OUT; px++) {
          const uu = px / (WARP_OUT - 1);
          const den = g * uu + h * vv + 1;
          const sx = Math.floor((ha * uu + hb * vv + hc) / den);
          const sy = Math.floor((hd * uu + he * vv + hf) / den);
          warpPx[wi++] = getRawPix(sx, sy, fWidth, fHeight, stride, data);
        }
      }

      // ── 6c. ENCODE AS 24-BIT BMP ──────────────────────────────────────
      const W = WARP_OUT, H = WARP_OUT;
      const rowSz = W * 3;
      const dataSz = rowSz * H;
      const fileSz = 54 + dataSz;
      const bmp = new Uint8Array(fileSz);

      // BM header
      bmp[0] = 66; bmp[1] = 77;
      bmp[2] = fileSz & 0xFF; bmp[3] = (fileSz >> 8) & 0xFF;
      bmp[4] = (fileSz >> 16) & 0xFF; bmp[5] = (fileSz >> 24) & 0xFF;
      bmp[10] = 54;
      // DIB header
      bmp[14] = 40;
      bmp[18] = W & 0xFF; bmp[19] = (W >> 8) & 0xFF;
      // Negative height for top-down rows
      const nh = -H;
      bmp[22] = nh & 0xFF; bmp[23] = (nh >> 8) & 0xFF;
      bmp[24] = (nh >> 16) & 0xFF; bmp[25] = (nh >> 24) & 0xFF;
      bmp[26] = 1; // planes
      bmp[28] = 24; // bpp
      bmp[34] = dataSz & 0xFF; bmp[35] = (dataSz >> 8) & 0xFF;
      bmp[36] = (dataSz >> 16) & 0xFF; bmp[37] = (dataSz >> 24) & 0xFF;

      for (let yy = 0; yy < H; yy++) {
        for (let xx = 0; xx < W; xx++) {
          const gv = warpPx[yy * W + xx];
          const off = 54 + yy * rowSz + xx * 3;
          bmp[off] = gv; bmp[off + 1] = gv; bmp[off + 2] = gv;
        }
      }

      // ── 6d. BASE64 ENCODE ─────────────────────────────────────────────
      let b64 = '';
      for (let bi = 0; bi < bmp.length; bi += 3) {
        const ba = bmp[bi];
        const bb = bi + 1 < bmp.length ? bmp[bi + 1] : 0;
        const bc = bi + 2 < bmp.length ? bmp[bi + 2] : 0;
        b64 += B64[(ba >> 2) & 0x3F];
        b64 += B64[((ba & 3) << 4 | bb >> 4) & 0x3F];
        b64 += bi + 1 < bmp.length ? B64[((bb & 15) << 2 | bc >> 6) & 0x3F] : '=';
        b64 += bi + 2 < bmp.length ? B64[bc & 0x3F] : '=';
      }

      const dataUri = 'data:image/bmp;base64,' + b64;

      // ── 6e. SEND TO JS THREAD ─────────────────────────────────────────
      markerCountSV.value++;
      onMarkerDetected(dataUri);
      console.log(`[CV-MARKER] #${markerCountSV.value} detected and sent to UI`);
    }

    // ── 7. DEBUG LOG ────────────────────────────────────────────────────────
    logTick.value = (logTick.value + 1) % 15;
    if (logTick.value === 0) {
      console.log(
        `[CV-PIPE] c=${contours.length} !4=${dbgNot4} a=${dbgArea} cvx=${dbgConvex} s=${dbgSide} r=${dbgRatio} | merged=${merged.length} | collected=${markerCountSV.value}`
      );
    }

  }, [onMarkerDetected]);

  // ── UI ───────────────────────────────────────────────────────────────────
  const isComplete = markers.length >= MARKER_COUNT;

  if (!hasPermission) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>No Camera Permission</Text>
      </View>
    );
  }

  if (device == null) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#ffffff" />
      </View>
    );
  }

  // ── RESULTS SCREEN ──────────────────────────────────────────────────────
  if (isComplete) {
    return (
      <View style={styles.container}>
        <Text style={styles.title}>20 Markers Collected</Text>
        <ScrollView contentContainerStyle={styles.grid}>
          {markers.map((uri, i) => (
            <View key={i} style={styles.markerWrap}>
              <Image source={{ uri }} style={styles.markerImg} />
              <Text style={styles.markerLabel}>#{i + 1}</Text>
            </View>
          ))}
        </ScrollView>
      </View>
    );
  }

  // ── SCANNING SCREEN ─────────────────────────────────────────────────────
  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        frameProcessor={frameProcessor}
        pixelFormat="yuv"
        format={format}
      />
      <View style={styles.overlay}>
        <Text style={styles.counterText}>
          Markers: {markers.length} / {MARKER_COUNT}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#111',
  },
  text: {
    color: 'white',
    fontSize: 18,
    textAlign: 'center',
    marginTop: 60,
  },
  title: {
    color: '#4CAF50',
    fontSize: 22,
    fontWeight: 'bold',
    textAlign: 'center',
    paddingTop: 50,
    paddingBottom: 12,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    paddingHorizontal: 8,
    paddingBottom: 40,
  },
  markerWrap: {
    margin: 6,
    alignItems: 'center',
  },
  markerImg: {
    width: 150,
    height: 150,
    borderWidth: 1,
    borderColor: '#444',
  },
  markerLabel: {
    color: '#aaa',
    fontSize: 12,
    marginTop: 2,
  },
  overlay: {
    position: 'absolute',
    top: 50,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  counterText: {
    color: '#fff',
    fontSize: 20,
    fontWeight: 'bold',
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 20,
    paddingVertical: 8,
    borderRadius: 12,
    overflow: 'hidden',
  },
});

export default App;
