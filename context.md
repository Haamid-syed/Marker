# Project Context: MarkerApp

## Overview
A React Native (0.72.x) Android application utilizing `react-native-vision-camera` to build a
highly optimized, completely custom computer vision pipeline running natively on the UI thread
via `react-native-worklets-core`. The goal is to detect, validate, and extract a specific
physical marker — an **internship assignment for Alemeno**.

---

## The Marker
- **Outer boundary:** 140×140 solid black square border (thick frame). This is what the contour tracing detects.
- **White interior:** Area inside the black border is white.
- **Inner marker:** 20×20 solid black square, positioned **anywhere** inside the white area (position varies in test images).
- Design constraints (per assignment): black & white only, overall shape must be square, ≥ 60% of interior area must be empty so data can be encoded.

---

## Assignment Requirements (Alemeno Frontend Internship)

| Requirement | Status |
|-------------|--------|
| React Native Android app | ✅ Done |
| Live camera feed (2000×2000 – 3000×3000 px) | ✅ Done (`useCameraFormat` enforced) |
| Detect custom marker in real-time | ✅ Done (pipeline with Otsu + contour + DP + geometry) |
| Isolate & extract marker (tight crop, no padding, no skew) | ✅ Done (projective homography → 300×300 warp) |
| Orientation correction (all rotations) | ⚠️ Marker is symmetric — needs design change or corner pip |
| Display 20 processed markers from 20 different frames (300×300 px each) | ✅ Done (results screen with scrollable grid) |
| Detection must be robust — no false positives on incorrect shapes | ✅ Done (border-dark + interior-white validation) |
| Speed: scan-to-result < 3000 ms | ⚠️ Needs measurement |

---

## Current Pipeline (All Steps Implemented)

1. **Luminance Extraction** — YUV Y-channel from first memory plane.
2. **Downsampling** — Nearest-neighbor 4× downscale (single pass).
3. **Otsu Adaptive Threshold** — Histogram-based optimal threshold, handles varying camera exposure/brightness automatically.
4. **Contour Tracing** — Moore-Neighbor with separate `visited[]` array. Max contour length filter (800) and iteration cap (2000).
5. **Polygon Approximation** — Douglas-Peucker with Euclidean split + Visvalingam `reduceToQuad` post-processor.
6. **Geometric Validation** — 4-vertex → Shoelace area → convexity → side uniformity → centroid dedup.
7. **Perspective Transform** — Projective homography (DLT). Fast 64×64 validation warp, then full 300×300 output warp.
8. **Marker Validation** — Border region (outer 15%) must be >55% dark. Interior (inner 70%) must be >60% white.
9. **BMP Encoding** — 24-bit BMP generated in-worklet, base64 encoded, passed to JS via `Worklets.createRunInJsFn`.
10. **Frame Collection & UI** — Collects up to 20 validated markers. Switches to results screen (scrollable grid of 300×300 tiles).

---

## Tuning Constants

```ts
const SCALE           = 4;      // Spatial downscale factor
const MIN_AREA        = 50;     // Min quad area in downscaled px²
const MAX_AREA        = 40000;  // Max quad area
const MIN_SIDE        = 8;      // Min side length in downscaled px
const MAX_CONTOUR_LEN = 800;    // Max contour points
const ASPECT_RATIO    = 1.5;    // Max side ratio
const DP_EPSILON_PCT  = 0.03;   // Douglas-Peucker epsilon as % of contour length
const MAX_DP_EPSILON  = 15;     // Hard cap on DP epsilon
const FRAME_SKIP      = 3;      // Process every 3rd frame
const MAX_CONTOUR_PTS = 2000;   // Hard iteration cap per contour trace
const MARKER_COUNT    = 20;     // Number of markers to collect
const WARP_OUT        = 300;    // Output marker image size
```

---

## Architecture Invariants — Do NOT Change These

1. **Package Versions:**
   - `react-native-vision-camera`: `^3.9.0`
   - `react-native-worklets-core`: `^0.2.4` *(v1.x removes `createRunInJsFn`. Fatal if upgraded.)*
2. **Hermes Evaluation Order:**
   - `import 'react-native-worklets-core';` must be the absolute first import in `index.js`.
3. **Babel Plugin Order:**
   - `['react-native-worklets-core/plugin']` BEFORE `['react-native-reanimated/plugin']`.
4. **Android `minSdkVersion = 26`** — required for `frame.toArrayBuffer()` (`HardwareBuffer` API).
5. **All CV helpers defined inside `useFrameProcessor` callback** — required for worklet serialization.
6. **`useSharedValue` from `react-native-worklets-core`**, NOT from `react-native-reanimated`.
7. **`Worklets.createRunInJsFn`** for worklet→JS bridge (returns Promise, fire-and-forget in worklet).

---

## Remaining Work

### Orientation Correction
Current marker design (symmetric square-in-square) has **no orientation anchor**.
Options:
- **Option A:** Move the 20×20 inner square to a specific asymmetric position (e.g., always top-left offset)
- **Option B:** Add a filled corner pip to one corner of the outer border
- Once an anchor exists, detect it in the warped image and rotate 0/90/180/270° accordingly.

### Performance Measurement
- Measure total scan-to-result time (should be <3000ms per assignment).
- If slow: reduce `WARP_OUT` to 150 for validation, scale display only.

### APK Build
- `cd android && ./gradlew assembleRelease` for the deliverable APK.
