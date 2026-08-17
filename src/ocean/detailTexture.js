import {
  DataTexture, RGBAFormat, UnsignedByteType, RepeatWrapping, LinearFilter, LinearMipmapLinearFilter,
} from 'three/webgpu';

// A seamless tiling fbm noise texture, baked once on the CPU. Sampling this is
// vastly cheaper than evaluating MaterialX fbm per fragment. Channels:
//   RG = detail normal perturbation (encoded *0.5+0.5)
//   B  = low-frequency value (foam break-up / variation)
//   A  = higher-frequency value
// Statistics of the field as it was originally baked (3 octaves, persistence
// 0.5), measured over the whole tile. Every carve constant in foamShading.js is
// tuned against these — the `sub(0.44)` and `sub(0.45)` biases scattered through
// that file are all sitting near this mean — so the field is renormalised back
// onto them below. That is what makes the octave count and the persistence FREE
// PARAMETERS: change the spectrum, keep the statistics, and nothing downstream
// has to be retuned.
const TARGET_MEAN = 0.4089;
const TARGET_STD = 0.1328;

// More octaves, and a flatter falloff than the classic 0.5. Both are aimed at
// one artifact: with three octaves at half amplitude the coarsest one carries
// 57% of the variance, so the field has essentially ONE feature size, and every
// carve built on it punches holes of that same size. A magnified capture of a
// whitecap showed exactly that — a raft of near-identical comma-shaped holes at
// near-identical spacing, which reads as one brush stamped repeatedly. At 0.68
// the third and fourth octaves still carry real weight, so holes come in a
// range of sizes and the raft stops looking printed.
//
// Five octaves and not six: the finest lands at 8 texels per feature on a 512
// tile, which the mip chain can still filter honestly. At six it is 4 texels and
// the bake is aliasing before the GPU ever sees it.
export function makeDetailTexture(size = 512, octaves = 5, persistence = 0.68) {
  const rand = new Float32Array(size * size);
  let seed = 1234567;
  const rng = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  for (let i = 0; i < size * size; i++) rand[i] = rng();

  const smooth = (t) => t * t * (3 - 2 * t);

  // Value-noise octave that wraps at frequency f, so the whole field tiles.
  const octave = (u, v, f) => {
    const x = u * f;
    const y = v * f;
    const xi = Math.floor(x);
    const yi = Math.floor(y);
    const uu = smooth(x - xi);
    const vv = smooth(y - yi);
    const A = (X, Y) => rand[((((Y % f) + f) % f) * size) + (((X % f) + f) % f)];
    const a = A(xi, yi);
    const b = A(xi + 1, yi);
    const c = A(xi, yi + 1);
    const d = A(xi + 1, yi + 1);
    return a * (1 - uu) * (1 - vv) + b * uu * (1 - vv) + c * (1 - uu) * vv + d * uu * vv;
  };

  const raw = (u, v) => {
    let s = 0;
    let amp = 0.5;
    let f = 4;
    for (let o = 0; o < octaves; o++) {
      s += amp * octave(u, v, f);
      amp *= persistence;
      f *= 2;
    }
    return s;
  };

  // Renormalise onto TARGET_MEAN / TARGET_STD — see above. Measured on a coarse
  // stride rather than every texel: this is estimating two moments of a smooth
  // field, and a quarter of the samples gets them to more decimal places than
  // anything downstream can tell apart, for a sixteenth of the bake cost.
  let n = 0;
  let sum = 0;
  let sum2 = 0;
  for (let y = 0; y < size; y += 4) {
    for (let x = 0; x < size; x += 4) {
      const h = raw(x / size, y / size);
      n++; sum += h; sum2 += h * h;
    }
  }
  const mean = sum / n;
  const gain = TARGET_STD / Math.sqrt(Math.max(sum2 / n - mean * mean, 1e-9));
  const fbm = (u, v) => (raw(u, v) - mean) * gain + TARGET_MEAN;

  const clamp255 = (x) => Math.max(0, Math.min(255, Math.round(x)));
  const data = new Uint8Array(size * size * 4);
  const eps = 1 / size;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const u = x / size;
      const v = y / size;
      const h = fbm(u, v);
      const hx = fbm(u + eps, v) - fbm(u - eps, v);
      const hy = fbm(u, v + eps) - fbm(u, v - eps);
      const i = (y * size + x) * 4;
      data[i] = clamp255((-hx * 3 * 0.5 + 0.5) * 255); // normal.x
      data[i + 1] = clamp255((-hy * 3 * 0.5 + 0.5) * 255); // normal.y
      data[i + 2] = clamp255(h * 255); // low-freq value
      data[i + 3] = clamp255(fbm(u * 2, v * 2) * 255); // higher-freq value
    }
  }

  const tex = new DataTexture(data, size, size, RGBAFormat, UnsignedByteType);
  tex.wrapS = RepeatWrapping;
  tex.wrapT = RepeatWrapping;
  tex.magFilter = LinearFilter;
  tex.minFilter = LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.needsUpdate = true;
  return tex;
}
