import { StorageTexture, HalfFloatType, RepeatWrapping, LinearMipmapLinearFilter } from 'three/webgpu';
import {
  Fn, instanceIndex, uint, uvec2, vec2, vec3, vec4, float, max, length, saturate,
  smoothstep, exp, attributeArray, textureStore, texture,
} from 'three/tsl';

// --- whitecap generation ---------------------------------------------------
// Three independent reads on "is this crest breaking". Each one alone is a bad
// detector; together they draw a whitecap where a whitecap belongs.
//
//   FOLD       the Jacobian of the horizontal displacement collapsing. This is
//              the classic test and the only one the file used to have. It is
//              exact but *late and thin*: it only lights up the handful of
//              texels where the surface has already passed through itself, so
//              on its own it gives a knife-line of foam at the very tip and
//              nothing else. Opening the threshold to catch more just paints
//              every mildly choppy texel instead.
//   STEEP      height gradient, taken in the displaced frame (dDy/dx is per
//              unit of *undisplaced* x, and choppiness has already squeezed
//              that by jxx). Marks a face that is standing up.
//   CONVERGE   d/dt of the horizontal area element — the divergence of the
//              surface velocity field, since v is d(displacement)/dt and the
//              area element is the Jacobian trace. Negative divergence is water
//              piling into itself, which is what a breaking crest is doing a
//              beat *before* it folds. Comes free from one stored scalar: no
//              neighbour taps, no extra pass.
// These thresholds decide WHERE a break is, and only loosely how much of the sea
// breaks. That second job deliberately belongs to the wave-group masks in
// foamShading.js instead, because the distribution of every quantity below moves
// by an order of magnitude between a light-air day and a gale: a detector tuned
// to fire on the top three percent at one wind fires on half the surface at
// another, which is how the first pass produced pack ice — 40% of the visible
// sea white, in evenly sized plates. Set here to catch a real fold on a face
// that is standing up, and left there; the ration is applied downstream.
const FOLD_OPEN = 0.68; // Jacobian where the surface starts to pinch
const FOLD_FULL = -0.10; // ...and where it has folded well through itself
const STEEP_OPEN = 0.45; // |grad h| where a face counts as standing up
const STEEP_FULL = 1.05;
const CONV_OPEN = 0.95; // 1/s of horizontal area collapse
const CONV_FULL = 3.20;
const W_FOLD = 0.72; // fold alone
const W_CONV = 0.85; // convergence, gated on a steep face
const W_BOTH = 0.85; // steep *and* folding — the actual break

// The sum above is then cut to a clean event. This is the shape that matters:
// breaking is RARE and TOTAL, not common and faint. Feeding the accumulators a
// weak continuous signal gives foam that is everywhere and never white; feeding
// them a hard, full-amplitude event on a small fraction of the surface gives
// whitecaps that are white, that persist long enough to trail, and that leave
// clean water between them.
//
// The window also sets how LONG a water particle spends inside a break, and
// that length is what the accumulators integrate. A wide window is not a softer
// look, it is a wider band of foam: at 0.18/0.56 the sum crossed the low edge a
// second or more before the crest actually folded and stayed over it a second
// after, so every crest laid down a twenty-metre stripe. Narrow and late.
const BRK_LO = 0.22;
const BRK_HI = 0.59;

// --- along-crest segmentation of the EVENT ---------------------------------
// A crest never goes white along its whole length at once. Breaking unzips in
// segments — Phillips' Lambda(c) is a distribution of breaking-crest LENGTH per
// unit area, and the segments run 1-15 m with unbroken crest between them. That
// is the difference between a whitecap and a cornice, and it has to be applied
// to the INJECTION rather than to the drawn mask: segmenting a continuous raft
// after the fact carves holes in it (a dashed cornice), whereas segmenting the
// event produces separate patches that each go on to sweep their own fully
// formed trail. foamShading.js used to do this downstream with `crestGap`; that
// is why its trails, when they cleared threshold at all, came out chopped.
//
// Sampled on the AXIS-ALIGNED fractional label coordinate, and the tile counts
// are integers, so the mask is exactly periodic on the cascade's square torus.
// This matters more than aligning it to the wave heading: a rotated UV is
// periodic on a square tile only at a measure-zero set of angles, and every
// other angle puts a hard seam in the injection field at the tile boundary — a
// straight line of foam across the sea every 1024 m and every 144 m. The
// segments come out sheared relative to the crest rather than square to it,
// which costs nothing, because the crest sweeping through the stencil is what
// makes the patch anyway.
//
// It also DRIFTS. A static label-space stencil is a permanent one: the same
// quarter of the water breaks forever and the rest never does, repeating at the
// patch size, which is visible within seconds from a fixed camera. The drift is
// along the foam heading at roughly the group velocity of the breaking band.
const EVT_LONG = 128; // m along the drift axis — long enough that an event survives the sweep
const EVT_WIDE = 32; // m across it — the fbm's coarsest octave is a quarter of the tile, so ~8 m segments
// Sigmas of the detail texture's N(0.4089, 0.1328). This pair is THE ration
// knob for the whole system: how much of the sea is foam is decided here, by a
// geometric stencil that closes on a fixed fraction whatever the waves are
// doing, and not by the detector thresholds above — whose output distribution
// moves by an order of magnitude between a light-air day and a gale, so a
// detector tuned to fire on the top 3% at one wind fires on half the surface at
// another. Measured with DEBUG=6 in foamShading.js plus tools/foamprobe.mjs:
// at +0.5/+1.3 sigma (a quarter of the surface eligible) the aerial preset came
// out at 6.7% of sea above alpha 0.3, against a literature whitecap coverage of
// 0.5-1.2% at this wind speed; at +0.84/+1.66 sigma it came out at 4.0% but the
// deck preset — a sea-level camera, which sees mostly the BACKS of crests and
// therefore residual foam rather than caps — dropped to 0.14% and read as scum.
// Settled at +0.65/+1.48, about 18% of the surface eligible. The remaining
// factor of ~4 over the field measurements is deliberate: this is an art target,
// and Kleiss & Melville's own caveat is that a single-threshold photographic W
// undercounts dim Stage B foam by close to an order of magnitude anyway.
// Opened from +0.65/+1.48 sigma, and the reason is that the RATION MOVED. This
// stencil segments at roughly 6 x 24 m, and measured with tools/foamblobs.mjs
// that left 67% of all foam on screen inside five connected components — the
// events were large enough that neighbours merged into continents. foamShading
// now carries a whitecap-scale scatter at about 3.3 m, which is where Lambda(c)
// says breaking segments actually live, so that is where the sea's foam budget
// should be spent. This term goes back to deciding only WHERE a stretch of crest
// is eligible, and hands the "how much" job downstream.
const EVT_LO = 0.560;
const EVT_HI = 0.670;
const EVT_DRIFT = 4.5; // m/s, ~c/2 for the 40-100 m band that actually folds

// Foam age, in seconds since this water particle was last inside a break. This
// is the one quantity that lets decay change SHAPE rather than opacity, and
// without it foamShading.js can only ramp alpha uniformly — which is exactly
// what "it fades arbitrarily" describes.
//
// Written explicitly rather than decoded downstream from log(lace/cap), which
// is the obvious move and is wrong four ways here: the decode rate involves
// foamDecay, which is a live GUI uniform spanning 0.02-1, so no compile-time
// constant is right; cap decays with no foamDecay divisor and hits the
// denominator floor after a few seconds, pinning the decode for all residual
// foam — precisely the population this whole pass exists to make visible; the
// channels are max()-unioned across cascades independently, so a pixel can take
// numerator and denominator from unrelated events; and both are mip-filtered,
// where the ratio of two means is not the mean of the ratio. A seconds-valued
// scalar mip-filters honestly and costs nothing: foamMap.w already existed,
// carrying `brk`, which nothing anywhere read.
//
// Re-injection resets it to zero, which is physically right — a second crest
// breaking over an old raft makes fresh foam.
const AGE_MAX = 20; // s, clamped so the accumulator cannot drift off in still water

// Foam has an age, and age is the whole difference between a thick cap and a
// lacy remnant. Three accumulators over the same injection give it for the
// price of a few multiplies. Because the maps are indexed by the *undisplaced*
// grid coordinate — which is the Lagrangian label of a water particle, not a
// fixed point in space — foam written here is already attached to the water it
// formed on and rides the orbital motion with it. The wave train moves through
// that label field at the phase speed, so the slow channels are left behind the
// crest on their own. That is where the trails come from; foamShading then
// dilates them along the heading to turn a hairline into a ribbon.
//
// Two things were wrong in the first pass and both made the sea read as poured
// milk rather than foam:
//
//   RATE  the injection was max(decayed, brk), so a single frame in which a
//         texel broke slammed it to full opacity. Foam has no interior
//         structure at all when it is written that way — every broken texel is
//         exactly 1.0. It is now an additive rate, so a texel has to spend real
//         time inside a break to go fully white and the edges of the event come
//         out partial. That partial edge is most of what makes it read as a
//         bubble raft.
//   TAU   the tails were 7.5 s and 22.5 s of effective decay on a periodic
//         patch a couple of hundred metres across. Over 22 s essentially every
//         texel in the tile has been under a breaking crest at some point, so
//         the lace channel sat near 1.0 everywhere and the ocean was a blanket.
//   AMPLITUDE  the three channels were all driven to saturation. A water
//         particle spends roughly a third of a second inside a break, and the
//         old rates deposited 2.4 / 1.0 / 0.35 of accumulator in that time
//         against a ceiling of 1 — so all three pinned at 1.0 together and the
//         ONLY thing distinguishing a thick cap from a week-old remnant was how
//         long ago it happened. Foam is not like that: a fresh cap is opaque, an
//         old raft is a thin veil you can see the sea through, and that
//         difference is an AMPLITUDE, not just a lifetime.
//
// Read each pair as (how much one ~0.3 s break deposits) x (how long it lasts).
// Cap deposits about 0.95 and is gone inside two seconds: opaque, and only
// while the crest is actually breaking. Trail deposits about 0.45 and survives
// a second and a half, which at the ~8 m/s the crest overtakes a water particle
// is a ten-metre ribbon behind the break — half-opaque, so it reads as foam ON
// water rather than as a second coat of paint. Lace deposits under 0.2 and
// lingers four seconds, a faint web spread over most of a wavelength.
const INJ_CAP = 3.2; // 1/s of accumulation at full break
const INJ_TRAIL = 1.45;
const INJ_LACE = 0.60;
const TAU_CAP = 1.00; // s — thick foam right on the break
// 1.05, not 0.62. Callaghan et al. 2012 measured the area-weighted effective
// decay of 552 individual whitecaps at 1.4-4.8 s; the old value sat at the very
// bottom of that band, and the band length a trail can draw is c*tau — at the
// 8-12 m/s phase speed of the 40-100 m waves that actually fold, 1.6 s of
// effective decay is a 14 m ribbon and 2.6 s is a 23 m one. This is a move
// within the measured range, not a correction of an error.
const TAU_TRAIL = 1.05; // s / foamDecay (~2.6 s) — dragged down the face and behind it
// 3.9 s, not the 22 s this started at. The chop cascade's patch is 144 m and
// periodic: over a long enough tail every texel in it has been under a breaking
// crest at some point, the lace channel converges on 1 everywhere, and the whole
// foreground comes out as one flat veil instead of as separate dissipating
// patches. What keeps it honest at this length is the low injection above — the
// channel is allowed to be old, it is not allowed to be opaque.
// 2.0 s (5.0 s effective), and deliberately NOT the 10 s that the measured
// per-event range (0.2-10.4 s) would permit. At 10 s the lace band is c*tau =
// 90-120 m with a tens-of-seconds lifetime, and a 100 m ribbon of foam that
// lives that long, elongated along the wind, is a Langmuir windrow — a real
// phenomenon that this sea state cannot produce, because windrows assemble over
// 60-300 s of crosswind sweeping and whitecap foam is gone in ten. Reaching it
// by accident, out of a decay constant, is the one way this file can draw
// something that is both convincing and physically impossible.
const TAU_LACE = 2.00; // s / foamDecay (~5.0 s) — the old dissipating web in the troughs

// What one break actually deposits in each accumulator, which is the number
// foamShading.js has to divide by if a single threshold is to mean the same
// thing to all three channels. Exported rather than restated over there: these
// are a function of the injection rates and taus above, they move whenever
// those move, and a stale copy in the other file is invisible until the foam
// silently rations itself wrong.
//
// deposit = INJ * tau * (1 - exp(-dwell/tau)), the closed form of integrating a
// constant injection against exponential decay for the time a water particle
// spends inside a break.
const DWELL = 0.30; // s inside a break, measured off the detector's own window
const FOAM_DECAY_REF = 0.4; // params.foamDecay's default; the GUI can move it live
const deposit = (inj, tau) => inj * tau * (1 - Math.exp(-DWELL / tau));
export const FOAM_PEAK = {
  cap: deposit(INJ_CAP, TAU_CAP),
  trail: deposit(INJ_TRAIL, TAU_TRAIL / FOAM_DECAY_REF),
  lace: deposit(INJ_LACE, TAU_LACE / FOAM_DECAY_REF),
};

// rgba16f storage texture: filterable (bilinear) AND storage-capable, tiling
// via RepeatWrapping.
//
// `mips` is load-bearing for the foam map. A mip chain is already being built
// for every storage texture here — Textures.needsMipmaps() is just
// generateMipmaps (true by default) and the binding is flagged needsMipmap on
// every compute write — but StorageTexture's default minFilter is LinearFilter,
// which never samples it. Left like that, a pixel covering fifty foam texels
// picks one at random, and a threshold on random is a speckled band across the
// horizon. Switching the min filter costs nothing and makes distant foam
// converge on its own local mean, which is exactly what distant foam looks like.
// The other two maps keep the flat filter: oceanSurfaceMaterial band-limits the
// derivative map itself and reads displacement at an explicit level(0).
function mapTexture(N, mips = false) {
  const tex = new StorageTexture(N, N);
  tex.type = HalfFloatType;
  tex.wrapS = RepeatWrapping;
  tex.wrapT = RepeatWrapping;
  if (mips) tex.minFilter = LinearMipmapLinearFilter;
  return tex;
}

// Assemble pass: pack a cascade's spatial IFFT buffers into sampled maps and
// accumulate foam.
//   displacement = ( lambda*Dx, height, lambda*Dz, surface divergence, 0.5 = neutral )
//   derivatives  = ( dDy/dx, dDy/dz, lambda*dDx/dx, lambda*dDz/dz )
//   foamMap      = ( cap, trail, lace, seconds since this water last broke )
//
// prevDisp still carries last frame's displacement alongside the Jacobian
// trace, but only the trace is read (the convergence term below). Differencing
// the displacement gives the true Lagrangian particle velocity — no
// approximation — and there was a velocity map exported for the spray emitter;
// with spray gone nothing consumed it, so it is not written. Restore the store
// if anything needs surface velocity again.
//
// foamMap is written here and read by foamShading.js, which owns how it is
// drawn. It gets its own texture, hung off the cascade object, rather than
// being squeezed into spare channels of the other two maps: those keep the
// channel contracts other files depend on, and only this one needs the mip
// filtering (which would change how oceanSurfaceMaterial's own band-limiting
// behaves if it were switched on for the derivative map). The two spare .w
// slots — displacement.w and foamMap.w — used to hold a duplicate of the trail
// channel and the raw break rate, and nothing anywhere read either one. They
// now carry the surface divergence and the foam age, which are the two scalars
// foamShading needs and cannot reconstruct downstream.
export function createCascadeMaps(cascade, {
  N, lambda, dt, foamDecay, detailTex, time, heading,
}) {
  const displacement = mapTexture(N);
  const derivatives = mapTexture(N);
  const foamMap = mapTexture(N, true);
  cascade.foamMap = foamMap; // consumed by foamShading.js

  const foam = attributeArray(N * N, 'vec4'); // (cap, trail, lace, age); zeroed = clean water, age 0
  const prevDisp = attributeArray(N * N, 'vec4'); // (last frame's displacement, last frame's Jacobian trace)

  // Integer tile counts, so the event stencil wraps exactly — see EVT_LONG.
  // The realised tile is L/n metres, which quantises the target: 128/32 m on
  // the 1024 m cascade, 144/28.8 m on the 144 m one.
  const L = cascade.lengthScale;
  const nLong = Math.max(1, Math.round(L / EVT_LONG));
  const nWide = Math.max(1, Math.round(L / EVT_WIDE));

  const assemble = Fn(() => {
    const id = instanceIndex;
    const coord = uvec2(id.mod(uint(N)), id.div(uint(N)));
    const DxDz = cascade.DxDz.element(id); // (Dx, Dz)
    const DyDxz = cascade.DyDxz.element(id); // (height, dDz/dx)
    const DyxDyz = cascade.DyxDyz.element(id); // (dDy/dx, dDy/dz)
    const DxxDzz = cascade.DxxDzz.element(id); // (dDx/dx, dDz/dz)

    const jxx = float(1).add(lambda.mul(DxxDzz.x)).toVar();
    const jzz = float(1).add(lambda.mul(DxxDzz.y)).toVar();
    const jxz = lambda.mul(DyDxz.y); // lambda * dDz/dx (= dDx/dz by symmetry)
    const J = jxx.mul(jzz).sub(jxz.mul(jxz));
    const trace = jxx.add(jzz).toVar(); // area element, differenced below

    // toVar() is load-bearing throughout this kernel: TSL inlines expressions at
    // their use site, so a read of prevDisp written as an expression would land
    // *after* the assign below and come out reading this frame's own value.
    const prev = prevDisp.element(id).toVar();

    const fold = saturate(float(FOLD_OPEN).sub(J).mul(1 / (FOLD_OPEN - FOLD_FULL))).toVar();
    const slope = length(vec2(
      DyxDyz.x.div(max(jxx, float(0.25))),
      DyxDyz.y.div(max(jzz, float(0.25))),
    ));
    const steep = saturate(slope.sub(STEEP_OPEN).mul(1 / (STEEP_FULL - STEEP_OPEN))).toVar();
    // prev.w is 0 on the first frame and trace is ~2, so this reads as a strong
    // *divergence* and clamps to zero — no frame-0 sheet of foam.
    // Signed rate of horizontal area collapse, 1/s. Positive is water piling
    // into itself. Used twice: thresholded into the break detector below, and
    // stored raw-ish in displacement.w for foamShading to thin residual foam on
    // the diverging half of the orbit — see the store at the end.
    const divRate = prev.w.sub(trace).div(max(dt, float(1e-4))).toVar();
    const converge = saturate(
      divRate.sub(CONV_OPEN).mul(1 / (CONV_FULL - CONV_OPEN)),
    ).toVar();

    const brk = smoothstep(float(BRK_LO), float(BRK_HI), saturate(
      fold.mul(W_FOLD)
        .add(converge.mul(steep).mul(W_CONV))
        .add(fold.mul(steep).mul(W_BOTH)),
    )).toVar();

    // Where breaking is ALLOWED, on the drifting label-space stencil — see
    // EVT_LONG. level(0) is not optional: a compute kernel has no derivatives,
    // so there is no implicit mip to select.
    const fx = float(coord.x).div(float(N)).toVar();
    const fz = float(coord.y).div(float(N)).toVar();
    const dr = time.mul(EVT_DRIFT).toVar();
    const evt = texture(detailTex, vec2(
      fx.mul(nLong).add(dr.mul(heading.x).mul(nLong / L)),
      fz.mul(nWide).add(dr.mul(heading.y).mul(nWide / L)),
    )).level(0).b.toVar();
    const gate = smoothstep(float(EVT_LO), float(EVT_HI), evt).toVar();

    const acc = foam.element(id).toVar();
    const inj = brk.mul(gate).mul(dt).toVar();
    const cap = saturate(acc.x.mul(exp(dt.mul(-1 / TAU_CAP))).add(inj.mul(INJ_CAP))).toVar();
    const trail = saturate(
      acc.y.mul(exp(dt.mul(foamDecay).mul(-1 / TAU_TRAIL))).add(inj.mul(INJ_TRAIL)),
    ).toVar();
    const lace = saturate(
      acc.z.mul(exp(dt.mul(foamDecay).mul(-1 / TAU_LACE))).add(inj.mul(INJ_LACE)),
    ).toVar();
    // Age: zeroed while breaking, integrating dt once the break has passed.
    // smoothstep rather than a step so a texel on the edge of an event does not
    // flicker its age between 0 and whatever it had accumulated.
    const fresh = smoothstep(float(0.25), float(0.45), inj.div(max(dt, float(1e-4)))).toVar();
    const age = acc.w.add(dt).mul(float(1).sub(fresh)).min(float(AGE_MAX)).toVar();
    foam.element(id).assign(vec4(cap, trail, lace, age));

    const disp = vec3(DxDz.x.mul(lambda), DyDxz.x, DxDz.y.mul(lambda)).toVar();
    prevDisp.element(id).assign(vec4(disp, trace));

    // displacement.w carried a duplicate of the trail channel that nothing read.
    // It now carries the surface divergence, remapped so 0.5 is neutral: for a
    // linear wave du/dx = -a*omega*k*sin(kx - wt), convergent on the forward
    // face and divergent on the rear, and with this project's measured RMS
    // slopes and lambda = 2.2 the area modulation is about +/-20%. That is why
    // real residual foam looks braided rather than evenly spread — it is
    // squeezed thicker and stretched thinner once per wave.
    const divR = saturate(divRate.mul(0.35).add(0.5)).toVar();
    textureStore(displacement, coord, vec4(disp, divR)).toWriteOnly();
    textureStore(derivatives, coord, vec4(DyxDyz.x, DyxDyz.y, DxxDzz.x.mul(lambda), DxxDzz.y.mul(lambda))).toWriteOnly();
    textureStore(foamMap, coord, vec4(cap, trail, lace, age)).toWriteOnly();
  })().compute(N * N);

  return { displacement, derivatives, foamMap, foam, assemble };
}
