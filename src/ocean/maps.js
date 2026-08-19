import { StorageTexture, HalfFloatType, RepeatWrapping, LinearMipmapLinearFilter } from 'three/webgpu';
import { foamLattice } from './params.js';
import {
  Fn, instanceIndex, uint, uvec2, vec2, vec3, vec4, float, max, length, saturate,
  smoothstep, exp, mix, attributeArray, textureStore, texture, floor, fract, pow, sin,
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
// ...and the whole sum is then gated on an ACTUAL CURL. The three detectors
// shape intensity, but each can fire without the surface visibly overturning:
// convergence leads the fold by a beat (by design), the steep path marks
// faces that never curl, and FOLD itself opens at J = 0.68 — a pinch, not a
// lip. The curl has an exact meaning in this displacement model: the Jacobian
// collapsing through zero is the surface folding over itself. So injection is
// permitted only where J has fallen into that collapse, and everything the
// eye reads as foam grows outward from a real curl via the diffusion, bloom,
// trail and remnant machinery. Side effect, deliberate: the chop cascade
// barely ever truly curls (its lambda is rolled off), so chop speckle dies.
const CURL_OPEN = 0.30; // J below which a lip is genuinely forming
const CURL_FULL = 0.02; // ...and where the overturn is essentially complete

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
// NARROW, for the same reason the event gate is — and this is the other half of
// "rare and total". At 0.22/0.59 the window spans most of the detector's usable
// range, so a texel whose fold and steepness are middling returns brk ~ 0.15 and
// deposits fifteen percent of a whitecap. There is no such event. Water either
// entrains air or it does not, and the sum crossing a line is the best proxy
// this detector has for which. Measured before narrowing: 15.4% of the sea
// carried cap above 0.05 and 0.37% above 0.30, i.e. 97.6% of all the foam being
// generated was a smear too faint to draw and too widespread to threshold
// cleanly. That is the speckle, and it is made here rather than in the shading.
const BRK_LO = 0.44;
const BRK_HI = 0.52;

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
// Now measured IN THE WAVE FRAME rather than on the world axes — see the
// rotated sampling in the kernel. A whitecap is a segment of a breaking crest,
// so the eligible region should be a band lying ALONG the crest: narrow across
// the direction of travel, long down the crest line. Feature size is 0.19 x
// tile, so these draw roughly 3.6 m across and 14 m along — inside the 1-15 m
// Lambda(c) breaking-segment range, and shaped like a whitecap rather than like
// a blob that happens to overlap one.
const EVT_ACROSS_CREST = 19; // m of tile along the heading
const EVT_ALONG_CREST = 75; // m of tile along the crest line
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
// NARROW, and that matters far more than where it sits. This is a gate on
// whether a stretch of water is allowed to break, so its output should be 0 or
// 1 — a partial value does not mean "half a whitecap", it means a break that
// deposits half the foam a break deposits, which is a thing that does not
// happen. Every version of this until now used a window about 0.8 sigma wide,
// so a large fraction of the surface sat at an intermediate gate and got a
// whisper of foam. Measured on the raw accumulators with DEBUG=1: 15.4% of the
// sea carried cap above 0.05 while only 0.37% carried it above 0.30, against a
// full break depositing 0.83. Common and faint — the exact failure the comment
// at the top of this block exists to prevent — and a weak wash sitting near the
// draw threshold is what a carve downstream dices into speckle.
//
// The eligibility tap is HEX-TILED (see hexTap in the kernel): sampled plain,
// the stencil repeats on an exact ~19 x 75 m lattice inside the cascade tile,
// and from altitude the caps form marching rows — the single most artificial-
// looking pattern an aerial framing shows. Stochastic tiling keeps the seam-
// free wrap on the tile torus (offsets are hashed from lattice cells of the
// stencil's own uv space, which wraps at integers) while making event sites
// aperiodic WITHIN the tile. The 1024 m macro-period of the wave field itself
// is intrinsic to FFT synthesis and is not this gate's problem to solve.
// The centre sets the ration; the width sets whether an event is an event.
// The centre is no longer a compile-time constant — it arrives as the
// evtLo/evtHi uniforms from foamSeaState() in params.js, which converts the
// wind speed into an eligibility fraction through the Monahan coverage law.
// The 0.034 width is preserved there, because "an event is an event" is a
// property of the detector, not of the sea state.
// Lateral diffusion of the foam accumulators, per 1/60 s — see the spreading
// block in the kernel. At 0.16 a hairline reaches about three texels, i.e.
// twelve metres, over the cap's one-second life, which is a whitecap rather
// than a knife line and is wide enough to survive the mip chain. Stable well
// below 0.25.
const FOAM_SPREAD = 0.16;
// ...and a slower one for the aeration channel: a bubble plume genuinely
// spreads laterally as it rises and shears, and it lives ~60 s, so even a
// small per-frame coefficient integrates to tens of metres of spread — which
// is what the reference's trough-filling clouds are. Mass-conserving, so the
// spread pays for itself in peak amplitude; the injection carries that.
const AER_SPREAD = 0.035;

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
const INJ_CAP = 7.5; // 1/s at full break — see the lateral spreading block,
// which costs peak amplitude in exchange for a cap that is a patch not a line
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
// per-event range (0.2-10.4 s) would permit: stretching ONE exponential until
// it draws windrows produces windrows with the wrong assembly physics. The
// long tail now exists as its own channel instead — see REMNANT below — with
// the convergence-lane decay modulation that real windrow assembly is.
const TAU_LACE = 2.00; // s / foamDecay (~5.0 s) — the old dissipating web in the troughs

// --- the second exponential: remnant and aeration ---------------------------
// Callaghan 2012/2013 measure whitecap decay as a DOUBLE exponential: a 1.4-
// 4.8 s bulk stage (cap/trail/lace above) and a surfactant-stabilised remnant
// at tens of seconds. That remnant is 80-90% of the foam AREA on a real sea,
// and it is the population the reference photo's lace web and wind streaks are
// made of. Two more accumulators over the same injection:
//
//   REMNANT  long-lived surface lace. What keeps a 30-60 s lifetime from
//            converging to a blanket on a periodic tile is not the tau — it is
//            that the DECAY IS SPATIAL: fast in divergence lanes, slow in
//            convergence lanes (the divR term) times a wind-elongated survival
//            field (the lane tap in the kernel). Old foam then self-organises
//            into downwind streaks, which is literally what Langmuir
//            circulation does to it: convergence lanes collect foam,
//            divergence lanes sweep clean, 0-20 degrees off the wind.
//   AERATION the subsurface bubble plume. The LONGEST-lived stage of the whole
//            lifecycle — deep plume layers stay optically visible for >50 wave
//            periods, and sub-Hinze microbubbles linger for minutes — and the
//            single most prominent foam feature in the reference: tens-of-
//            metre milky turquoise patches filling the near-field troughs.
//            oceanSurfaceMaterial.js reads it for the plume term; it never
//            draws as white foam itself.
//
// Both injections are scaled by the residScale uniform (foamSeaState): the
// budget a rising wind adds is almost entirely THIS population.
const INJ_REM = 0.40;
const INJ_AER = 0.09;
// Off-lane decay runs this many times faster than in-lane, which is the 10:1
// wind-elongated survival contrast that assembles streaks out of a uniform
// deposit. divR modulates on top: convergent water holds its foam.
const REM_FAST = 7.0;
const AER_FAST = 3.5;
// —- onset bloom ——————————————————————————————-
// A whitecap does not print at full deposit on its first frame. INJ_CAP fills
// the accumulator past the draw threshold in ~3 frames (~50 ms), so a cap used
// to materialise fully formed — a pop. Real entrainment BLOOMS: the
// whitewater front expands from first touchdown over a third of a second. So
// each texel carries an onset clock (foam2.z) that ramps up while an event is
// on it and dies with the cap once it passes, and the cap's injection is
// scaled by that envelope. Because the clock is per-texel and the crest
// sweeps through the label field, the patch also GROWS spatially from the
// event core outward instead of switching on as a block.
//
// The floor keeps the first frames from being invisible (the seed the bloom
// grows from), and it is why INJ_CAP needs no compensation: dwell-integrated
// deposit drops only ~40%, which the saturate() ceiling was clipping off
// mature caps anyway.
const BLOOM_T = 0.45; // s from first break to full injection, at the median rate
const BLOOM_FLOOR = 0.12; // envelope at the instant of onset
// —- foam slip: how foam moves RELATIVE to the water ————————————-
// The maps are Lagrangian, so foam already rides the orbital motion for free.
// What a label-space state cannot do on its own is move foam ACROSS the water
// — and real foam does, three ways, which together are what make rafts
// split, collect and surge instead of sitting where they were born:
//
//   WINDAGE    the constant downwind skate (advDrift, ~3.5% of U10). Uniform,
//              so alone it translates the whole field and separates nothing.
//   DRAINAGE   foam slides DOWNSLOPE. As a wave runs through a raft the slip
//              first points backward off the advancing face, then forward off
//              the back — an oscillating smear of several metres per period
//              that bunches foam into troughs and convergence lines (merging)
//              and rakes it thin over crests (separating). -grad(h) per texel.
//   SURGE      while a texel is actually breaking, its whitewater is carried
//              forward with the crest at a large fraction of the phase speed
//              (Kleiss & Melville: 0.3-0.8 of c for most of a period). This is
//              what detaches a fresh cap from its birth label and drives it
//              down the face.
//
// All three feed the semi-Lagrangian backtrace as one per-texel velocity, so
// divergent slip genuinely tears accumulated foam apart and convergent slip
// genuinely collects it — the dynamics are the transport, not a decal.
const SLOPE_SLIP = 2.2; // m/s of drainage per unit of surface slope
const SURGE = 3.5; // m/s of forward carry while breaking, along the heading
// Per-texel spread of the bloom RATE, read off the stencil tap's coarse
// cellular channel. Without it every texel of an event blooms in lockstep and
// the patch still arrives as one formed unit, only slower; with a ~4:1 rate
// spread at whitecap scale the fast cells surface first and the froth EXPANDS
// from seeds outward, which is what a breaking crest actually does.
// Nominal in-lane taus for the FOAM_PEAK export below; the live values are the
// remTau/aerTau uniforms.
const REM_TAU_REF = 30;
// Lane geometry: windrow lanes are long ALONG the wind and spaced tens of
// metres ACROSS it. Sampled on a second, deliberately COARSER integer lattice
// (maxDen 4) than the event stencil: at lattice magnitude ~3.6 one period
// along the heading is ~280 m on the 1024 m cascade, so the survival lanes are
// genuinely long features, which the stencil's magnitude-14 lattice cannot be.
const LANE_ACROSS = 30; // m between lanes, across the wind
const LANE_DRIFT = 0.5; // m/s the lane pattern crawls downwind

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
  // Nominal, at the reference tau and unit residScale — the norm downstream
  // only needs a stable "what one break deposits" yardstick, and the wind
  // scaling is MEANT to push the mask past 1 in a hard sea.
  rem: deposit(INJ_REM, REM_TAU_REF),
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
  N, lambda, dt, foamDecay, detailTex, time, heading, headingVec,
  foamSea, evtDrift = 4.5, advDrift = 0.5,
}) {
  const displacement = mapTexture(N);
  const derivatives = mapTexture(N);
  const foamMap = mapTexture(N, true);
  cascade.foamMap = foamMap; // consumed by foamShading.js
  // The long channels: (remnant, aeration, lane, 0). Its own texture rather
  // than spare channels because it needs the same mip filtering as foamMap and
  // both consumers (foamShading's windrow lace, oceanSurfaceMaterial's plume)
  // read it at coarse mips.
  const foamMap2 = mapTexture(N, true);
  cascade.foamMap2 = foamMap2;

  const foam = attributeArray(N * N, 'vec4'); // (cap, trail, lace, age); zeroed = clean water, age 0
  const foam2 = attributeArray(N * N, 'vec4'); // (remnant, aeration, 0, 0)
  const prevDisp = attributeArray(N * N, 'vec4'); // (last frame's displacement, last frame's Jacobian trace)

  // Integer tile counts, so the event stencil wraps exactly — see EVT_LONG.
  // The realised tile is L/n metres, which quantises the target: 128/32 m on
  // the 1024 m cascade, 144/28.8 m on the 144 m one.
  const L = cascade.lengthScale;
  // Integer lattice pointing along the wave heading — see foamLattice(). This is
  // what lets the stencil be laid out ALONG THE CREST instead of along the world
  // axes, without a tile seam. Baked at construction, like the tile counts it
  // replaces; a runtime wind change leaves it stale until the cascade is rebuilt.
  const lat = foamLattice(headingVec[0], headingVec[1]);
  // One period along an axis is L / (k * |lattice|) metres, so k is chosen from
  // the target feature size the same way the old axis-aligned counts were.
  const kAcross = Math.max(1, Math.round(L / (lat.mag * EVT_ACROSS_CREST)));
  const kAlong = Math.max(1, Math.round(L / (lat.mag * EVT_ALONG_CREST)));
  // The windrow lane lattice — same trick at the COARSEST usable rational
  // approximation (maxDen 2, magnitude ~1.4), because a lane has to be a LONG
  // feature: one period along this lattice is ~700 m on the 1024 m cascade,
  // which delivers ~140 m survival lanes — real windrow scale — where the
  // magnitude-14 stencil lattice tops out at 73 m periods. The ~9 degrees of
  // angular error is inside the 0-20 degrees real windrows sit off the wind.
  const lane = foamLattice(headingVec[0], headingVec[1], 2, (2 * Math.PI) / 180);
  const kLaneAcross = Math.max(1, Math.round(L / (lane.mag * LANE_ACROSS)));
  // Advection: metres per frame of windage, in texels of this cascade.
  const texelM = L / N;

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
    // Remapped so 0.5 is neutral — used by the remnant/aeration decay below
    // and stored in displacement.w for foamShading's braiding term.
    const divR = saturate(divRate.mul(0.35).add(0.5)).toVar();

    const curl = float(1).sub(smoothstep(float(CURL_FULL), float(CURL_OPEN), J)).toVar();
    const brk = smoothstep(float(BRK_LO), float(BRK_HI), saturate(
      fold.mul(W_FOLD)
        .add(converge.mul(steep).mul(W_CONV))
        .add(fold.mul(steep).mul(W_BOTH)),
    )).mul(curl).toVar();

    // Where breaking is ALLOWED, on the drifting label-space stencil — see
    // EVT_LONG. level(0) is not optional: a compute kernel has no derivatives,
    // so there is no implicit mip to select.
    const fx = float(coord.x).div(float(N)).toVar();
    const fz = float(coord.y).div(float(N)).toVar();
    const dr = time.mul(evtDrift).toVar();
    // Rotated into the wave frame, on an integer lattice so it still wraps
    // exactly. `u` runs along the heading (the narrow axis — this is what
    // segments a crest), `v` runs along the crest line (the long axis, so an
    // eligible stretch is a band lying along the crest rather than a blob
    // crossing several of them). The drift is along the heading only.
    const u = fx.mul(lat.q).add(fz.mul(lat.p)).toVar();
    const v = fx.mul(-lat.p).add(fz.mul(lat.q)).toVar();
    // Per-strip drift-phase jitter. The stencil is an exactly-periodic lattice
    // drifting at one speed, and the horizon showed it: a single row of evenly
    // spaced caps. Offsetting each along-crest strip's phase by a (periodic —
    // the multiplier is the integer kAlong) tap of the same field decorrelates
    // the rows without touching the ration or the seam-free wrap.
    const jit = texture(detailTex, vec2(v.mul(kAlong), float(0.71))).level(0).b.toVar();
    // Heitz-style hex tile-and-blend, compute flavour (explicit level 0) —
    // same construction as the shader's, so the field statistics the EVT gate
    // was calibrated against are preserved. See the note above EVT_LO.
    const hash2 = (hv) => fract(sin(vec2(
      hv.x.mul(127.1).add(hv.y.mul(311.7)),
      hv.x.mul(269.5).add(hv.y.mul(183.3)),
    )).mul(43758.5453));
    const hexTap = (huv) => {
      const hg = huv.mul(1.8);
      const sk = vec2(hg.x.sub(hg.y.mul(0.57735)), hg.y.mul(1.15470)).toVar();
      const hb = floor(sk).toVar();
      const hf = fract(sk).toVar();
      const up = smoothstep(float(0.9999), float(1.0001), hf.x.add(hf.y)).toVar();
      const h0 = hb.add(mix(vec2(0, 0), vec2(1, 1), up)).toVar();
      const h1 = hb.add(vec2(1, 0)).toVar();
      const h2 = hb.add(vec2(0, 1)).toVar();
      const w0 = mix(float(1).sub(hf.x).sub(hf.y), hf.x.add(hf.y).sub(1), up).toVar();
      const w1 = mix(hf.x, float(1).sub(hf.y), up).toVar();
      const w2 = mix(hf.y, float(1).sub(hf.x), up).toVar();
      const M = float(0.4089);
      const t0 = texture(detailTex, huv.add(hash2(h0))).level(0).sub(M);
      const t1 = texture(detailTex, huv.add(hash2(h1))).level(0).sub(M);
      const t2 = texture(detailTex, huv.add(hash2(h2))).level(0).sub(M);
      const nrm = max(float(1e-4),
        w0.mul(w0).add(w1.mul(w1)).add(w2.mul(w2))).sqrt();
      return t0.mul(w0).add(t1.mul(w1)).add(t2.mul(w2)).div(nrm).add(M).toVar();
    };
    const evtT = hexTap(vec2(
      u.mul(kAcross).add(dr.mul(lat.mag * kAcross / L)).add(jit.sub(0.41).mul(0.9)),
      v.mul(kAlong),
    ));
    const evt = evtT.b.toVar();
    const gate = smoothstep(foamSea.evtLo, foamSea.evtHi, evt).toVar();
    // The windrow survival lanes — see LANE_ACROSS. Long along the wind (one
    // period of the coarse lattice, ~250-350 m, features ~50 m), tens of
    // metres apart across it, crawling downwind. This field never gates
    // injection; it modulates the DECAY of the two long channels, which is how
    // a uniform deposit self-organises into streaks.
    const u2 = fx.mul(lane.q).add(fz.mul(lane.p)).toVar();
    const v2 = fx.mul(-lane.p).add(fz.mul(lane.q)).toVar();
    const laneN = hexTap(vec2(
      u2.add(time.mul(LANE_DRIFT * lane.mag / L)),
      v2.mul(kLaneAcross),
    )).b.toVar();
    const laneS = smoothstep(float(0.40), float(0.52), laneN).toVar();

    // --- lateral spreading ---------------------------------------------------
    // A break entrains air over a PATCH. This detector finds a hairline: `fold`
    // is the Jacobian passing through zero, which happens on the one or two
    // texels where the surface has actually crossed itself, and the file has
    // said so from the beginning — "a fold is a hairline, and one hairline per
    // crest is not a whitecap". The compensation until now was a directed
    // dilation in the SHADER, which widens what is drawn but cannot widen what
    // is stored.
    //
    // That distinction turned out to be the whole problem. Cascade 0's texel is
    // 4 m, so a one-texel cap is a delta function on the map, and every read of
    // it downstream is mip-filtered by the pixel footprint: a sharp 1-texel
    // event averaged over a several-metre footprint comes back as a faint wide
    // smear. Measured on the raw accumulators, 15% of the sea carried cap above
    // 0.05 while 0.4% carried it above 0.30. A weak wash over a wide area,
    // sitting near the draw threshold, is exactly what the carves downstream
    // dice into speckle — and no amount of work on the carves can fix a signal
    // that is faint because it was band-limited away.
    //
    // Diffusing the accumulator is both the cheap fix and the physically honest
    // one: whitewater spreads laterally from where it was entrained. Four
    // neighbour reads on a buffer already in registers, with wraparound, which
    // the periodic tile makes exact.
    //
    // ponytail: reads neighbours from the buffer this pass also writes, so a
    // neighbour may be pre- or post-update depending on scheduling. That makes
    // this Gauss-Seidel rather than Jacobi — order-dependent by at most one
    // frame of diffusion, which is invisible at this coefficient and stable well
    // below 0.25. Ping-pong buffers if it ever needs to be exact.
    const row = coord.y.mul(uint(N));
    const rowU = coord.y.add(uint(N - 1)).mod(uint(N)).mul(uint(N));
    const rowD = coord.y.add(uint(1)).mod(uint(N)).mul(uint(N));
    const cxL = coord.x.add(uint(N - 1)).mod(uint(N));
    const cxR = coord.x.add(uint(1)).mod(uint(N));
    const nb = foam.element(row.add(cxL))
      .add(foam.element(row.add(cxR)))
      .add(foam.element(rowU.add(coord.x)))
      .add(foam.element(rowD.add(coord.x)))
      .mul(0.25).toVar();
    // ...and the aeration channel's own, slower spread — see AER_SPREAD.
    const nb2 = foam2.element(row.add(cxL)).y
      .add(foam2.element(row.add(cxR)).y)
      .add(foam2.element(rowU.add(coord.x)).y)
      .add(foam2.element(rowD.add(coord.x)).y)
      .mul(0.25).toVar();
    // --- transport -----------------------------------------------------------
    // Semi-Lagrangian backtrace (the Crest UpdateFoam pattern): the state is
    // read from one frame of WINDAGE upwind, so accumulated foam drifts
    // downwind through the label field instead of being nailed forever to the
    // water particle it formed on. Windage is real — foam sits on the surface
    // and the wind pushes it at ~3.5% of U10 relative to the water — and it is
    // half of what lets the long channels organise into streaks rather than a
    // blanket (the lane-modulated decay below is the other half).
    //
    // Manual bilinear on the state buffers, exact on the periodic tile. Same
    // in-place Gauss-Seidel caveat as the diffusion above: the offset stays a
    // few hundredths of a texel per frame (max slip ~7 m/s against a 4 m
    // texel at 60 fps), so a neighbour being pre- or post-update is at most
    // one frame of extra smear along the slip axis.
    //
    // The slip field — see SLOPE_SLIP. Windage is constant, drainage is
    // -grad(h) (undisplaced frame, which is the frame this state lives in),
    // and the surge rides the break detector so only actively-breaking texels
    // surge. Per-texel and divergent, which is the whole point.
    const brkG = brk.mul(gate).toVar();
    const slipX = float(headingVec[0] * advDrift)
      .sub(DyxDyz.x.mul(SLOPE_SLIP))
      .add(brkG.mul(SURGE * headingVec[0])).toVar();
    const slipZ = float(headingVec[1] * advDrift)
      .sub(DyxDyz.y.mul(SLOPE_SLIP))
      .add(brkG.mul(SURGE * headingVec[1])).toVar();
    const px = float(coord.x).sub(slipX.mul(dt).mul(1 / texelM)).add(float(N)).toVar();
    const py = float(coord.y).sub(slipZ.mul(dt).mul(1 / texelM)).add(float(N)).toVar();
    const ix0 = uint(floor(px)).mod(uint(N)).toVar();
    const iy0 = uint(floor(py)).mod(uint(N)).toVar();
    const ix1 = ix0.add(uint(1)).mod(uint(N)).toVar();
    const iy1 = iy0.add(uint(1)).mod(uint(N)).toVar();
    const bx = fract(px).toVar();
    const by = fract(py).toVar();
    const i00 = iy0.mul(uint(N)).add(ix0).toVar();
    const i01 = iy0.mul(uint(N)).add(ix1).toVar();
    const i10 = iy1.mul(uint(N)).add(ix0).toVar();
    const i11 = iy1.mul(uint(N)).add(ix1).toVar();
    const self = mix(
      mix(foam.element(i00), foam.element(i01), bx),
      mix(foam.element(i10), foam.element(i11), bx), by,
    ).toVar();
    const self2 = mix(
      mix(foam2.element(i00), foam2.element(i01), bx),
      mix(foam2.element(i10), foam2.element(i11), bx), by,
    ).toVar();
    // Age is NOT diffused — it is a per-particle clock, and averaging it with
    // the neighbours would make the age of foam depend on how long ago the water
    // NEXT to it broke.
    // CAP ONLY. Diffusion conserves mass, so spreading a channel costs it peak
    // amplitude, and the aged channels cannot afford that: they live for seconds,
    // so they diffuse for seconds, and measured with the spread applied to all
    // three the lace channel lost its entire population above 0.30. They also do
    // not need it — the wave train sweeping through the label field already
    // gives them a band tens of metres wide. It is only the cap that is born as
    // a hairline, and only fresh whitewater that actually spreads under its own
    // turbulence. INJ_CAP carries the compensation for what the diffusion costs.
    // .min(0.4): the Gauss-Seidel mix destabilises past ~0.5, and dt is only
    // clamped at 0.1 s -- 0.1 s x timeScale 3 would have reached 1.0.
    const spread = saturate(dt.mul(60).mul(FOAM_SPREAD)).min(float(0.4)).toVar();
    const acc = vec4(mix(self.x, nb.x, spread), self.y, self.z, self.w).toVar();
    const inj = brk.mul(gate).mul(dt).toVar();
    // The onset clock. Keyed on the CAP ACCUMULATOR being present, NOT on the
    // detector: the convergence term fires a beat before the fold (by
    // design), so a detector-keyed clock completes during the invisible
    // pre-fold phase and the cap still arrived fully formed. Keyed on
    // deposition, the ramp overlaps exactly the phase the eye watches.
    // Advected with the rest of the state (self2), so a blooming patch rides
    // its water; dies with the cap's own tau once the event passes.
    const onDep = smoothstep(float(0.03), float(0.15), acc.x).toVar();
    // rate jitter ~0.45x-1.8x off the cellular channel — the seeds
    const bRate = evtT.r.sub(0.41).mul(3.2).add(1).clamp(0.45, 1.8).toVar();
    const bloom = mix(
      self2.z.mul(exp(dt.mul(-1 / TAU_CAP))),
      self2.z.add(dt.mul(bRate).mul(1 / BLOOM_T)).min(float(1)),
      onDep,
    ).toVar();
    const env = mix(float(BLOOM_FLOOR), float(1), bloom).toVar();
    // The residual channels follow the wind — see foamSeaState() in params.js.
    // Cap deliberately does not: the extra budget of a harder sea is aged foam
    // and plume, not more simultaneous Stage-A caps.
    const injR = inj.mul(foamSea.residScale).toVar();
    const cap = saturate(acc.x.mul(exp(dt.mul(-1 / TAU_CAP))).add(inj.mul(INJ_CAP).mul(env))).toVar();
    const trail = saturate(
      acc.y.mul(exp(dt.mul(foamDecay).mul(-1 / TAU_TRAIL))).add(injR.mul(INJ_TRAIL)),
    ).toVar();
    // ...with a mild lane factor of its own: the bulk web is most of the foam
    // the eye sees, and windrow organisation that only reaches the faint
    // remnant channel underneath it is invisible. 1.6:0.75 against the
    // remnant's 7:1 — a bias, not a stencil.
    const holdL = mix(float(1.6), float(0.75), laneS).toVar();
    const lace = saturate(
      acc.z.mul(exp(dt.mul(foamDecay).mul(holdL).mul(-1 / TAU_LACE))).add(injR.mul(INJ_LACE)),
    ).toVar();
    // The second exponential — see INJ_REM. Decay RATE is spatial: fast off
    // the survival lanes and in divergent water, slow on them and in
    // convergence, which is Langmuir assembly acting on a lifetime rather than
    // a stamp of streaks painted on. Injected from the same events (a remnant
    // is always born under bulk foam).
    const hold = mix(float(REM_FAST), float(1.0), laneS)
      .mul(mix(float(1.5), float(0.6), divR)).toVar();
    const rem = saturate(
      self2.x.mul(exp(dt.mul(hold).div(foamSea.remTau).negate())).add(injR.mul(INJ_REM)),
    ).toVar();
    const holdA = mix(float(AER_FAST), float(1.0), laneS)
      .mul(mix(float(1.3), float(0.75), divR)).toVar();
    const aerAcc = mix(self2.y, nb2, saturate(dt.mul(60).mul(AER_SPREAD)).min(float(0.4))).toVar();
    const aer = saturate(
      aerAcc.mul(exp(dt.mul(holdA).div(foamSea.aerTau).negate())).add(injR.mul(INJ_AER)),
    ).toVar();
    foam2.element(id).assign(vec4(rem, aer, bloom, laneS));
    // Age: zeroed while breaking, integrating dt once the break has passed.
    // smoothstep rather than a step so a texel on the edge of an event does not
    // flicker its age between 0 and whatever it had accumulated.
    const fresh = smoothstep(float(0.25), float(0.45), inj.div(max(dt, float(1e-4)))).toVar();
    // The reset is exponentiated by the frame count a real 60 fps second has,
    // so the age a break-edge texel equilibrates to does not depend on dt --
    // a bare (1 - fresh) baked 2x the rim age at the capture loop's 30 fps.
    const keep = pow(max(float(1).sub(fresh), float(1e-6)), dt.mul(60)).toVar();
    const age = acc.w.add(dt).mul(keep).min(float(AGE_MAX)).toVar();
    foam.element(id).assign(vec4(cap, trail, lace, age));

    const disp = vec3(DxDz.x.mul(lambda), DyDxz.x, DxDz.y.mul(lambda)).toVar();
    prevDisp.element(id).assign(vec4(disp, trace));

    // displacement.w carried a duplicate of the trail channel that nothing read.
    // It now carries the surface divergence, remapped so 0.5 is neutral: for a
    // linear wave du/dx = -a*omega*k*sin(kx - wt), convergent on the forward
    // face and divergent on the rear, and with this project's measured RMS
    // slopes and lambda = 2.2 the area modulation is about +/-20%. That is why
    // real residual foam looks braided rather than evenly spread — it is
    // squeezed thicker and stretched thinner once per wave. (divR itself is
    // computed up with the detector, where the decay modulation needs it.)
    textureStore(displacement, coord, vec4(disp, divR)).toWriteOnly();
    textureStore(derivatives, coord, vec4(DyxDyz.x, DyxDyz.y, DxxDzz.x.mul(lambda), DxxDzz.y.mul(lambda))).toWriteOnly();
    textureStore(foamMap, coord, vec4(cap, trail, lace, age)).toWriteOnly();
    textureStore(foamMap2, coord, vec4(rem, aer, bloom, laneS)).toWriteOnly();
  })().compute(N * N);

  return { displacement, derivatives, foamMap, foamMap2, foam, foam2, assemble };
}
