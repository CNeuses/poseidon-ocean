import { StorageTexture, HalfFloatType, RepeatWrapping, LinearMipmapLinearFilter } from 'three/webgpu';
import {
  Fn, instanceIndex, uint, uvec2, vec2, vec3, vec4, float, max, length, saturate,
  smoothstep, exp, attributeArray, textureStore,
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
const TAU_TRAIL = 0.62; // s / foamDecay (~1.6 s) — dragged down the face and behind it
// 3.9 s, not the 22 s this started at. The chop cascade's patch is 144 m and
// periodic: over a long enough tail every texel in it has been under a breaking
// crest at some point, the lace channel converges on 1 everywhere, and the whole
// foreground comes out as one flat veil instead of as separate dissipating
// patches. What keeps it honest at this length is the low injection above — the
// channel is allowed to be old, it is not allowed to be opaque.
const TAU_LACE = 1.55; // s / foamDecay (~3.9 s) — the old dissipating web in the troughs

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
//   displacement = ( lambda*Dx, height, lambda*Dz, foam trail )
//   derivatives  = ( dDy/dx, dDy/dz, lambda*dDx/dx, lambda*dDz/dz )
//   foamMap      = ( cap, trail, lace, break rate this frame )
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
// behaves if it were switched on for the derivative map). displacement.w still
// carries the trail channel so that documented slot stays a foam value rather
// than becoming meaningless padding — nothing reads it today.
export function createCascadeMaps(cascade, { N, lambda, dt, foamDecay }) {
  const displacement = mapTexture(N);
  const derivatives = mapTexture(N);
  const foamMap = mapTexture(N, true);
  cascade.foamMap = foamMap; // consumed by foamShading.js

  const foam = attributeArray(N * N, 'vec4'); // (cap, trail, lace, break); zeroed = clean water
  const prevDisp = attributeArray(N * N, 'vec4'); // (last frame's displacement, last frame's Jacobian trace)

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
    const converge = saturate(
      prev.w.sub(trace).div(max(dt, float(1e-4))).sub(CONV_OPEN).mul(1 / (CONV_FULL - CONV_OPEN)),
    ).toVar();

    const brk = smoothstep(float(BRK_LO), float(BRK_HI), saturate(
      fold.mul(W_FOLD)
        .add(converge.mul(steep).mul(W_CONV))
        .add(fold.mul(steep).mul(W_BOTH)),
    )).toVar();

    const acc = foam.element(id).toVar();
    const inj = brk.mul(dt).toVar();
    const cap = saturate(acc.x.mul(exp(dt.mul(-1 / TAU_CAP))).add(inj.mul(INJ_CAP))).toVar();
    const trail = saturate(
      acc.y.mul(exp(dt.mul(foamDecay).mul(-1 / TAU_TRAIL))).add(inj.mul(INJ_TRAIL)),
    ).toVar();
    const lace = saturate(
      acc.z.mul(exp(dt.mul(foamDecay).mul(-1 / TAU_LACE))).add(inj.mul(INJ_LACE)),
    ).toVar();
    foam.element(id).assign(vec4(cap, trail, lace, brk));

    const disp = vec3(DxDz.x.mul(lambda), DyDxz.x, DxDz.y.mul(lambda)).toVar();
    prevDisp.element(id).assign(vec4(disp, trace));

    textureStore(displacement, coord, vec4(disp, trail)).toWriteOnly();
    textureStore(derivatives, coord, vec4(DyxDyz.x, DyxDyz.y, DxxDzz.x.mul(lambda), DxxDzz.y.mul(lambda))).toWriteOnly();
    textureStore(foamMap, coord, vec4(cap, trail, lace, brk)).toWriteOnly();
  })().compute(N * N);

  return { displacement, derivatives, foamMap, foam, assemble };
}
