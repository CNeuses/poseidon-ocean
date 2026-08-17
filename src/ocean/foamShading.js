import {
  vec2, vec3, float, texture, saturate, smoothstep, dot, mix, max, log2, fwidth,
  positionWorld,
} from 'three/tsl';
import { FOAM_PEAK } from './maps.js';

// How foam is *drawn*. maps.js writes a per-cascade foamMap whose channels are
// (cap, trail, lace, age): the same break run through three accumulators with
// different amplitudes and lifetimes, plus the seconds elapsed since the water
// under this texel last broke.
//
// The three regimes are two physical OBJECTS, not two brightness levels:
//
//   CAP    Stage A. The actively-breaking crest, while air is still being
//          entrained. Optically thick, essentially no through-holes, its
//          structure is RELIEF rather than lacunae. Short-lived — about a third
//          of a wave period — and roughly a tenth of the total foam area.
//   TRAIL  early Stage B. The bubble raft the crest has just run out from
//          under. This is the streak, and it is not painted: the foam map is
//          indexed by the undisplaced grid coordinate, which is the Lagrangian
//          label of a water particle, so foam written into it does not move
//          while the wave train sweeps through the label field at the phase
//          speed. At the 8-12 m/s of the 40-100 m waves that actually fold, a
//          2.6 s effective lifetime IS a 23 m band lying behind the crest.
//   LACE   late Stage B. The same raft, drained and fragmented into a filament
//          network, ~45 m of it, dying in the trough it slid into.
//
// Stage B is 80-90% of all foam on a real sea and the previous version of this
// file drew none of it. Every mask was normalised so that the cap peaked at
// three times the threshold while the trail peaked at 1.05x and the lace never
// cleared at all; then `max()` of the three discarded whichever had survived.
// One threshold plus a max is structurally the same operation that made Kleiss
// & Melville (2010) measure a Stage-B-to-Stage-A area ratio of 1.5 against the
// literature's 10, and they flagged it themselves: a single brightness
// threshold "may not capture the full extent of dim old foam and foam streaks".
//
// So the arrangement here is:
//   - every mask normalised to a nominal peak of 1 by the file's own stated
//     rule, 1 / (deposit x gate mean x carve mean), with the deposit imported
//     from maps.js rather than restated;
//   - one threshold, which only gives a mask its edge;
//   - the three composited OVER each other rather than max()ed, so a fresh cap
//     genuinely sits on top of the older foam around it;
//   - the age ladder expressed purely as ALPHA. Foam is a conservative
//     scatterer — its single-scattering albedo is within a millionth of 1 across
//     the visible — so old foam is THINNER, never greyer. One white albedo, one
//     alpha ladder, and the measured composite reflectances of fresh, thin and
//     residual foam all fall out of it.
//
// ctx: { cascades, lengthScales, worldXZ, shading, detailTex, N (surface normal) }
// returns { coverage: float 0..1, color: vec3 }

// Height that reads as a full crest — matches WAVE_SCALE in
// oceanSurfaceMaterial.js, which is the same wave the value ramp is keyed to.
const WAVE_SCALE = 2.6;

// Noise tile sizes in metres. The baked fbm's coarsest octave is a quarter of
// its tile and its finest is a thirty-second, so a 30 m tile draws roughly
// 7 m features down to 1 m ones. Long over wide is about 9:1 on both pairs,
// which is the streakiness. Each tile is also, implicitly, a DISTANCE: the
// texture is mipmapped, so a tap stops carrying structure and starts returning
// its own mean once the pixel footprint passes about a quarter of the tile.
const STREAK_LONG = 30; // along the wave
const STREAK_WIDE = 3.2; // across it
const STREAK_LONG2 = 8.5;
const STREAK_WIDE2 = 1.0;
// The same first tile at a longer aspect, crossfaded in with AGE. A foam patch
// does elongate as it ages, but bounded: the breaker jet's ~0.2 1/s strain acts
// for well under a second, the orbital strain is oscillatory and nets to zero
// over a period, and the Langmuir strain that could do it properly is 0.003-
// 0.016 1/s — two orders too slow to matter inside a whitecap's life. So this
// is a 2.5x aspect change, not an exponential.
//
// Done as a second tap and a crossfade rather than by dividing the `along`
// coordinate by an age-varying factor, and that is not a style choice. worldXZ
// is absolute world position and runs to thousands of metres, so d(along/s)/dx
// picks up an along*(ds/dx)/s^2 term tens of times the normal UV rate; the
// hardware would select the coarsest mip and the carve would return its own
// mean — the exact opposite of the intent, and worsening with distance from the
// world origin.
const STREAK_LONG_OLD = 75;
const CHUNK_TILE = 6.5; // chunky cap break-up, at whitecap scale
// Bubble-cell holes — what stops the cap being a sheet.
//
// A TILE IS NOT A FEATURE. The baked field's octave variance shares are
// 55/25/12/5/2%, so it is dominated by its coarsest octave, and its radial
// autocorrelation reaches half height at 0.095 of a tile. Stating that as
// feature DIAMETER, which is what the eye reads:
//
//     B channel: feature ~= 0.19 x tile.   A channel: ~= 0.095 x tile.
//
// (A is the same field at twice the frequency — detailTexture.js:102.) So a
// 1.7 m tile read mostly through A was delivering holes about SIXTEEN
// CENTIMETRES across, and the comments in this file that described it as
// metre-scale were out by a factor of ten. That is the whole of "the foam
// texture is too concentrated": ~45 pits inside a median 1.4 m whitecap, all
// the same size, in a field with negative kurtosis so there is no tail of rare
// large ones either. Correctly sized holes, far too many of them — which reads
// as a repeating screen rather than as a raft coming apart.
//
// 2.8 m delivers ~40 cm fresh and ~65 cm aged: five to six times fewer holes at
// the same hole-area fraction. Bounded above at about 3.5 m, past which a median
// whitecap gets one or two holes and reads as a card with bites taken out of it.
//
// The hole carves now read the CELLULAR channels (R and G) rather than the fbm
// ones, and the cell counts in detailTexture.js were picked so their delivered
// feature size matches the fbm pair they replaced — 0.18 and 0.078 of a tile
// against 0.176 and 0.088. So this constant still means what it says, and the
// sizes above are unchanged by the swap. What changed is the ARRANGEMENT: see
// the `cells` carve below.
const CELL_TILE = 2.8;
// ...and the two decorrelation defences have to scale WITH the tile, or they
// weaken by exactly the factor that makes the features more visible. Both are
// pure translations of the sampling frame, so neither moves a mean, a variance
// or a spectrum, and no constant downstream has to follow.
const CELL_WARP = 2.15; // m — holds the warp at 0.77 tiles
// The whole fine-detail frame wanders by up to a couple of metres over a tile
// this long. Every carve below repeats at ITS OWN tile size — the cells every
// 1.7 m, the chunk every 6.5 m — and at deck range those are large on screen and
// small in the world, which is the worst combination: a magnified capture of a
// whitecap showed the same comma-shaped hole at the same spacing over and over,
// a brush stamped across the raft. A slow domain offset means the fine tiles
// never line up with themselves twice in any stretch of sea the eye can hold.
//
// A pure translation, so it changes nothing statistical — no mean, no variance,
// no spectrum, and therefore no constant below has to move. Non-harmonic with
// every fine tile on purpose, and coarse enough that it never itself reads as a
// feature: at 137 m it is well past the point where the texture has mipped to
// its own mean in the far field, and in the near field it varies too slowly to
// see. It is also, deliberately, not applied to the group masks — those ARE
// meant to be a fixed geometry of where foam may live.
const WANDER_TILE = 137;
// Multiplies a channel whose sigma is 0.1328, so this is 0.53 m of actual
// offset, not 4 m — scaled with CELL_TILE to hold it at ~0.19 tiles.
const WANDER = 4.0;
// Wave-group scale, and the other way round: features hundreds of metres long
// down the wave and tens of metres apart ALONG the crest. Two tiles at a
// deliberately non-harmonic ratio, multiplied rather than maxed: one group
// frequency alone beats regularly enough to read as a lattice, and a product
// genuinely closes — whole stretches of sea carry no foam at all.
//
// These are also the only carves coarse enough to survive to the horizon, so
// they are what makes distant foam read as separate dashes rather than a band.
// The along-crest segmentation that used to live here as `crestGap` has moved
// upstream to the break INJECTION in maps.js, where it belongs: segmenting the
// drawn mask carves holes in a continuous raft and gives a dashed cornice,
// whereas segmenting the event gives separate patches that each sweep their own
// fully formed trail.
// Whitecap-scale scatter — see the `scatter` carve. The tile sets the DENSITY
// of whitecaps and the thresholds set their SIZE, which is the pair of knobs an
// fbm can never give you separately. The coarse cellular channel's feature is
// 0.18 x tile, so 18 m draws discs about 3.3 m across on a 18 m lattice: inside
// the 1-15 m breaking-segment range, and small enough to cut the 20-40 m masses
// the event stencil paints into separate whitecaps.
//
// SCATTER_LO/HI are in the baked field's own units, N(0.4089, 0.1328). Passing
// at r < ~0.36 keeps roughly a third of the lattice, so foam comes out as
// scattered discs rather than a sheet; the event stencil in maps.js is opened to
// compensate for the area that costs, because rationing at whitecap scale is
// physically the right place to do it and rationing at 24 m was not.
// Sampled ANISOTROPICALLY, in the heading frame, and that is not a refinement —
// it is the difference between whitecaps and coins. A cellular field thresholded
// isotropically gives round discs on a jittered lattice, and a round disc is the
// single most artificial shape foam can take: a real whitecap is a segment of a
// breaking crest, so it is elongated ALONG the crest line and short along the
// direction of travel, typically 2:1 to 5:1. Long across, short along.
const SCATTER_ALONG = 28.0; // m of tile along the heading  -> ~5.0 m of feature
const SCATTER_ACROSS = 76.0; // m across it, i.e. along the crest -> ~13.7 m
// ...and domain-warped before it is sampled, for the same reason every other
// tiled carve in this file is. One feature per cell makes a cellular field
// MORE regular than random — which is what kills the clumping, and is also
// exactly what reads as an evenly spaced array once the cells are large enough
// to see. The warp has to be comparable to the cell spacing to break that up,
// not a fraction of it.
const SCATTER_WARP_TILE = 65.0; // m — coarser than the scatter it displaces
const SCATTER_WARP = 7.0; // m of displacement (1 sigma), against ~6 m of spacing
const SCATTER_LO = 0.32;
const SCATTER_HI = 0.50;
const SCATTER_FLOOR = 0.26;
const SCATTER_DRIFT = 1.1; // m/s along the heading, so it is not a fixed stencil
const GAP_A_LONG = 300;
const GAP_A_WIDE = 48;
const GAP_B_LONG = 1000;
const GAP_B_WIDE = 240;
const WARP_COARSE = 3.6; // m of isotropic domain warp on the whole streak frame
const WARP = 3.2; // m of domain warp on the fine streaks
const DRIFT = 0.55; // m/s the streak pattern is dragged along the wave

// --- value ------------------------------------------------------------------
// Foam's bulk albedo. Measured composite reflectances of foam over water are
// 0.40-0.50 for a fresh multi-layer raft, ~0.18 for thin residual foam, and
// 0.02-0.07 for bare water; Koepke's lifetime-and-area average is 0.22. All
// three composites fall out of ONE white albedo and the alpha ladder below —
// 0.75*0.55 + 0.25*0.05 = 0.425, 0.50*0.55 + 0.50*0.05 = 0.30, and
// 0.25*0.55 + 0.75*0.05 = 0.175 — which is the whole argument for expressing
// age as opacity and never as value. The slight red deficit is Frouin's
// measured foam albedo at 670 nm (0.889 against 1.00 across 412-555 nm).
const ALBEDO = 0.55;
const ALBEDO_TILT = vec3(0.94, 1.0, 1.0);
// Value, linear, before exposure (1.2) and the tone curve. NOTE the curve is
// Khronos PBR Neutral, not ACES — see main.js. Neutral takes linear 1.0 to
// sRGB 242 and 2.0 to 250, so it is even flatter up there than ACES and the old
// constants in this file, which reasoned against ACES throughout, were authored
// about 1.5x hot for the curve that actually runs. The albedo above is what
// pays for that; the gains are unchanged.
const SUN_GAIN = 2.05;
const SKY_GAIN = 1.05;
// How far the sun is pulled toward its own luminance-grey BEFORE it lights foam,
// and this is a geometry argument rather than a grade. The sun here is golden
// hour: 0xffd395 resolves to a linear (1.15, 0.749, 0.346), a red-to-blue ratio
// of 3.3. A specular surface sees that beam and nothing else, which is why the
// glitter is allowed to be that colour. Foam does not: it is a dense multiple-
// scattering medium with a near-Lambertian response, so the light leaving it is
// an integral over an entire hemisphere in which the sun is one small, very
// bright solid angle among a whole sky. Lighting it with the undiluted beam is
// what makes a whitecap print at 241/216/171 — putty, the failure this file's
// ambient terms already exist to avoid, arriving through the other door.
// Measured: 0.35 takes the ratio to 2.0 and the foam back to a warm white.
const FOAM_SUN_DESAT = 0.35;
// How far past the terminator the bubble raft keeps lighting. Multiple
// scattering inside foam wraps illumination across its transport length, which
// is 2.4-8.5 mm — millimetre structure, i.e. sub-pixel roughness. It does NOT
// wrap light across the 5-50 cm relief of a roller, which has to self-shadow
// instead; that is the foam normal's job below, and it is why this stays small.
const WRAP = 0.22;
const FOAM_TINT = 0.10; // how much of the palette's foam hue survives (rest is white)
// The sky ambient is why foam went tan: shading.horizon under a 135-degree sun
// azimuth resolves to a warm sand colour, and 45% of that on white foam is
// putty. Foam is not a mirror — it is a diffuse white solid, and a diffuse white
// solid under a full sky dome reads as the AVERAGE of the dome, far nearer
// neutral than any one direction of it. So the ambient is pulled most of the way
// to its own luminance and then pushed COOL: neutral is the trap, because
// luminance is 72% green and a desaturated warm sky lands on khaki. A shadowed
// whitecap is blue-white or it is dirt.
const AMBIENT_NEUTRAL = 0.78;
const AMBIENT_COOL = vec3(0.84, 0.97, 1.20);
// A trough sees only a slot of sky. Raised from 0.60 because the trough is now
// populated: with the crest-lock gates gone, most Stage B foam lives down there,
// and a 40% cut on its only light source was turning the majority of the foam
// on screen to mud.
const SKY_VIS_MIN = 0.70;

// Pixel footprint, in metres, at which the foam MAP is fully band-limited and
// its mip is returning the local area fraction rather than a mask. Past this the
// threshold is gone and coverage is read linearly. It is not a fade-out: a sea
// that is 14% foam should render 14% foam, and at the horizon that is a faint
// wash rather than the hard bar a threshold prints.
// ...and it starts at an ONSET rather than at zero footprint, which the old
// single-constant form got wrong in a way that mattered. `lod` for the foam map
// is log2(footprint / 4.0).max(0), so below a 4 m footprint the tap is plain
// bilinear at full sharpness and the mip has averaged nothing at all — the claim
// that the map is "fully band-limited and returning the local area fraction"
// was false everywhere in 0 < footprint < 4 m. At FAR_FOOT 2.4 measured from
// zero, `far` was already 0.42 at a one-metre footprint, so most of the visible
// mid-field was being drawn by the unthresholded linear branch, which is
// inherently soft. That is half of why residual foam read as an airbrush.
const FAR_ONSET = 2.0; // m of footprint before the linear read starts to mix in
const FAR_SPAN = 2.0; // ...and over which it takes over completely
const FAR_GAIN = 1.7; // ...compensating for the carves having gone to their means
const FAR_CEIL = 0.60; // and a ceiling, so a grazing view is caps on blue, never a sheet
// The blunt second distance ramp that used to sit alongside this — a straight
// fade of COVERAGE over view distance — is gone. It double-counted a term that
// oceanSurfaceMaterial.js already applies correctly at the end of colorNode,
// compositing Beer-Lambert aerial perspective over the finished surface
// including the foam. Scaling coverage rather than colour does not veil distant
// foam, it makes there be LESS of it, so the same patch of water changed
// opacity as the camera dollied toward it.

// Per-regime opacity ceilings, and this is where "old foam is thin" is
// expressed — see ALBEDO. Coverage is used as a straight lerp against the water
// in oceanSurfaceMaterial, so it is also the opacity.
const OP_CAP = 0.75;
const OP_TRAIL = 0.50;
const OP_LACE = 0.25;
// Ceiling on the OVER composite, which is a different number from OP_CAP and
// was wrong to share it. With trail at 0.50 and lace at 0.25 the composite base
// is already 0.625, so clamping the sum at 0.75 started clipping once capC
// reached 0.333 — 44% of its own ceiling — and since most of a cap's interior
// is saturated anyway, the whole cap-over-trail region came out as one flat
// alpha. That is a lozenge in VALUE as well as in outline. 0.90 is what the
// file's own albedo argument allows: 0.90*0.55 + 0.10*0.05 = 0.50, the top of
// the measured 0.40-0.50 composite reflectance band for a fresh raft.
const COMPOSITE_MAX = 0.90;

// Each regime's mask is normalised to a nominal peak of 1 so the one GUI
// threshold means the same thing to all three. The rule is the reciprocal of
// (what one break deposits) x (the mean of this regime's gate product) x (the
// mean of its carve) — stated in this file for a long time and violated by the
// literals that used to be here, which over-normalised the cap 1.3x and
// under-normalised the trail 2.1x. That 2.8x relative error in the cap's favour
// was exactly the ratio by which the cap out-ran the trail at the threshold.
//
// Gate means are conditional on being INSIDE a patch — the norm has to
// normalise the peak the mask reaches where foam actually is, not the average
// over open water. Each is the product of its mix() floors against an estimated
// mean of the gate there.
const GATE_CAP = 0.905 * 0.806 * 0.79 * 0.865; // veto x face x lip x patch
const GATE_TRAIL = 0.74 * 0.865; // height x patch
const GATE_LACE = 0.865; // patch
const CARVE_CHUNK = 0.557;
const CARVE_STREAK = 0.723;
const CARVE_WEB = 0.670;
const CAP_NORM = 1 / (FOAM_PEAK.cap * GATE_CAP * CARVE_CHUNK);
const TRAIL_NORM = 1 / (FOAM_PEAK.trail * GATE_TRAIL * CARVE_STREAK);
const LACE_NORM = 1 / (FOAM_PEAK.lace * GATE_LACE * CARVE_WEB);

// params.foamThreshold is 0.4 and params.js belongs to another lane, so the
// working value is scaled here — the GUI slider still bites, the default just
// lands where the normalised masks above actually live.
const THRESH_SCALE = 1.10;
// Per-regime threshold PLACEMENT, in units of 1/foamScale. These used to be
// edge WIDTHS; the width now comes from the pixel footprint instead (see the
// cut() below). The values are unchanged because placing the threshold at the
// old window's midpoint is area-neutral by construction — at the default
// foamScale of 2.5 they put the crossings at 0.55 / 0.61 / 0.70 against a base
// T of 0.44, which is where the old ramps were half open.
const EDGE_CAP = 0.55;
const EDGE_TRAIL = 0.85;
const EDGE_LACE = 1.30;

// Local prominence, in metres of sea the "height around here" probe averages
// over, and the height above that average which reads as a full lip. Height
// above MEAN sea level says a point is high; it does not say the point is a
// crest, and on a swell the two are different by a whole wave.
//
// Cascade 0 ONLY, which is a change: summed over cascades 0 and 1 the probe was
// chop-dominated, because a 10 m average barely differs from the peak on a 55 m
// wave (0.07-0.15 m of prominence) while removing almost all of a 4-24 m chop
// crest (0.2-0.3 m). Making a chop-dominated signal the shape driver decorates
// every wavelet on the sea, which is the white lace tablecloth. LIP_DEPTH is
// re-derived for the swell alone and is an order smaller than it was.
const LIP_RADIUS = 10.0;
const LIP_DEPTH = 0.15;
// The tail. Offsets in metres DOWNWIND of the shaded point, so a break sampled
// ahead of us is painted behind itself. Lengthened from [0, 1.6, 3.4, 5.6]:
// cascade 0's texel is 4.0 m, so the old offsets were 0.4, 0.85 and 1.4 TEXELS
// and three of the four taps were reading essentially the same bilinear
// neighbourhood. These reach across the 23 m band the sim now writes.
const TAIL_D = [0.0, 4.0, 9.0, 15.0];
const TAIL_W = [1.0, 0.78, 0.58, 0.36];
// Metres of lateral shear per tap index — see the loop. Tap 3 picks up ~0.75 m,
// which is enough to make the dilation's structuring element a curve and small
// enough that the taps stay in order.
const TAIL_SHEAR = 0.25;

// Per-cascade weight on the cap. The cascades are a band split: cascade 0
// carries the swell and the 24-1024 m waves — including the whole 40-100 m band
// that actually folds — and cascade 1 the 4-24 m chop that rides on them. Both
// fold, and folds in the chop band are far more numerous, so unioned at equal
// weight the chop wins outright and every wavelet gets its own whitecap.
const CASCADE_W = [1.0, 0.32];

// Erosion. A carve on the MASK only ever bites near the threshold — once the
// interior of a cap is comfortably above it no amount of carve contrast reaches
// in, and the cap stays a solid card with a lacy rim. This one is taken out of
// the finished OPACITY instead, so it punches turquoise through the MIDDLE of a
// raft, which is what a raft of bubbles actually looks like.
//
// It is now driven by AGE, in two ways, because "the holes are always there and
// always the same size" is most of what "it fades arbitrarily" means:
//
//   DEPTH  a fresh Stage-A cap has essentially no through-holes at all — it is
//          optically thick, and its structure is the relief the foam normal
//          draws. Foam ages by drainage into Plateau borders, film thinning and
//          rupture, so the holes OPEN with age. The floor runs from 1.0 (no
//          holes) to 0.04 (nearly to water) rather than sitting at a constant
//          0.14, which was not a hole at all but a 14%-opacity dimple.
//   SIZE   and they GROW, about 40 cm to 65 cm, because a patch fragments — field
//          work on individual breakers needs a metre of dilation to reconnect
//          one whitecap's foam pixels late in its life. This is film rupture and
//          patch break-up, not bubble coarsening, which acts on a 0.1-1 mm
//          distribution and could never grow a metre-scale lacuna.
//
// The size term gets its own dedicated tap rather than re-parameterising the
// shared `cellN`: that one also feeds the cap's silhouette carve, the web and
// the grain, and age-scaling it would make a cap's own outline jitter with the
// age of the foam on it.
//
// Four frequencies and two anisotropies, never one, and re-weighted here: the
// old mix gave 53% of its variance to a single 1.7 m isotropic tap, and the deep
// excursion sets of a near-Gaussian isotropic field are round convex blobs at
// its correlation length — which is the mathematical definition of "evenly
// sized round holes". No tap now exceeds a third.
const AGE_FULL = 4.0; // s at which foam counts as fully aged, for erosion and aspect
const HOLE_FLOOR_NEW = 1.0; // a fresh Stage-A cap has no through-holes at all
// ...and an old raft opens most of the way to water. 0.15 rather than the 0.04
// this was first set to: the win here is the age GATING, not the depth — a
// fresh cap now gets no holes whatever, which is what the constant 0.14 floor
// could never express — and 25:1 of contrast on a baked value-noise field is
// enough to print the texture's own lattice. See the erosion block below.
const HOLE_FLOOR_OLD = 0.15;
// Cell tile multiplier at full age: 2.8 m fresh to 5.1 m of tile old, which is
// ~40 cm of feature fresh and ~65 cm aged — see CELL_TILE. The ratio matters
// far less than the RATIO — see the erosion block below. At 1.6 (4.4 m) and at
// 1.0 (3.4 m) the aged tile landed within a few percent of half CHUNK_TILE, and
// two value-noise lattices an octave apart reinforce instead of interfering.
const HOLE_GROW = 0.55;

// Diagnostic channel dump, off in every shipped frame. 1 = raw map ages,
// 2 = gated masks, 3 = thresholded coverages, 4 = group mask / face / footprint,
// 5 = no foam at all, 6 = finished coverage as grey (this is what tools/probe.mjs
// measures the foam area fraction from), 7 = age / lip / divergence.
const DEBUG = 0;

export function foamShading(ctx) {
  const { cascades, lengthScales, worldXZ, shading, detailTex, N } = ctx;
  const t = shading.time;
  // The one anisotropy axis, as a uniform — see foamHeading() in params.js. It
  // used to be a module constant of 45 degrees while windDirection was a live
  // slider, so every streak in the render silently desynced from the sea the
  // moment the wind was touched.
  const H = shading.foamHeading;
  const HX = H.x;
  const HZ = H.y;

  // world size of this pixel — the same band-limit the surface normal uses
  const footprint = max(fwidth(worldXZ.x), fwidth(worldXZ.y)).max(1e-3).toVar();
  const far = saturate(footprint.sub(FAR_ONSET).div(FAR_SPAN)).toVar();

  // --- break-up noise -------------------------------------------------------
  // Everything anisotropic is sampled in a frame aligned to the wave's heading,
  // then squashed: features come out long along the wave and thin across it,
  // which is the whole trick behind streaks. The fine octave's coordinate is
  // warped by the coarse one, because an unwarped set of parallel ribbons at one
  // spacing is a picket fence, and a picket fence is the second most synthetic
  // thing foam can do.
  // Everything fine-grained is sampled in a slowly wandering frame — see
  // WANDER_TILE. The group masks below deliberately stay on true worldXZ.
  const wanderN = texture(detailTex, worldXZ.div(WANDER_TILE)).toVar();
  const fineXZ = worldXZ.add(
    vec2(wanderN.b.sub(0.41), wanderN.a.sub(0.41)).mul(WANDER),
  ).toVar();
  const along = worldXZ.x.mul(HX).add(worldXZ.y.mul(HZ)).toVar();
  const across = worldXZ.x.mul(HZ.negate()).add(worldXZ.y.mul(HX)).toVar();
  const alongF = fineXZ.x.mul(HX).add(fineXZ.y.mul(HZ)).toVar();
  const acrossF = fineXZ.x.mul(HZ.negate()).add(fineXZ.y.mul(HX)).toVar();
  const drift = t.mul(DRIFT).toVar();

  const chunkN = texture(detailTex, fineXZ.div(CHUNK_TILE)).toVar();
  // The bubble-cell tap is domain-warped by the chunk tap. Without it the fine
  // tile's coarsest octave sits on the world axes at one fixed spacing, and any
  // threshold downstream prints it as a lattice of evenly spaced round dots — a
  // halftone screen laid over a wave face.
  //
  // It reads B and A, not RG. RG is the baked GRADIENT of the fbm, and the
  // gradient of a smooth field is small: detailTexture packs it as
  // (-h' * 3 * 0.5 + 0.5) into 8 bits and the whole field lands inside 128 +/- 3.
  const cellUV = fineXZ.add(vec2(chunkN.b.sub(0.41), chunkN.a.sub(0.41)).mul(CELL_WARP)).toVar();
  const cellN = texture(detailTex, cellUV.div(CELL_TILE)).toVar();
  // Two warps, at two scales, before either streak tile is sampled. One warp is
  // not enough: warping only the fine tile leaves the COARSE ribbons on a fixed
  // spacing, and the comb of fingers running down a wave face comes out evenly
  // spaced — a picket fence at whitecap scale.
  const across0 = acrossF.add(chunkN.b.sub(0.44).mul(WARP_COARSE)).toVar();
  const sA = texture(detailTex, vec2(alongF.add(drift).div(STREAK_LONG), across0.div(STREAK_WIDE))).toVar();
  const sAold = texture(detailTex, vec2(alongF.add(drift).div(STREAK_LONG_OLD), across0.div(STREAK_WIDE))).toVar();
  const acrossW = across0.add(sA.b.sub(0.44).mul(WARP)).toVar();
  const sB = texture(detailTex, vec2(alongF.add(drift.mul(1.7)).div(STREAK_LONG2), acrossW.div(STREAK_WIDE2))).toVar();
  const gapA = texture(detailTex, vec2(along.div(GAP_A_LONG), across.div(GAP_A_WIDE))).toVar();
  const gapB = texture(detailTex, vec2(along.div(GAP_B_LONG), across.div(GAP_B_WIDE))).toVar();

  // --- the three ages -------------------------------------------------------
  // The maps are indexed by the undisplaced grid coordinate — the Lagrangian
  // label of the water particle — so foam in them already rides the orbital
  // motion, and the wave train sliding through that label field already leaves
  // the aged channels behind the crest. What it does not do is REACH: a fold is
  // a hairline. The tail taps are that reach, a directed dilation along the
  // heading rather than a symmetric blur, so the result is a ribbon pointing the
  // right way instead of a dot with a halo.
  //
  // The cap unions cascades 0 and 1; the AGED channels take cascade 0 only.
  // That is a hard requirement, not a saving. With the crest-lock gates removed
  // below, cascade 1's trail — the 4-24 m chop band — clears threshold in
  // troughs with nothing left to stop it, and the result is the flat veil over
  // the whole foreground that this file has already had to back out of once.
  // Chop detexturises the swell's caps; it does not get to lay its own streaks.
  const cap = float(0).toVar();
  const trail = float(0).toVar();
  const lace = float(0).toVar();
  const ageV = float(0).toVar();
  // one shared lateral kink so the ribbons are not dead straight
  const kink = sA.a.sub(0.44).mul(1.6).toVar();
  cascades.forEach((c, i) => {
    if (!c.foamMap || i >= cascades.length - 1) return;
    const L = lengthScales[i];
    const uv0 = worldXZ.add(vec2(kink.mul(HZ.negate()), kink.mul(HX))).div(L).toVar();
    const texelM = L / c.N;
    // The mip level this pixel needs, worked out by hand rather than left to the
    // hardware: never sharper than the footprint allows.
    const lod = log2(footprint.div(texelM)).max(0).toVar();
    const cw = CASCADE_W[i] ?? CASCADE_W[CASCADE_W.length - 1];

    const f = texture(c.foamMap, uv0).level(lod).toVar();
    cap.assign(max(cap, f.x.mul(cw)));
    if (i > 0) return;

    trail.assign(max(trail, f.y));
    lace.assign(max(lace, f.z));
    // Age is NOT cascade-weighted and NOT unioned — it is a seconds-valued
    // scalar belonging to one event, and scaling it or maxing it across two
    // independent cascades would produce a number describing neither.
    ageV.assign(f.w);

    for (let k = 1; k < TAIL_D.length; k++) {
      // Sheared, not collinear. max() over taps offset along ONE axis is a
      // grayscale dilation by a line segment: it fills every boundary concavity
      // narrower than the tap spacing unconditionally, and that — plus a
      // monotone weight taper — is a lozenge by construction, whatever the mask
      // underneath it looks like. Displacing each tap laterally by a bounded
      // amount of an independent field makes the structuring element a ragged
      // curve instead. Bounded well under TAIL_D[1] = 4 m on purpose, so the
      // taps can never leapfrog and reverse the tail's order.
      const shear = vec2(sB.a.sub(0.41), sB.b.sub(0.41)).mul((TAIL_SHEAR * k) / 0.1328);
      const off = vec2(HX.mul(TAIL_D[k]), HZ.mul(TAIL_D[k])).add(shear).div(L);
      const g = texture(c.foamMap, uv0.add(off)).level(lod).toVar();
      trail.assign(max(trail, g.y.mul(TAIL_W[k])));
      lace.assign(max(lace, g.z.mul(TAIL_W[k])));
    }
  });
  // 0 fresh, 1 at AGE_FULL seconds. Drives hole depth, hole size and streak
  // aspect — everything that makes decay a change of SHAPE rather than a ramp
  // on alpha, which is the difference between foam dying and foam dissolving.
  const aged = saturate(ageV.div(AGE_FULL)).toVar();

  // --- where the lip is, and which way the water is straining ---------------
  // Two taps on cascade 0's displacement: this pixel's own mip level, and the
  // same fetch from whatever level averages it over LIP_RADIUS metres. Their
  // difference is how far this point stands proud of the sea immediately around
  // it. The sharp tap's .w carries the surface divergence maps.js stored there.
  const prom = float(0).toVar();
  const divR = float(0.5).toVar();
  cascades.forEach((c, i) => {
    if (i > 0 || !c.displacement) return;
    const L = lengthScales[i];
    const texelM = L / c.N;
    const lodD = log2(footprint.div(texelM)).max(0).toVar();
    const blur = Math.max(0, Math.log2(LIP_RADIUS / texelM));
    const here = texture(c.displacement, worldXZ.div(L)).level(lodD).toVar();
    const wide = texture(c.displacement, worldXZ.div(L)).level(lodD.max(float(blur))).y;
    prom.addAssign(here.y.sub(wide));
    divR.assign(here.w);
  });
  const lip = saturate(prom.div(LIP_DEPTH)).toVar();
  // Convergent water piles foam up, divergent water stretches it thin. For a
  // linear wave this is +/-20% at this project's measured slopes, and it is why
  // real residual foam looks braided instead of evenly spread. On the aged
  // channels only — the cap is pinned to the break by its own physics.
  const conc = mix(float(0.80), float(1.22), divR).toVar();

  // --- carves ---------------------------------------------------------------
  // Each carve is a multiplier on the MASK, never on the colour, and each one is
  // allowed to go well below 1 so it can punch a hole rather than only dim an
  // edge. None of them is faded with distance by hand: the detail texture is
  // mipmapped, so each tap gives up its structure and returns its own mean
  // exactly when the pixel stops being able to resolve it — cells at ~0.4 m of
  // footprint, chunk at ~1.6 m, the group masks not until ~12 m. That is a
  // three-stage band-limit for free.
  // Weight moved off A and onto B — see CELL_TILE. A is the same field at twice
  // the frequency, so its features are half B's, and at 1.45/0.50 this carve sat
  // 77.5% on A: the tile said 1.7 m and the eye got 16 cm. The SUM is held
  // identical (1.95) on purpose, because detailTexture renormalises to a fixed
  // mean and std, so every carve mean downstream — CARVE_CHUNK 0.557,
  // CARVE_STREAK 0.723, CARVE_WEB 0.670, and the norms derived from them — stays
  // valid without retuning. What it does NOT hold is the spread: pre-saturate
  // variance falls 19%, which shallows the low tail, and the low tail is what
  // punches the holes. If contrast reads short, widen the erode window rather
  // than moving weight back onto A, which would undo the change.
  // R and G, not B and A: the two cellular channels — see detailTexture.js.
  // Every carve that PUNCHES HOLES now reads them, and the ones that shape
  // silhouettes or stretch into streaks keep the fbm. The reason is spectral. An
  // fbm carries energy at every scale, so whole regions of it sit low and the
  // holes a threshold cuts inside those regions pile up — measured, its
  // autocorrelation is still 0.414 at a tenth of a tile and the block dispersion
  // of below-threshold pixels is 521. That clustering is what reads as clumpy,
  // and it is a property of the spectrum, so no tile size fixes it. A cellular
  // field has one feature per cell and cannot pile up: 0.117 and 303 at the
  // baked settings. The weights are unchanged, and so are CARVE_CHUNK,
  // CARVE_STREAK and CARVE_WEB, because the bake renormalises both families onto
  // the same mean and std and their CDFs agree within a point and a half at
  // every threshold this file cuts at.
  const cells = saturate(cellN.g.mul(0.95).add(cellN.r.mul(1.00)).add(sB.b.mul(0.6)).sub(0.45)).toVar();
  // The streak tile crossfades to a longer aspect with age — see STREAK_LONG_OLD.
  const sMix = mix(sA, sAold, aged).toVar();
  const streak = sMix.b.mul(1.25).add(sMix.a.mul(0.6)).add(sB.a.mul(0.8)).sub(0.15)
    .mul(mix(float(1.0), cells, float(0.55))).toVar();
  const chunk = chunkN.b.mul(1.55).add(chunkN.a.mul(0.85)).sub(0.20)
    .mul(mix(float(1.0), cells, float(0.70))).toVar();
  // The lacy carve mixes three frequencies *and* two anisotropies on purpose.
  // One dominant frequency at a threshold is a halftone screen.
  const web = cellN.g.mul(0.55).add(cellN.r.mul(0.70)).add(sB.a.mul(0.55)).add(sA.b.mul(0.45)).sub(0.25).toVar();

  // --- where foam is allowed to live ---------------------------------------
  // The height itself is JITTERED before any ramp, and that is not a detail. A
  // ramp on raw height draws an iso-height contour across the sea and the bottom
  // of every foam patch lands on it: a dead-straight horizontal cut under a white
  // sheet, which is the strongest single reason foam reads as a glacier lip.
  //
  // Its coarse term now comes from the 6.5 m chunk tile rather than from the
  // streak tile, whose feature ALONG the crest was STREAK_WIDE/4 = 0.8 m — so
  // the jitter had essentially no energy at the scale it needed to serrate. The
  // amplitude is unchanged and deliberately so: 1 sigma is about 14% of the ramp
  // width below, enough to make the boundary ragged and not enough to let the
  // gate open at random in a trough.
  const hJit = chunkN.b.sub(0.41).mul(0.62).add(cellN.b.sub(0.41).mul(0.38)).toVar();
  const hN = positionWorld.y.div(WAVE_SCALE).add(hJit).toVar();

  // The cap's height term is a VETO, not a silhouette. It used to be
  // smoothstep(0.04, 0.72) applied at mix(0.02, 1.0) — a 50:1 gate, five times
  // the dynamic range of every other gate in the file, and a monotone function
  // of world Y. That made the outline of every whitecap the y = 0.7 m level set
  // of the wave field: a smooth closed curve round the crest, which is the
  // elliptical raft. Wider and shallower, so it stops the cap drawing in a
  // trough without drawing a contour of its own — narrowing this window would
  // make the contour SHARPER, not weaker.
  const capVeto = smoothstep(float(-0.60), float(0.10), hN).toVar();
  // Trails live below and behind the crest that made them, because that crest
  // departed at 8-12 m/s while the label they are written on did not move at
  // all. Any gate keying trail visibility to present wave height enforces
  // kinematically impossible behaviour, and the one that used to be here —
  // smoothstep(-0.40, 0.48) at mix(0.02, 1.0) — attenuated the trail 50x at
  // exactly the 14 m mark where the sim was writing it. What is left is broad
  // and low-contrast: it still closes on the floor of a deep hollow, and does
  // nothing at all over the range a real trail occupies.
  const trailH = smoothstep(float(-1.40), float(-0.20), hN).toVar();
  // ...and lace has no height term. It BELONGS in a trough — that is where it
  // got to, by sliding off the crest that made it.
  const old = smoothstep(float(-1.00), float(0.05), hN).toVar(); // sky visibility only

  // A wave breaks forward, over its own leading face — Duncan's entraining
  // plume is thickest at the crest and tapers to the toe, and the back of the
  // crest stays clean dark water right to the edge. So a dot against the heading
  // separates front from back. This is now the cap's HARD gate, taking over the
  // dynamic range the height ramp used to have: at mix(0.45, 1.0) the back face
  // still drew at nearly half strength, which is a cap wrapped right over the
  // crest like icing. It also breaks the silhouette up on its own, because the
  // normal carries the ripple detail. Off the aged channels entirely — a trail
  // is behind the crest by definition, which is the back face.
  const lean = dot(vec2(N.x, N.z), H).toVar();
  // The bias sets what the crest LINE itself gets, where lean is 0 by
  // definition: at 0.30 the very top of a breaking crest came out at a third of
  // its cap, which fights the height veto — one wants the water high, the other
  // wants it on the forward slope, and their overlap was a narrow band partway
  // down the face. 0.40 leaves the back face gated as hard as before.
  const face = saturate(lean.mul(4.5).add(0.40)).toVar();

  // Two group masks at a deliberately non-harmonic ratio. They are the coarsest
  // carves in the file and the only ones the horizon can still resolve, so they
  // are also what decides whether distant foam reads as separate dashes or as a
  // continuous ribbon. Floored, not bare multipliers: with the injection in
  // maps.js now gated to a quarter of the surface, whitecaps arrive already
  // scattered, and multiplying that by a pair of masks that each close to zero
  // leaves three whitecaps on the whole sea. Their job is to VARY the foam, not
  // to ration it — that is the event stencil's job now.
  const patch = saturate(gapA.b.mul(2.5).sub(0.45))
    .mul(saturate(gapB.b.mul(2.5).sub(0.45))).toVar();
  const patchG = mix(float(0.70), float(1.0), patch).toVar();

  // --- scatter: the carve that makes whitecaps SEPARATE objects -------------
  // Measured on the coverage field with tools/foamblobs.mjs: 67% of all foam on
  // screen lived in FIVE connected components, and only twenty components in the
  // whole frame were bigger than 500 px. The rest was two-pixel dust. A real
  // whitecap field is hundreds of separate patches with a broad size
  // distribution; this was a handful of continents.
  //
  // The cause was a hole in the scale ladder. Isotropic separating structure
  // existed at 0.50 m (`cells`) and 1.24 m (`chunk`), then jumped to 9-57 m
  // (gapA/gapB) — and those are deliberately weak, mix(0.70, 1.0), because their
  // job is to vary foam rather than ration it. Nothing at all acted between
  // 1.2 m and 9 m, which is exactly whitecap scale: Callaghan's median foam
  // patch is 1.5 m^2, about 1.4 m across, and the Lambda(c) breaking-segment
  // range is 1-15 m. So the field could get surface texture and wave-group
  // variation, but nothing could ever cut one connected mass into two whitecaps.
  //
  // This is what a cellular field is actually for, and it is where the cellular
  // channels earn their place: F1 is LOW near each feature point, so gating on
  // low F1 leaves a disc per cell — discrete, separated, one per cell by
  // construction, at a density set by the cell count and a size set by the
  // threshold, with the baked lognormal radius giving them a size spread. Used
  // for holes at sub-metre scale it did nothing measurable, because at these
  // viewing distances it was already mipping to its mean.
  //
  // It DRIFTS along the heading, for the same reason the injection stencil in
  // maps.js does: a static world-space scatter is a permanent stencil, and the
  // sea would break in the same places forever.
  const scatW = texture(detailTex, worldXZ.div(SCATTER_WARP_TILE)).toVar();
  const scatOff = vec2(scatW.b.sub(0.41), scatW.a.sub(0.41)).mul(SCATTER_WARP / 0.1328).toVar();
  const scatXZ = worldXZ.add(scatOff).add(vec2(HX, HZ).mul(t.mul(SCATTER_DRIFT))).toVar();
  const scatN = texture(detailTex, vec2(
    scatXZ.x.mul(HX).add(scatXZ.y.mul(HZ)).div(SCATTER_ALONG),
    scatXZ.x.mul(HZ.negate()).add(scatXZ.y.mul(HX)).div(SCATTER_ACROSS),
  )).toVar();
  // Reversed smoothstep: high where the cellular field is low, i.e. inside a
  // disc. Floored rather than closing to zero — this decides where whitecaps
  // are, and a hard zero would delete the aged foam that has drifted off the
  // patch that made it.
  const scatter = mix(
    float(SCATTER_FLOOR), float(1.0),
    smoothstep(float(SCATTER_HI), float(SCATTER_LO), scatN.r),
  ).toVar();

  const capA = cap.mul(scatter).mul(mix(float(0.05), float(1.0), capVeto))
    .mul(mix(float(0.03), float(1.0), face))
    .mul(mix(float(0.30), float(1.0), lip))
    .mul(patchG).toVar();
  const trailA = trail.mul(scatter).mul(mix(float(0.35), float(1.0), trailH))
    .mul(patchG).mul(conc).toVar();
  const laceA = lace.mul(scatter).mul(patchG).mul(conc).toVar();

  // --- coverage -------------------------------------------------------------
  const capM = capA.mul(chunk).mul(CAP_NORM).toVar();
  const trailM = trailA.mul(streak).mul(TRAIL_NORM).toVar();
  const laceM = laceA.mul(web).mul(LACE_NORM).toVar();

  const T = shading.foamThreshold.mul(THRESH_SCALE).toVar();
  // The GUI slider now governs where the threshold sits, not how wide it is —
  // those were one number and should never have been. Capped, because uncapped
  // it inverts: at foamScale 0.2 the lace placement ran to 3.69, unreachable by
  // a mask normalised to peak 1, so lace vanished entirely over the slider's
  // bottom third.
  const T_PLACE = float(1).div(max(shading.foamScale, float(0.2))).min(float(0.8)).toVar();

  // Width comes from the pixel instead, and this is the change that hands the
  // contour from the mask to the carve.
  //
  // The window used to be a fixed soft*EDGE — 0.22 / 0.34 / 0.52 mask units for
  // cap / trail / lace. Lace's 0.52 is, at the ~0.2-per-metre roll-off of a
  // bilinear 4 m texel, about 2.6 METRES of sea, 27 px across the sight line at
  // 100 m. And lace is always the outermost non-zero coverage in the OVER
  // composite, so it is lace that draws every patch's outer silhouette. The
  // ratio of how far the carves can wander the crossing to how wide the window
  // is came out at 0.40; an edge reads as torn only when that exceeds about 1.
  // Below it the contour defaults to the mask's own level set, which is smooth
  // by construction because the mask is band-limited to a 4 m grid. That is the
  // lozenge.
  //
  // At ~1 px the same carves cross the threshold instead of dimming it, and the
  // boundary becomes theirs. But the window cannot go to one pixel everywhere:
  // on flat backs at several metres of footprint the carves have all mipped to
  // their means, and a hard threshold on a bare bilinear interpolant draws a
  // piecewise-hyperbolic contour that kinks at every texel boundary — the same
  // lattice failure this file has hit twice before. So the floor widens with
  // footprint over exactly the range that kills carve authority: tight where the
  // carves are alive, wide where they are not.
  const AA_MIN = 0.010; // guards fwidth == 0 on perfectly flat runs
  const AA_MAX = 0.120;
  const W_FLAT = 0.35; // ~the old lace band, for the carve-dead regime
  const wFloor = saturate(footprint.div(float(1.6))).mul(W_FLAT).toVar();

  // Near: a threshold, for a hard fractal edge. Far: the same mask read
  // LINEARLY, because past FAR_ONSET the foam map's mip has replaced it with the
  // local area fraction of foam, and a step function on an area fraction is what
  // paints the white bar along the horizon.
  const cut = (m, edge) => {
    const t = T.add(T_PLACE.mul(edge * 0.5)).toVar();
    const w = fwidth(m).mul(0.5).clamp(AA_MIN, AA_MAX).max(wFloor).toVar();
    return mix(
      smoothstep(t.sub(w), t.add(w), m),
      saturate(m.mul(FAR_GAIN)).mul(FAR_CEIL),
      far,
    );
  };
  const capC = cut(capM, EDGE_CAP).mul(OP_CAP).toVar();
  const trailC = cut(trailM, EDGE_TRAIL).mul(OP_TRAIL).toVar();
  const laceC = cut(laceM, EDGE_LACE).mul(OP_LACE).toVar();

  // Near field: composite the three OVER each other, so a fresh cap sits ON the
  // older foam around it and the regimes read as layers of one process rather
  // than as one mask at three opacities. Far field: max(), because once the
  // mip has area-averaged a pixel there is no layering left to model — the three
  // channels are nested, lace containing trail containing cap, so the widest one
  // is the answer and OVER would just inflate it into the horizon bar.
  const over = capC.add(
    trailC.add(laceC.mul(float(1).sub(trailC))).mul(float(1).sub(capC)),
  ).toVar();
  const raw = mix(over, max(capC, max(trailC, laceC)), far).min(float(COMPOSITE_MAX)).toVar();

  // Erosion, punched into the finished OPACITY rather than into the mask, and
  // driven by age — see AGE_FULL. Shaped as MOSTLY-OPEN WITH HOLES rather than
  // as a smooth multiplier: a smooth 0.1-to-1 noise with a mean near 0.65 does
  // not punch holes in a raft, it makes the whole raft two-thirds transparent,
  // and a cap that is everywhere translucent reads as spray or as dust on the
  // water. Through a steep smoothstep it sits at 1 over most of its area and
  // falls off a cliff in the lowest fifth.
  // Sampled on a ROTATED copy of the cell frame, which is load-bearing. The
  // detail texture is value noise on a square lattice, so every tap carries that
  // lattice; at 1.7 m it is small enough to disappear into the carve, but grown
  // for age it lands at metres and, crossed with the other axis-aligned taps,
  // printed a visible burlap weave across every foreground raft. A 30 degree
  // rotation is the cheapest thing that stops two lattices agreeing — domain
  // warping does not help, because a bent lattice is still a lattice.
  //
  // The fourth term is the 6.5 m chunk tile rather than the 8.5 x 1.0 m streak
  // tile for the same reason: crossing one 8.5:1 anisotropic field with two
  // isotropic ones is what makes a plaid. Streakiness is the mask carve's job;
  // this field's job is holes, and it wants two isotropic scales so the holes
  // come in a range of sizes.
  // Rotated 30 degrees AND warped by a metre of an independent field, and the
  // warp is the part that actually matters. Rotating alone only changed the
  // ANGLE of the weave, which is what identified the cause: this is one tap's
  // own value-noise lattice, printed because the age-driven hole floor swings
  // the opacity over the interior of a raft by nearly an order of magnitude.
  // The warp cellUV already carries is 0.17 m of standard deviation against a
  // 2.6 m tile — the comment upstream claiming half a metre is optimistic, and
  // at that amplitude a lattice survives intact. Displacing by 0.93 m of the
  // streak field instead is comparable to the tile itself, which does not blur
  // the lattice, it destroys it.
  // Rotated 30 degrees against the cell frame it is otherwise identical to.
  // This tap and `cellN` land in the same sum, at the same kind of scale, from
  // the same baked texture; sampled in the same frame their lattices would
  // agree, and two agreeing value-noise lattices under a threshold are a
  // halftone screen. Cheap, and it costs no fetch.
  const eUV = vec2(
    cellUV.x.mul(0.87).sub(cellUV.y.mul(0.50)),
    cellUV.x.mul(0.50).add(cellUV.y.mul(0.87)),
  ).toVar();
  // The 1.27 base sits here rather than in HOLE_GROW so the aged tile lands at
  // 2.8 x 1.82 = 5.10 m against CHUNK_TILE/2 = 3.25 m — a ratio of 1.57, safely
  // clear of the octave that makes two value-noise lattices reinforce instead of
  // interfere. Shipped before this change it was 2.64 against 3.25, a ratio of
  // 1.23, which is close enough to that octave to have been a standing risk.
  const eCell = texture(detailTex, eUV.div(
    float(CELL_TILE).mul(aged.mul(HOLE_GROW).add(1.27)),
  )).toVar();
  // Four terms across three scales and two frames, and a window wide enough not
  // to trace any one of their lattices. Three of the four taps are now cellular
  // (see `cells` above), which removes the specific failure this window was
  // widened for: the fbm channels are value noise on a square lattice with a
  // smoothstep interpolant, whose gradient is zero at every lattice point, so
  // its iso-contours bunch on the lattice edges and a steep enough threshold
  // prints them as a grid. A cellular field's structure is radial around its
  // feature points and has no such preferred direction. The window stays wide
  // anyway because the fourth tap is still fbm and because HOLE_FLOOR_OLD makes
  // this a ~7:1 swing on aged foam — deliberately not the 25:1 it was first set
  // to, which is what printed a burlap weave across every foreground raft.
  const erode = smoothstep(float(0.10), float(0.62), saturate(
    eCell.g.mul(0.70).add(eCell.r.mul(0.90))
      .add(chunkN.b.mul(0.80)).add(cellN.g.mul(0.70)).sub(0.79),
  )).toVar();
  const holeFloor = mix(float(HOLE_FLOOR_NEW), float(HOLE_FLOOR_OLD), aged).toVar();
  const coverage = raw.mul(mix(holeFloor, float(1.0), erode)).toVar();

  // --- shading --------------------------------------------------------------
  // One white albedo. Age is alpha and nothing else: foam scatters conservatively
  // in the visible, so a thinner raft transmits more turquoise, it does not turn
  // grey. Rendering old foam darker is what makes it read as ash.
  const fc = vec3(shading.foamColor).toVar();
  const hue = fc.div(max(max(fc.x, fc.y), fc.z).max(1e-3));
  const albedo = mix(ALBEDO_TILT, hue, float(FOAM_TINT)).mul(ALBEDO).toVar();

  // NO foam normal, and this is a measured decision rather than an omission.
  // A bubble raft does have 5-50 cm of roller relief that self-shadows, and the
  // obvious way to get it is to finite-difference the detail texture and perturb
  // N. That was built, and it does not work here: the gradient of an fbm is
  // dominated by its FINEST octave — this texture's persistence of 0.68 against
  // a doubling frequency makes each octave contribute 1.36x the gradient of the
  // one below — so on a 1.1 m tile the gradient lives at 8.6 mm, and
  // differencing it at any step large enough to be roller-scale aliases it into
  // a coherent beat. It printed a burlap weave across every foreground raft.
  // Band-limiting the taps to the step does not help either: at that mip the
  // texture IS its own texel grid, and the weave becomes a cleaner one.
  //
  // What settled it was measuring rather than tuning. With the perturbation on,
  // the aerial preset's brightest foam decile ran 236 -> 153 sRGB; with it off,
  // 235 -> 153. Identical. The value modelling the albedo drop was supposed to
  // need a normal to populate is already being populated — by the alpha ladder,
  // the age-driven holes and the turquoise coming through them. So this is three
  // texture fetches and an artifact in exchange for one code value.
  const wrap = saturate(dot(N, shading.sunDir).add(WRAP).div(1 + WRAP)).toVar();
  const sunC = vec3(shading.sunColor).toVar();
  const sunL = dot(sunC, vec3(0.2126, 0.7152, 0.0722)).toVar();
  const sunF = mix(sunC, vec3(sunL), float(FOAM_SUN_DESAT)).toVar(); // see FOAM_SUN_DESAT
  const sun = sunF.mul(wrap.mul(wrap.mul(0.35).add(0.65)).mul(SUN_GAIN));
  const amb = mix(vec3(shading.horizon), vec3(shading.zenith), float(0.35)).toVar();
  const ambL = dot(amb, vec3(0.2126, 0.7152, 0.0722)).toVar();
  const sky = mix(amb, AMBIENT_COOL.mul(ambL), float(AMBIENT_NEUTRAL)).mul(SKY_GAIN);
  const skyVis = mix(float(SKY_VIS_MIN), float(1.0), old).toVar();
  const lit = albedo.mul(sun.add(sky.mul(skyVis))).toVar();

  // A few percent of grain so a thick cap is a raft of bubbles with pockets in
  // it rather than a smooth card. Bounded on purpose: the break-up is the
  // coverage's job and this is the one term that turns foam grey if it is
  // allowed to do the shaping.
  const grain = saturate(cellN.a.mul(0.20).add(sB.b.mul(0.10)).add(0.85))
    .clamp(0.88, 1.06).toVar();
  const color = lit.mul(grain);

  if (DEBUG === 1) return { coverage: float(1), color: vec3(cap, trail, lace) };
  if (DEBUG === 2) return { coverage: float(1), color: vec3(capA, trailA, laceA) };
  if (DEBUG === 3) return { coverage: float(1), color: vec3(capC, trailC, laceC) };
  if (DEBUG === 4) return { coverage: float(1), color: vec3(patch, face, far) };
  if (DEBUG === 5) return { coverage: float(0), color: vec3(1) };
  if (DEBUG === 6) return { coverage: float(1), color: vec3(coverage) };
  if (DEBUG === 7) return { coverage: float(1), color: vec3(aged, lip, divR) };
  return { coverage, color };
}
