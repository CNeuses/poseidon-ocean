import {
  vec2, vec3, float, texture, saturate, smoothstep, dot, mix, max, log2, fwidth, length,
  positionWorld, cameraPosition,
} from 'three/tsl';

// How foam is *drawn*. maps.js writes a per-cascade foamMap whose channels are
// the same break run through three accumulators — (cap, trail, lace, break) —
// so foam arrives here already sorted by age AND by amplitude: a fresh cap
// saturates, a trail peaks near two-thirds, old lace never gets past a third.
// This file turns that into the three things Sea of Thieves' ocean shows:
//
//   CAP    thick, near-opaque white sitting on the break itself. Gated hard to
//          the top of the wave and to its leading face, and eroded by a bubble
//          -cell mask so it is a raft with water showing through it and not a
//          cornice of icing.
//   TRAIL  what the crest dragged down its own face and left behind it. The
//          same mask dilated BACKWARDS along the wave heading and carved by
//          noise stretched ~12:1 the same way, which is what turns a blob into
//          ribbons. Half-opaque at most: a streak is water with foam ON it.
//   LACE   the oldest foam, thinned to a dissipating web out in the troughs.
//          Low-contrast holed veil, low coverage ceiling — a tint on turquoise,
//          never a coat of paint.
//
// Three things drove this pass, all of them visible as "that is snow, not foam":
//
//   1. VALUE. The previous tuning drove a lit cap to 5.6 linear. ACES at
//      exposure 1.2 is nearly flat above 2 — 2.0 comes out at 232, 5.6 at 248 —
//      so the whole authored range from a shaded flank to a sunlit crown
//      collapsed into about four code values and every cap printed as one white
//      card. The sun term now peaks near 2.3 and the wrap is tight, so a cap
//      spans roughly 130 (sky-only flank) to 240 (crown): a hundred code values
//      of modelling, and no pixel reaches 255.
//   2. COVERAGE. Thin foam is THIN, not dark. Darkening it (the old
//      THIN_VALUE 0.46) and *also* reducing its opacity paid for thinness twice
//      and produced exactly the neutral 47-72% grey smeared down every wave face
//      that the reference forbids. Foam is white at every age here; only the
//      per-regime opacity ceiling changes, and the old regimes are capped low.
//   3. DISTANCE. Every carve in this file is a tap of a mipmapped tiling noise,
//      so each one converges on its own mean the moment the pixel footprint
//      outgrows it — the fine cell noise gives up first, the 6 m chunk next, the
//      300 m wave-group masks survive to the horizon. That hierarchy is the
//      whole distance model and it needs no help. What the far field DOES need
//      is a different transfer: out there the foam map's own mip has already
//      replaced the mask with its local area fraction, and a threshold on an
//      area fraction is a step function on a smooth field — it prints a solid
//      white bar along the horizon. Past a footprint of a metre or so the
//      threshold is therefore cross-faded out for a straight linear read, which
//      is what "this pixel is 14% foam" should look like.
//
// ctx: { cascades, lengthScales, worldXZ, shading, detailTex, N (surface normal) }
// returns { coverage: float 0..1, color: vec3 }

// Height that reads as a full crest — matches WAVE_SCALE in
// oceanSurfaceMaterial.js, which is the same wave the value ramp is keyed to.
const WAVE_SCALE = 2.6;

// Heading the wave train travels, degrees, in world XZ. Matches
// params.local.windDirection; every anisotropy in the file is built around it.
const FLOW_DEG = 45;
const FX = Math.cos((FLOW_DEG * Math.PI) / 180);
const FZ = Math.sin((FLOW_DEG * Math.PI) / 180);

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
const CHUNK_TILE = 6.5; // chunky cap break-up, at whitecap scale
const CELL_TILE = 1.7; // bubble-cell holes — what stops the cap being a sheet
const CELL_WARP = 1.3; // m the cell tile is dragged around by the chunk tap
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
const WANDER = 2.4; // m
// Wave-group scale, and the other way round: features hundreds of metres long
// down the wave and tens of metres apart ALONG the crest, which cuts a
// continuous ridge of icing into separate whitecaps with clean water between
// them. Two tiles at a deliberately non-harmonic ratio, multiplied rather than
// maxed: one group frequency alone beats regularly enough to read as a lattice,
// and a product genuinely closes — whole stretches of sea carry no foam at all.
// These are also the only carves coarse enough to survive to the horizon, so
// they are what makes distant foam read as separate dashes rather than a band.
const GAP_A_LONG = 300;
const GAP_A_WIDE = 48;
const GAP_B_LONG = 1000;
const GAP_B_WIDE = 240;
const WARP_COARSE = 3.6; // m of isotropic domain warp on the whole streak frame
const WARP = 3.2; // m of domain warp on the fine streaks
const DRIFT = 0.55; // m/s the streak pattern is dragged along the wave

// Value, linear, before exposure (1.2) and ACES. A crown facing the sun lands
// near 2.2 linear, which the filmic curve puts at ~238/255 — white, but with the
// roll-off still holding shape rather than clipping to a flat card, and with the
// channels still separated. A flank the sun has left lands near 0.24, which is
// ~128: a cool blue-grey shadow. That hundred-code-value span is the whole
// difference between a bubble raft and a snowdrift, and it is the reason these
// numbers are less than half what they were: at 5.6 linear the sunlit crown and
// the shaded flank were four code values apart, because ACES above 2 is nearly
// flat and everything authored up there arrives as the same white card.
const SUN_GAIN = 2.05;
const SKY_GAIN = 1.05;
// How far past the terminator the bubble raft keeps lighting. Foam scatters
// many times so it does light past 90 degrees, but only just: at the 0.55 this
// started at the wrapped lambert never fell below 0.35 and every cap was lit
// from every side at once, which is the other half of why they read as solid
// white cards.
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
const SKY_VIS_MIN = 0.60; // a trough sees only a slot of sky

// Pixel footprint, in metres, at which the foam MAP is fully band-limited and
// its mip is returning the local area fraction rather than a mask. Past this the
// threshold is gone and coverage is read linearly. It is not a fade-out: a sea
// that is 14% foam should render 14% foam, and at the horizon that is a faint
// wash rather than the hard bar a threshold prints.
const FAR_FOOT = 2.4;
const FAR_GAIN = 1.7; // ...compensating for the carves having gone to their means
const FAR_CEIL = 0.60; // and a ceiling, so a grazing view is caps on blue, never a sheet
// Beyond the transfer change there is a second, blunter distance term, and it
// exists because of what a horizon pixel physically is: at a deck camera's
// grazing angle a pixel two hundred metres out covers forty metres of sea ALONG
// the view ray. Every whitecap inside that strip is averaged into it, so the
// mip's area fraction climbs toward the fraction of the SEA that is foam rather
// than the fraction of a wave face — and printed at face value that is a
// continuous pale ribbon welded along the skyline, which is the single loudest
// tell that a render is not Sea of Thieves. Real distant whitecaps also lose
// most of their contrast to the air between: they are the first thing aerial
// perspective takes. So coverage is ramped down over the same footprint range,
// to a floor that leaves scattered dashes rather than a band.
// It is driven by the LARGER of two fades, because there are two different ways
// a whitecap stops being resolvable and only one of them is the footprint. A
// crest face two hundred metres out that happens to be tilted towards the lens
// has a perfectly small footprint — the pixels on it are sharp — and yet the
// twelve-metre gaps that segment its foam are already sub-pixel ALONG the crest,
// so the group masks have mipped away to their own means and the caps merge into
// one unbroken ribbon lying along the skyline. Plain distance catches that case;
// footprint catches the grazing one. Both end at the same floor.
const HORIZON_FOOT = 3.5; // m/px at which the far ceiling is fully applied
const HORIZON_DIST = 480; // ...or metres of view distance, whichever bites first
const HORIZON_CEIL = 0.28;

// Per-regime opacity ceilings. Coverage is used as a straight lerp against the
// water in oceanSurfaceMaterial, so it is also the opacity, and this is where
// "old foam is thin" is expressed — NOT in the colour. Even a fresh raft lets a
// little turquoise through; a streak is mostly water; lace is a tint.
const OP_CAP = 0.93;
const OP_TRAIL = 0.62;
const OP_LACE = 0.30;

// Each regime's mask is normalised to a nominal peak of 1 by these, so the one
// GUI threshold below means the same thing to all three. The numbers are the
// reciprocal of (channel peak from maps.js) x (gate product) x (carve mean).
const CAP_NORM = 10.5;
const TRAIL_NORM = 4.4;
const LACE_NORM = 6.0;

// params.foamThreshold is 0.4 and params.js belongs to another lane, so the
// working value is scaled here — the GUI slider still bites, the default just
// lands where the normalised masks above actually live.
const THRESH_SCALE = 1.10;
const EDGE_CAP = 0.55; // edge width, in units of 1/foamScale
const EDGE_TRAIL = 0.85;
const EDGE_LACE = 1.30;

// Lateral spread of the aged channels, in METRES — converted to a mip level per
// cascade below. Expressed as a level instead, the same number would blur the
// 1024 m cascade over tens of metres and the 144 m one over half a metre.
const SPREAD_M = 1.60;
// Local prominence, in metres of sea the "height around here" probe averages
// over, and the height above that average which reads as a full lip. Height
// above MEAN sea level says a point is high; it does not say the point is a
// crest, and on a swell the two are different by a whole wave — the top of a
// broad rolling hump is high and is not breaking, while the sharp little lip on
// the shoulder of a group is lower and is. Differencing the displacement map
// against a blurred read of itself gives the second thing directly: positive at
// a lip, zero on a plateau, negative in a hollow, continuous everywhere, and
// unable to draw an iso-height contour because it is a purely local comparison.
// This is the same probe oceanSurfaceMaterial builds its subsurface thickness
// from, which is also why the glow and the foam land on the same water.
const LIP_RADIUS = 10.0;
const LIP_DEPTH = 0.55;
const CAP_HALO = 0.18; // spread tap's share of the cap: enough to thicken, not to blur
const CAP_GAIN = 1.5; // ...gained back up, since the blur took the peak off it
const TRAIL_GAIN = 1.30; // gain back what the blurred tail taps averaged away
const LACE_GAIN = 1.15;
// The tail. Offsets in metres DOWNWIND of the shaded point, so a break sampled
// ahead of us is painted behind itself: that is a streak dragged out of the
// crest and left on the water. Weights fall off along it so the ribbon fades
// out rather than ending on a line.
const TAIL_D = [0.0, 1.6, 3.4, 5.6];
const TAIL_W = [1.0, 0.72, 0.50, 0.30];

// Per-cascade weight on the foam channels, and this is a look decision rather
// than a filtering one. The cascades are a band split: cascade 0 carries the
// swell and the waves you read AS waves, cascade 1 the four-to-twenty-metre chop
// that rides on them. Both fold, and folds in the chop band are far more
// numerous — so unioned at equal weight the chop wins outright and every little
// wavelet on the sea gets its own whitecap. That is what turns the field into a
// uniform white lace tablecloth no matter how the thresholds are set, and it is
// wrong about the reference besides: Sea of Thieves puts foam on the big waves,
// and the chop only breaks the EDGE of it up. Weighted down, cascade 1 can no
// longer clear the threshold on its own — it detexturises the swell's caps and
// adds near-field detail, which is the job it should have.
const CASCADE_W = [1.0, 0.32];

// Erosion. A carve on the MASK only ever bites near the threshold — once the
// interior of a cap is comfortably above it no amount of carve contrast reaches
// in, and the cap stays a solid card with a lacy rim. This one is taken out of
// the finished OPACITY instead, so it punches turquoise through the MIDDLE of a
// thick raft, which is what a raft of bubbles actually looks like. Three
// frequencies and two anisotropies, never one: a single octave of value noise
// through a threshold is a halftone screen, and prints the noise texture's own
// grid as evenly spaced round dots straight across the cap.
const ERODE_MIN = 0.14;

// Diagnostic channel dump, off in every shipped frame. 1 = raw map ages,
// 2 = gated masks, 3 = thresholded coverages, 4 = group masks + footprint,
// 5 = no foam at all (what the rest of the render looks like underneath).
const DEBUG = 0;

export function foamShading(ctx) {
  const { cascades, lengthScales, worldXZ, shading, detailTex, N } = ctx;
  const t = shading.time;

  // world size of this pixel — the same band-limit the surface normal uses
  const footprint = max(fwidth(worldXZ.x), fwidth(worldXZ.y)).max(1e-3).toVar();
  const far = saturate(footprint.div(FAR_FOOT)).toVar();

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
  const along = worldXZ.x.mul(FX).add(worldXZ.y.mul(FZ)).toVar();
  const across = worldXZ.x.mul(-FZ).add(worldXZ.y.mul(FX)).toVar();
  const alongF = fineXZ.x.mul(FX).add(fineXZ.y.mul(FZ)).toVar();
  const acrossF = fineXZ.x.mul(-FZ).add(fineXZ.y.mul(FX)).toVar();
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
  // Three code values. So `(chunkN.r - 0.5) * 1.3` was displacing the cell tile
  // by about one and a half CENTIMETRES — the warp this comment described has
  // never actually run, which is most of why the 1.7 m tile was legible as a
  // repeat. B and A are value channels spanning most of the byte range, so the
  // same constant now buys the ~0.5 m it was always meant to.
  const cellUV = fineXZ.add(vec2(chunkN.b.sub(0.41), chunkN.a.sub(0.41)).mul(CELL_WARP)).toVar();
  const cellN = texture(detailTex, cellUV.div(CELL_TILE)).toVar();
  // Two warps, at two scales, before either streak tile is sampled. One warp is
  // not enough: warping only the fine tile leaves the COARSE ribbons on a fixed
  // spacing, and the comb of fingers running down a wave face comes out evenly
  // spaced — a picket fence at whitecap scale. Displacing the across-axis by an
  // isotropic several metres first makes the ribbon spacing itself wander.
  const across0 = acrossF.add(chunkN.b.sub(0.44).mul(WARP_COARSE)).toVar();
  const sA = texture(detailTex, vec2(alongF.add(drift).div(STREAK_LONG), across0.div(STREAK_WIDE))).toVar();
  const acrossW = across0.add(sA.b.sub(0.44).mul(WARP)).toVar();
  const sB = texture(detailTex, vec2(alongF.add(drift.mul(1.7)).div(STREAK_LONG2), acrossW.div(STREAK_WIDE2))).toVar();
  const gapA = texture(detailTex, vec2(along.div(GAP_A_LONG), across.div(GAP_A_WIDE))).toVar();
  const gapB = texture(detailTex, vec2(along.div(GAP_B_LONG), across.div(GAP_B_WIDE))).toVar();

  // --- the three ages -------------------------------------------------------
  // Unioned over the cascades coarse enough to resolve foam; the finest one is
  // 9 cm texels, which is pure speckle at any viewing distance. The maps are
  // mip-filtered (see mapTexture in maps.js), so these reads average correctly
  // as the footprint grows instead of picking one random texel.
  //
  // The maps are indexed by the undisplaced grid coordinate, which is the
  // Lagrangian label of the water particle — so the foam in them already moves
  // with the water it formed on, and the wave train sliding through that label
  // field already leaves the aged channels behind the crest. What it does not
  // do is REACH: a fold is a hairline, and one hairline per crest is not a
  // whitecap. The tail taps below are that reach, a directed dilation along the
  // heading rather than a symmetric blur, so the result is a ribbon pointing the
  // right way instead of a dot with a halo.
  const cap = float(0).toVar();
  const trail = float(0).toVar();
  const lace = float(0).toVar();
  // one shared lateral kink so the ribbons are not dead straight
  const kink = sA.a.sub(0.44).mul(1.6).toVar();
  cascades.forEach((c, i) => {
    if (!c.foamMap || i >= cascades.length - 1) return;
    const L = lengthScales[i];
    const uv0 = worldXZ.add(vec2(kink.mul(-FZ), kink.mul(FX))).div(L).toVar();
    const texelM = L / c.N;
    // The mip level this pixel needs, worked out by hand rather than left to
    // the hardware, so the deliberate blur below can max() against it instead
    // of fighting it: never sharper than the footprint allows, never coarser
    // than the spread asks for.
    const lod = log2(footprint.div(texelM)).max(0).toVar();
    const spread = Math.max(0, Math.log2(SPREAD_M / texelM));
    const lodS = lod.max(float(spread)).toVar();

    // Sharp tap, and the same tap read from a coarser mip: the level is a
    // lateral spread radius in metres, which is how a hairline fold becomes
    // something with a body. The cap keeps most of the sharp read — under a
    // fifth is halo — because past that the near-field crest is a blurred sheet
    // and no amount of carving downstream puts an edge back on it.
    const cw = CASCADE_W[i] ?? CASCADE_W[CASCADE_W.length - 1];
    const f = texture(c.foamMap, uv0).level(lod).mul(cw).toVar();
    const fb = texture(c.foamMap, uv0).level(lodS).mul(cw).toVar();
    cap.assign(max(cap, mix(f.x, saturate(fb.x.mul(CAP_GAIN)), float(CAP_HALO))));
    trail.assign(max(trail, max(f.y, fb.y.mul(TRAIL_GAIN))));
    lace.assign(max(lace, max(f.z, fb.z.mul(LACE_GAIN))));

    for (let k = 1; k < TAIL_D.length; k++) {
      const off = vec2((FX * TAIL_D[k]) / L, (FZ * TAIL_D[k]) / L);
      const g = texture(c.foamMap, uv0.add(off)).level(lodS).mul(cw).toVar();
      trail.assign(max(trail, g.y.mul(TAIL_W[k] * TRAIL_GAIN)));
      lace.assign(max(lace, g.z.mul(TAIL_W[k] * LACE_GAIN)));
    }
  });

  // --- where the lip is -----------------------------------------------------
  // Two taps per wave-carrying cascade: the displacement at this pixel's own mip
  // level, and the same fetch from whatever level averages it over LIP_RADIUS
  // metres. Their difference is how far this point stands proud of the sea
  // immediately around it. See LIP_RADIUS.
  const prom = float(0).toVar();
  cascades.forEach((c, i) => {
    if (i > 1 || !c.displacement) return;
    const L = lengthScales[i];
    const texelM = L / c.N;
    const lodD = log2(footprint.div(texelM)).max(0).toVar();
    const blur = Math.max(0, Math.log2(LIP_RADIUS / texelM));
    const here = texture(c.displacement, worldXZ.div(L)).level(lodD).y.toVar();
    const wide = texture(c.displacement, worldXZ.div(L)).level(lodD.max(float(blur))).y;
    prom.addAssign(here.sub(wide));
  });
  const lip = saturate(prom.div(LIP_DEPTH)).toVar();

  // --- carves ---------------------------------------------------------------
  // Each carve is a multiplier on the MASK, never on the colour, and each one is
  // allowed to go well below 1 so it can punch a hole rather than only dim an
  // edge. Contrast is everything here: a carve whose swing is small compared
  // with the gap between the mask and the threshold only nibbles the rim — the
  // interior still passes at full opacity and the cap comes out as a smooth
  // white sheet with a soft edge, which is precisely the ice cornice.
  //
  // None of them is faded with distance by hand. The detail texture is
  // mipmapped, so each tap gives up its structure and returns its own mean
  // exactly when the pixel stops being able to resolve it — cells at ~0.4 m of
  // footprint, chunk at ~1.6 m, the group masks not until ~12 m. That is a
  // three-stage band-limit for free, and it is why the far field can be read
  // linearly below without also going flat.
  // The streak carve gets the bubble-cell tile multiplied into it for the same
  // reason the cap's does. Built from the three stretched tiles alone it is a
  // smooth, low-frequency ribbon field, and a smooth ribbon painted at 60%
  // opacity down a wave face is an airbrushed stroke — it reads as a boat wake,
  // not as foam. The cell tile puts sub-metre holes through it so the ribbon is
  // made of bubbles.
  // Chunk sets whitecap-sized bites out of the cap; cells punch bubble-sized
  // holes right through it. Without the second one a cap is a smooth dollop
  // with a Gaussian falloff — an ice cornice — however good the silhouette is.
  const cells = saturate(cellN.a.mul(1.45).add(cellN.b.mul(0.5)).add(sB.b.mul(0.6)).sub(0.45)).toVar();
  const streak = sA.b.mul(1.25).add(sA.a.mul(0.6)).add(sB.a.mul(0.8)).sub(0.15)
    .mul(mix(float(1.0), cells, float(0.55))).toVar();
  const chunk = chunkN.b.mul(1.55).add(chunkN.a.mul(0.85)).sub(0.20)
    .mul(mix(float(1.0), cells, float(0.70))).toVar();
  // The lacy carve mixes three different frequencies *and* two different
  // anisotropies on purpose. One dominant frequency at a threshold is a
  // halftone screen — evenly spaced round dots, the most synthetic thing foam
  // can do. Irregular hole sizes are what makes it read as a dissipating web.
  const web = cellN.a.mul(0.70).add(cellN.b.mul(0.55)).add(sB.a.mul(0.55)).add(sA.b.mul(0.45)).sub(0.25).toVar();

  // --- where foam is allowed to live ---------------------------------------
  // The cascades are statistically independent, so the chop's folds land
  // wherever they like, trough included — and a thick white cap sitting in a
  // trough is the single most obvious tell of procedural foam. Height above
  // mean sea level is the missing correlation, and it is taken here rather than
  // in maps.js because only here is the TOTAL height of all cascades known: a
  // single cascade's own height says nothing about whether the point is on a
  // crest of the sea.
  // The height itself is JITTERED by noise before either ramp, and that is not
  // a detail. A ramp on raw height draws an iso-height contour across the sea,
  // and the bottom of every foam patch lands on it: a dead-straight horizontal
  // cut under a white sheet, which is the strongest single reason foam reads as
  // a glacier lip. Perturbing the height by half a metre of two different noise
  // frequencies turns that contour into a ragged, fractal boundary.
  const hJit = sA.b.sub(0.44).mul(0.62).add(cellN.b.sub(0.44).mul(0.38)).toVar();
  const hN = positionWorld.y.div(WAVE_SCALE).add(hJit).toVar();
  const crest = smoothstep(float(0.04), float(0.72), hN).toVar();
  // Trails live on the wave, not in the sea generally: a streak is something a
  // crest dragged down its own face, so it stops well above mean water. This
  // gate is much tighter than it was — at smoothstep(-0.85, 0.45) it passed
  // essentially the whole field and the trail channel blanketed the foreground.
  const upper = smoothstep(float(-0.40), float(0.48), hN).toVar();
  // ...and lace is the one regime that BELONGS in a trough, because it got there
  // by sliding off the crest that made it. Shallowest gate of the three, but it
  // still closes on the floor of a deep hollow.
  const old = smoothstep(float(-1.00), float(0.05), hN).toVar();
  // The second half of the height correlation: a wave breaks forward, over its
  // own leading face. That face is the one whose normal leans DOWNWIND, so a
  // cheap dot against the heading separates the front of the crest (where the
  // cap belongs, hard-edged at the lip) from the back (where only the older
  // channels should reach). It also breaks the cap's silhouette up on its own,
  // because the normal carries the ripple detail.
  const lean = dot(vec2(N.x, N.z), vec2(FX, FZ)).toVar();
  const face = saturate(lean.mul(2.6).add(0.55)).toVar();

  // Two group masks at a deliberately non-harmonic ratio. They are the coarsest
  // carves in the file and the only ones the horizon can still resolve, so they
  // are also what decides whether distant foam reads as separate dashes or as a
  // continuous ribbon.
  //
  // They are applied as FLOORED mixes rather than as bare multipliers, and that
  // matters now in a way it did not before: with the generation in maps.js cut
  // back to a rare, short, full-amplitude event, the whitecaps arrive from the
  // sim already scattered — separate patches sitting on separate crests with
  // clean water between them (DEBUG=1 shows it directly). Multiplying that by a
  // pair of masks that each close to zero deletes most of what is left, which is
  // exactly what happened: the sea came out with three whitecaps on it. Their
  // job here is to VARY the foam, not to ration it.
  const patch = saturate(gapA.b.mul(2.5).sub(0.45))
    .mul(saturate(gapB.b.mul(2.5).sub(0.45))).toVar();
  // gapA's other channel runs at twice the frequency, so this one segments the
  // crest LINE — features about twelve metres apart along it, which is whitecap
  // spacing. A second, finer term rides on it so consecutive caps are not all
  // the same length: one frequency through a threshold gives a dashed line, and
  // a dashed line is not a sea.
  const crestGap = saturate(gapA.a.mul(2.2).add(sA.b.mul(0.45)).sub(0.75)).toVar();

  // How much of the sea is foam is decided HERE, by the two group masks, not by
  // the detector thresholds in maps.js. That is deliberate and it is the only
  // arrangement that survives a change of sea state: the fold statistics of an
  // FFT ocean move by an order of magnitude between a 9 m/s and a 16 m/s wind,
  // so a detector tuned to fire on "the top 3%" at one wind fires on half the
  // surface at another and the ocean turns to milk. These masks are geometry,
  // not physics — they close on a fixed fraction of the sea whatever the waves
  // are doing — so the physics is left to decide only WHERE inside the allowed
  // stretches a break lands, which is the thing it is actually good at.
  const capA = cap.mul(mix(float(0.02), float(1.0), crest))
    .mul(mix(float(0.30), float(1.0), lip))
    .mul(mix(float(0.45), float(1.0), face))
    .mul(mix(float(0.10), float(1.0), crestGap))
    .mul(mix(float(0.45), float(1.0), patch)).toVar();
  const trailA = trail.mul(mix(float(0.02), float(1.0), upper))
    .mul(mix(float(0.60), float(1.0), lip))
    .mul(mix(float(0.55), float(1.0), face))
    .mul(mix(float(0.20), float(1.0), crestGap))
    .mul(mix(float(0.40), float(1.0), patch)).toVar();
  // Old foam is suppressed under fresh foam so the two regimes read as different
  // ages rather than as one mask at two opacities.
  const laceA = saturate(lace.sub(cap.mul(0.45))).mul(mix(float(0.15), float(1.0), old))
    .mul(mix(float(0.50), float(1.0), crestGap))
    .mul(mix(float(0.30), float(1.0), patch)).toVar();

  // --- coverage -------------------------------------------------------------
  // Masks normalised to a nominal peak of 1 so one threshold means the same
  // thing to all three regimes. The threshold's job is NOT to decide how much of
  // the sea is foam — that is set upstream, by how rare a break is in maps.js,
  // and trying to do it here instead is what produced a sea of evenly sized grey
  // plates. Its job is only to give the mask an edge.
  const capM = capA.mul(chunk).mul(CAP_NORM).toVar();
  const trailM = trailA.mul(streak).mul(TRAIL_NORM).toVar();
  const laceM = laceA.mul(web).mul(LACE_NORM).toVar();

  const T = shading.foamThreshold.mul(THRESH_SCALE).toVar();
  const soft = float(1).div(max(shading.foamScale, float(0.2))).toVar();

  // Near: a threshold, for a hard fractal edge. Far: the same mask read
  // LINEARLY, because past FAR_FOOT the foam map's mip has already replaced it
  // with the local area fraction of foam and a step function on an area fraction
  // is what paints the white bar along the horizon. The carves have converged on
  // their means by then, so the linear branch carries a gain to put the average
  // back where the thresholded branch had it.
  const cut = (m, edge) => mix(
    smoothstep(T, T.add(soft.mul(edge)), m),
    saturate(m.mul(FAR_GAIN)).mul(FAR_CEIL),
    far,
  );
  const capC = cut(capM, EDGE_CAP).mul(OP_CAP).toVar();
  const trailC = cut(trailM, EDGE_TRAIL).mul(OP_TRAIL).toVar();
  const laceC = cut(laceM, EDGE_LACE).mul(OP_LACE).toVar();

  const thick = saturate(capC.div(OP_CAP)).toVar();
  const raw = max(capC, max(trailC, laceC)).toVar();

  // Erosion, punched into the finished OPACITY rather than into the mask — see
  // ERODE_MIN. It never closes to zero: a hole in foam still has foam round its
  // edges and water you can see the sky in.
  // ...and it is deliberately shaped as MOSTLY-OPEN WITH HOLES rather than as a
  // smooth multiplier. A smooth 0.1-to-1 noise with a mean near 0.65 does not
  // punch holes in a raft, it makes the whole raft two-thirds transparent — a
  // cap that is everywhere translucent reads as spray or as dust on the water,
  // never as the solid white mass a breaking crest actually is. Run through a
  // steep smoothstep it sits at 1 over most of its area and falls off a cliff in
  // the lowest fifth, so the raft is opaque and the sea shows through in
  // discrete gaps.
  const erode = smoothstep(float(0.16), float(0.56), saturate(
    cellN.a.mul(1.30).add(cellN.b.mul(0.70)).add(sB.a.mul(0.80)).add(chunkN.a.mul(0.60)).sub(0.85),
  )).toVar();
  const viewDist = length(cameraPosition.sub(positionWorld)).toVar();
  const horizon = mix(float(1.0), float(HORIZON_CEIL), max(
    saturate(footprint.div(HORIZON_FOOT)),
    saturate(viewDist.div(HORIZON_DIST)),
  )).toVar();
  const coverage = raw.mul(mix(float(ERODE_MIN), float(1.0), erode)).mul(horizon).toVar();

  // --- shading --------------------------------------------------------------
  const fc = vec3(shading.foamColor).toVar();
  const hue = fc.div(max(max(fc.x, fc.y), fc.z).max(1e-3));
  // Thin foam is not darker, only very slightly cooler — the luminance of the
  // tint below is 0.99, so nothing about age costs value. Age is expressed by
  // the opacity ceilings above and by nothing else.
  const albedo = mix(vec3(1, 1, 1), hue, float(FOAM_TINT))
    .mul(mix(vec3(0.93, 0.97, 1.05), vec3(1, 1, 1), thick)).toVar();

  const wrap = saturate(dot(N, shading.sunDir).add(WRAP).div(1 + WRAP)).toVar();
  const sun = vec3(shading.sunColor).mul(wrap.mul(wrap.mul(0.35).add(0.65)).mul(SUN_GAIN));
  const amb = mix(vec3(shading.horizon), vec3(shading.zenith), float(0.35)).toVar();
  const ambL = dot(amb, vec3(0.2126, 0.7152, 0.0722)).toVar();
  const sky = mix(amb, AMBIENT_COOL.mul(ambL), float(AMBIENT_NEUTRAL)).mul(SKY_GAIN);
  const skyVis = mix(float(SKY_VIS_MIN), float(1.0), old).toVar();
  const lit = albedo.mul(sun.add(sky.mul(skyVis))).toVar();

  // A few percent of grain so a thick cap is a raft of bubbles with pockets in
  // it rather than a smooth card of cotton wool. A few percent and no more:
  // this is the exact term that turns foam grey if it is allowed to do the
  // shaping. The break-up is the coverage's job, not the colour's.
  // Bounded on purpose: the sunlit crown already sits where ACES is nearly flat,
  // so an unbounded multiplier here is the one term that can push a channel to a
  // literal 255 and weld a cap to the sky behind it. +-8% is enough to read as
  // bubble texture and cannot clip.
  const grain = saturate(cellN.a.mul(0.20).add(sB.b.mul(0.10)).add(0.85))
    .clamp(0.88, 1.06).toVar();
  const color = lit.mul(grain);

  if (DEBUG === 1) return { coverage: float(1), color: vec3(cap, trail, lace) };
  if (DEBUG === 2) return { coverage: float(1), color: vec3(capA, trailA, laceA) };
  if (DEBUG === 3) return { coverage: float(1), color: vec3(capC, trailC, laceC) };
  if (DEBUG === 4) return { coverage: float(1), color: vec3(patch, crestGap, far) };
  if (DEBUG === 5) return { coverage: float(0), color: vec3(1) };
  return { coverage, color };
}
