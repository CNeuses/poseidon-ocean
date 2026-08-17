import { MeshBasicNodeMaterial, DoubleSide } from 'three/webgpu';
import {
  Fn, positionGeometry, positionWorld, cameraPosition, vec2, vec3, vec4, float,
  texture, normalize, reflect, refract, dot, max, min, pow, mix, saturate, abs, smoothstep,
  length, fwidth, sqrt, exp, log2, PI,
} from 'three/tsl';
import { skyColor } from './sky.js';
import { foamShading } from './foamShading.js';
import { diffusionAttenuation } from './water.js';

// --- look constants, local to this file -----------------------------------
// params.js and the GUI belong to the integration pass; only the palette and
// the two strength knobs arrive as uniforms. Everything that shapes the water
// itself is tuned here so the whole model stays readable in one place.

// --- the air/water interface, done once and used three times ---------------
// Refractive index of sea water at 35 PSU, 589 nm (Quan & Fry 1995). Every
// number below falls out of it: F0 = ((n-1)/(n+1))^2 = 0.0211, and the critical
// angle for the water->air direction is asin(1/n) = 48.27 degrees.
const N_WATER = 1.34;
const ETA_AW = 1 / N_WATER; // air -> water
const ETA2 = ETA_AW * ETA_AW;

// The exact unpolarised Fresnel reflectance, not Schlick, and the reason is the
// water->air direction rather than fussiness about the third decimal.
//
// Going water->air there is a critical angle, and past it reflectance is
// exactly 1: a camera under the surface sees a mirror everywhere outside a
// 48-degree cone. Schlick has no critical angle at all — fed the water-side
// cosine at 48.27 degrees it returns 0.025, which is a factor of FORTY out, and
// it is why the old backface path could not do anything but crossfade to a flat
// swatch. There is no branch here: past the critical angle k goes negative, the
// max() pins cosT to 0, and both r terms go to +-1 on their own, so the
// function returns exactly 1 by construction.
//
// Above water it also removes a hack. Schlick^5 under-reports at 45-60 degrees
// and over-reports past 75, and the file was carrying `mix(5, 3.4, rough)` on
// the exponent to bend the curve back — which made the grazing error worse
// (+30% at 78.8 degrees at rough 0.4) in exactly the far field where the
// hold-back below then had to pay for it.
//
// eta is the ratio n_incident / n_transmitted: ETA_AW looking down into water,
// N_WATER looking up out of it.
const fresnelDielectric = (cosI, eta) => {
  const ci = abs(cosI).toVar();
  const k = float(1).sub(eta.mul(eta).mul(float(1).sub(ci.mul(ci)))).toVar();
  const ct = sqrt(max(k, float(0))).toVar();
  const ec = eta.mul(ci).toVar();
  const et = eta.mul(ct).toVar();
  const rs = ec.sub(ct).div(max(ec.add(ct), float(1e-5))).toVar();
  const rp = et.sub(ci).div(max(et.add(ci), float(1e-5))).toVar();
  return rs.mul(rs).add(rp.mul(rp)).mul(0.5);
};

// Henyey-Greenstein, peak-normalised so that g sets the WIDTH of the lobe and
// never its level — p(c)/p(1) = ((1-g)^2 / (1 + g^2 - 2gc))^1.5, and r*sqrt(r)
// is pow(r, 1.5) exactly. Peak-normalising is what lets the knee downstream keep
// its tuning when g moves.
const hgForward = (g, cosT) => {
  const r = float((1 - g) * (1 - g))
    .div(max(float(1 + g * g).sub(cosT.mul(2 * g)), float(1e-4))).toVar();
  return r.mul(sqrt(r));
};

// Band-limiting. Every map here is a StorageTexture with generateMipmaps left
// on, and three flags the binding needsMipmap on each compute write, so a mip
// chain *does* exist — the default LinearFilter min filter simply never reaches
// for it. An explicit level(log2(footprint / texel)) does, which is the correct
// average over the pixel rather than the bilinear lottery that used to print
// per-pixel static. That moves the aliasing problem out of the fade entirely.
//
// FEATURE_DIV is therefore no longer an anti-aliasing device; it is only the
// Toksvig bookkeeping — the point past which a cascade has been averaged away
// hard enough that what is left of it belongs in the roughness budget instead
// of in the normal. At 160 it was deleting cascade 2 (14 m patch, 5.5 cm texels)
// at a footprint of 8.75 cm, which a deck-height camera already exceeds in the
// FOREGROUND at a grazing angle: the near water lost every ripple it had and
// the mid-field showed nothing but cascade 1's 30 cm texel grid as hard-edged
// cells. At 45 cascade 2 survives out to a 31 cm footprint — several metres of
// real chop in front of the lens — and the mip chain keeps it clean past that.
const FEATURE_DIV = 45;
const ROUGH_FLOOR = 0.075; // never a perfect mirror, so no one-pixel spark
const ROUGH_FROM_VAR = 0.60; // sub-pixel slope variance -> lobe width
// A mip-filtered tap *is* the local mean slope, so it tends to zero on far
// water and takes the variance estimate with it. The footprint term is what
// carries roughness out there instead, so it has to bite much sooner than it
// did when the aliased tap was propping the variance up.
const ROUGH_FROM_FOOT = 0.30; // far water is rough even where it samples flat
const ROUGH_FOOT_SCALE = 0.55; // 1/m — footprint at which that term saturates
const ROUGH_MAX = 0.60;
const REFL_RELAX = 1.7; // how fast the mirror ray falls back to flat-water with roughness
// ...but never ALL the way. REFL_RELAX * ROUGH_MAX came to 1.02, so past a
// kilometre every pixel relaxed to the same flat-water ray, reflected the same
// point of sky and the far band went dead flat. Capping the relaxation leaves a
// quarter of the geometric normal in the ray at any roughness, which is what
// keeps swell modulation visible out to the horizon.
const REFL_RELAX_MAX = 0.74;
const GLINT_CLAMP = 0.9; // peak-radiance compression of the reflected sun
const REFL_FILTER = vec3(0.88, 1.00, 1.10); // the sea's own filter on a rough reflection
const REFL_DESAT = 1.8; // roughness at which the reflection is fully lobe-averaged
// How much chroma survives that averaging, and how much of the pixel the
// averaging is allowed to claim. Both were far too aggressive: saturate(rough *
// 1.8) hits 1 at rough 0.556 and ROUGH_MAX is 0.60, so essentially the whole
// frame ran at the 0.42 chroma floor and the sea came out R=G=B grey looking
// into the sun. The product (1 - CAP*(1 - CHROMA)) is the real number — the
// worst-case chroma anywhere on the water — and it now sits at 0.77.
const REFL_CHROMA = 0.62;
const REFL_DESAT_CAP = 0.60;
const REFL_GREY = vec3(0.84, 1.00, 1.21); // the cool grey it averages toward, never khaki
// The sky's haze band is a strongly saturated sand-orange directly under the
// sun's azimuth, and a deck camera reflects almost nothing else: every ripple
// facet in the near field aims within a couple of degrees of the line. A raw
// mirror tap of that band is a tan film lying over green water, which grades
// out as olive — the one colour a tropical sea must never be. A facet integrates
// over several degrees of the steepest gradient in the whole sky, and the mean
// of that band is far cooler than its peak, so cooling the reflection in
// proportion to how horizontal the reflected ray is fixes the film at source
// without muting the reflection anywhere else in the dome.
//
// Keyed on the reflection's own WARMTH rather than only on where it came from.
// An elevation gate alone is too blunt: the sand runs a long way up the sky
// under the sun's azimuth, and the cloud bases up there are warm too, so a gate
// tight enough to spare the blue overhead left a brown band across the sunward
// half of any high shot. (r - b) is positive only for the reflections that
// actually make olive on green water, and it is exactly zero for blue sky and
// for white cloud, so this can be driven hard without touching either.
const REFL_BAND = vec3(0.86, 0.99, 1.16); // the silver it is pulled toward
const REFL_BAND_DESAT = 0.80; // ...and how much of that warmth's chroma it costs
const REFL_WARM_GAIN = 3.5; // how sharply (r - b) counts as "warm"
const REFL_BAND_ELEV = 13.0; // 1/rad — the extra bite inside the haze band itself

// Sub-grid ripple. detailTexture.js packs a slope into RG, but it packs it as
// (-h' * 3 * 0.5 + 0.5) into 8 bits, and the fbm's own gradient is small enough
// that the whole field lands inside 128 +/- 3 — three usable code values, whose
// plateaus print as a hard-outlined cellular iso-contour wherever a sharp sky
// reflection runs over them. The height channels do not have that problem: B
// and A span most of the byte range. So the slope is reconstructed here by
// differencing the height instead, which costs two extra taps per scale and
// gives a continuous gradient at full precision. B is the base octave and A the
// same field at twice the frequency, so each 3-tap set carries two ripple
// scales, and two sets an octave apart carry four.
const RIPPLE_TILE = 0.55; // 1/m — ~1.8 m tile, so features are 45 cm and 22 cm
const RIPPLE_STEP = 2.5 / 512; // finite-difference offset, in tile units
const RIPPLE_GAIN = 2.6; // slope per unit of differenced height, times shading.detail
const RIPPLE_FINE = 0.7; // weight of the second (A-channel) octave
const RIPPLE_FOOT = 0.22; // m of footprint at which the coarse octave is filtered out
const RIPPLE_OCT = 4.0; // the second octave's scale against the first (~11 cm and 6 cm)
const RIPPLE_OCT_AMP = 0.85; // ...and its weight

// Body. Mean sea level is y = 0, so height above it is a direct read on where
// a point sits between trough floor and crest lip. Every ramp built on it is
// *asymptotic*, never clamped: a saturate() here paints a hard iso-height
// contour across the sea and the fill between contours reads as spilled paint.
const WAVE_SCALE = 2.6; // metres of height that read as "a full crest"
const HEIGHT_SOFT = 1.3; // softness of the height ramp; larger = lazier gradient
// The trough->body ramp, as its own asymptotic curve rather than a smoothstep
// over `lift`. smoothstep(0, 0.50, lift) reached 1 at mean sea level, so the
// ENTIRE upper half of the sea was one flat body colour with a hard iso-height
// contour under it — the biggest single source of the paint-by-numbers look.
// These two put the ramp at 0.11 in a 5 m trough, 0.60 at 2.6 m below mean,
// 0.90 at mean level and 0.94 on a crest: same value range, no ceiling.
const BODY_BIAS = 1.07;
const BODY_SOFT = 0.27;
// Red is small but never zero. At literally 0.002 the sea comes out of ACES as
// printer cyan — a dye, not a body of water — and the highlights have nothing
// to roll toward. The deep end is biased GREEN rather than blue as well: a
// shadowed tropical trough is bottle-green, not navy.
const TROUGH_COLOR = vec3(0.0080, 0.0190, 0.0160); // near-black bottle green, shadowed floor
const ABYSS_COLOR = vec3(0.0170, 0.0440, 0.0400); // the deep end of the mass wander
const BODY_COLOR = vec3(0.0280, 0.0930, 0.0995); // deep blue-green: the bulk of the sea
const CREST_COLOR = vec3(0.1650, 0.5200, 0.4700); // thin sunlit water at the lip
const BODY_BLUE = vec3(0.0210, 0.0720, 0.1150); // the bluer end of the mass wander
const SHALLOW_MAX = 0.88; // crest water is never *pure* scatter paint
// Wander of the water mass' own colour. The tile sizes matter as much as the
// amount: at 0.0016 (a 625 m tile) the near field crossed three or four code
// values of an 8-bit channel, and the saturate() that used to sit on the end of
// this turned each of those into a flat plateau of one body colour with a
// one-pixel staircase boundary — continents on a map. Shorter tiles traverse
// far more texels per frame and the ramp below is asymptotic, so neither the
// quantisation nor the ramp can print a contour now.
const MASS_SCALE = 0.0055; // 1/m — ~180 m: the slow one, depth and turbidity
const MASS_NEAR = 0.0210; // ...and ~48 m, so near water varies within one wave group
const MASS_FINE = 0.1100; // ...and ~9 m, so ONE swell face in the foreground is not flat
const MASS_AMOUNT = 1.35; // how hard that wander swings deep <-> body
const MASS_SOFT = 0.55; // knee of the asymptotic ramp it runs through
const MASS_HUE = 0.0030; // 1/m — a third, independent axis: blue-green <-> green
const MASS_HUE_AMOUNT = 0.50;
const SKY_VIS_MIN = 0.34; // a trough only sees a slot of sky between its walls
// Sky-only light on faces turned away from the sun. This was a cool blue,
// which is correct under a midday sky and wrong under this one: the measured
// hemisphere irradiance of the current panorama is 0xa3939b, a warm mauve-grey
// (tools/skylight.mjs). A blue shadow under a golden sky is the second most
// obvious tell after a white sun. Derived, not picked — the ambient hue
// renormalised to the luminance the old constant had (0.487), so this changes
// only the COLOUR of shadow and never how dark it is. Recompute if the sky
// changes: hue = ambientLinear / luminance(ambientLinear) * 0.487.
const SHADOW_TINT = vec3(0.572, 0.458, 0.519);

// Wave-group occlusion. Everything else in the value chain — the colour ramp,
// the sky visibility, the wrapped sun — is keyed off the same two numbers
// (height above mean, and how far the normal faces up), so on a near-flat
// foreground patch they all pin to constants together and the hero wave face
// comes out a 2%-range flat card. Cascade 0 carries the swell alone, which is
// the blurred sea level the rest of the field rides on: reading it separately
// gives a term that is *independent* of both, so a patch sitting low inside a
// swell group goes dark whatever its own height or facing is doing.
const GROUP_SCALE = 3.4; // metres of swell height that read as a full group
const GROUP_SOFT = 1.0;
const GROUP_DARK = 0.40; // how much value the bottom of a group loses
const CHOP_LIFT = 0.55; // ...and how much a chop crest standing on it wins back

// Subsurface scatter — the Sea of Thieves cue, and the one this file kept
// getting inside out. Thickness used to be (height above mean) AND (1 - facing
// up): both of those switch hard at the crest ridge and both are wrong about
// where a lip is, so the product printed a knife-edged mossy patch on crest
// backs that read as an island breaking the horizon.
//
// Thickness is now measured, not inferred: the displacement map is read twice,
// once sharp and once from the mip level that averages it over CREST_RADIUS
// metres of sea, and the difference is how far this point stands proud of the
// water immediately around it. That is largest at a lip, zero on a plateau and
// negative in a trough, it is a *local* comparison so it cannot draw an
// iso-height contour anywhere, and because both taps are of a smooth field it
// runs as a continuous gradient down the face instead of snapping at a ridge.
const CREST_RADIUS = 5.0; // metres of sea the "local mean height" probe averages over
const SSS_DEPTH = 0.80; // metres above that local mean that read as fully thin
const SSS_GAIN = 1.7; // peak input to the knee; low enough that it never plateaus
const SSS_MAX = 0.75;

// Effective asymmetry of the forward-scatter lobe. A single droplet of sea
// water is g ~ 0.92, but what is being looked through is a volume many
// scattering events deep and multiple scattering smears that lobe very wide.
// 0.62 puts the half-maximum 21 degrees off the sun's bearing, so the whole
// sunward side of a crest carries the glow as a gradient rather than a ring
// around the sun's exact azimuth.
const SSS_G = 0.62;
// The path light actually crosses, in METRES, at the thinnest and thickest ends
// of the lip detector. This pair is the thing that replaced a pair of authored
// colour swatches: exp(-mu*d) at 0.6 m comes out a warm near-white and at 3 m a
// jade green off the same coefficients, so the gold tip and the green run down
// the face are one exponential sampled twice instead of two constants and a
// blend factor between them.
//
// Note the direction. crestRel measures how far a point stands PROUD of the
// water around it, so it is largest where the water is THINNEST — the path
// shortens as thinP rises, which is why LIP_TIP is the second argument to the
// mix and not the first. Getting this backwards inverts the whole cue and is
// the single easiest mistake to make here.
const LIP_TIP = 0.60;
const LIP_BASE = 3.00;

// Diffusion attenuation of the two water types the mass wander interpolates
// between, 1/m per RGB, computed from published (a, b) tables — see water.js.
// These are the numbers that used to be the hand-picked SSS_COLOR swatch, and
// unlike a swatch they carry a length, so the glow can change hue with
// thickness instead of only with strength.
const MU_CLEAR = vec3(...diffusionAttenuation('1C'));
const MU_TURBID = vec3(...diffusionAttenuation('3C'));

// Single scattering — the missing half of the water's brightness, and the one
// that fills the value hole. SSS above is view-DEPENDENT: it only fires where
// the camera is looking down the forward-scattered ray, which is a small part
// of any frame. Everything else in the body chain only ever ATTENUATES a
// palette swatch whose brightest channel is 0.44 linear, so the frame had
// nothing at all between the near water's near-black teal and the far water's
// 90%-sky wash — the exact band where Sea of Thieves' turquoise lives.
//
// This term is view-independent: sun on the water, gated on the same measured
// thinness, so it lights the whole sunlit shoulder of every swell rather than
// only a back-lit lip. It is what puts real energy into 0.15-0.45 linear.
// Also a transmittance, but NOT the same one as SSS_COLOR, and the difference
// is path length. SSS is light crossing a sub-metre lip once. This is light
// that goes down INTO the water, turns around and comes back out — several
// metres of round trip, over which red really is gone. So red stays near the
// floor here even though the light is warm.
//
// Lifting it to 0.165 to "match the golden sun" was a mistake worth recording:
// this term is ADDED on the whole sunlit shoulder of every swell, so the red
// went everywhere the sun hit and the sea turned olive — the one colour the
// original note in this file warns against. Only the lip may go gold.
const SCATTER_COLOR = vec3(0.088, 0.300, 0.282);
const SCATTER_GAIN = 1.05;
const SCATTER_BASE = 0.26; // ...at its weakest, on water that stands proud of nothing

// Reflectivity with distance. Near water at eye level is grazing enough that
// pure Schlick hands the whole frame to the sky and the sea turns grey-tan;
// SoT keeps its body colour far out to sea and only lets the last kilometre go
// to mirror. REFL_NEAR is that hold-back, GRAZE the release.
// Both numbers were far too eager. At 1200 m / 0.52 the far HALF of every frame
// ran at 90% reflection or better, which is why a kilometre of sea was
// indistinguishable from the sky above it. The ceiling is the real guarantee:
// however grazing the angle and however far the water, a fixed share of the
// pixel is the sea's own colour, so the horizon band stays blue-green and can
// never resolve to the same putty as the sky.
// ...and it went up by a third when the Schlick approximation below was
// replaced with the exact Fresnel. Exact is 0.771x what `mix(5, 3.4, rough)`
// returned at 78.8 degrees and roughness 0.4, which is where a deck camera
// spends most of its frame, so holding the same amount back off a smaller
// number would have taken the near field's sky with it.
const REFL_NEAR = 0.26;
const GRAZE_RANGE = 4000; // metres over which the hold-back is released
const GRAZE_AMOUNT = 0.15; // ...and how hard the far field is then pushed to sky
const REFL_CEIL = 0.86; // the sea always keeps at least 14% of itself

// Sun specular. A saturating GGX lobe: broad glitter, no fireflies.
//
// Fresnel is applied AFTER the compressor, not folded into the lobe before it.
// Multiplied in first, it scales the number the knee sees, so the hold-back
// that keeps near water from going to mirror was also deciding how bright the
// glitter path is allowed to get — and with a knee of 13 the broad moderate
// part of the path, which is the part that makes a path *read*, was compressed
// to a fifth of its value. Looking straight into the sun, the water under it
// came out the darkest water in the frame. Knee at 1.5 and Fresnel outside puts
// the wash back and lets the true peaks clip white.
//
// What the knee actually controls is CONTRAST, and the direction is not the
// obvious one. x/(x+k)*M has slope M/k near zero and ceiling M, so lowering the
// knee makes the *wash* brighter while lowering the ceiling makes the *peaks*
// dimmer — together they flatten the path into the uniform sheet of light the
// last pass shipped, where a facet 25 degrees off the half-vector returned a
// third of what a mirror facet did. Raising k and M together is what opens the
// gap: at roughness 0.35 the core now returns 3.5 and the 26-degree skirt 0.15,
// a 23:1 ratio against the 13:1 it had. That ratio is the glitter road.
const SPEC_KNEE = 3.0;
const SPEC_MAX = 4.0;
// How much of the sun's own colour the glitter gives up. It was 0.70 — the
// brightest thing in the frame was being repainted seven-tenths white — and a
// measured deck capture came back with the specular road at a ratio of
// 1.000 / 0.994 / 0.984, which is white, under a sun the sky panorama measures
// at 1.000 / 0.827 / 0.584. The light source and the light did not agree.
//
// Physically this should be near zero: a dielectric's specular reflectance is
// almost flat across the visible band, so a mirror hands back the source's hue
// unchanged. The original note ("never tan") was guarding against a real thing,
// but the wrong one — what makes a tan film on green water is the reflected
// HAZE BAND, and that has its own desaturation chain (REFL_BAND and friends)
// which this constant never touched. What this constant governs is the sun.
//
// Not zero, though. The GGX lobe is ADDED to the water rather than replacing
// it, so its dim skirts land on green: a fully saturated gold added at low
// amplitude to a green body is olive, and the skirt covers far more of the
// frame than the core. A quarter white keeps the wash neutral enough while the
// road itself carries the sun's colour, and the core clips to white on its own
// through ACES — which is what a real sun road does.
const SPEC_WHITE = 0.25;
// The lobe's own roughness is clamped at both ends, and the top clamp is the
// one that makes a road. Far water reaches ROUGH_MAX plus the distance spread,
// which is a 35-degree lobe: every facet within thirty degrees of the sun's
// bearing returned most of the peak, so a high shot came out with a tan wash
// over its whole sunward half instead of a path. 0.40 is still a 23-degree
// lobe — far too wide to alias, narrow enough to have edges. The floor stops
// the near field going razor-thin and printing single-pixel sparks.
const SPEC_ROUGH_MIN = 0.12;
const SPEC_ROUGH_MAX = 0.40;
// A glitter path widens with distance because each far pixel averages more sea.
// Roughness alone half-does that; this adds the rest explicitly, so the path
// broadens toward the horizon instead of depending on lucky facets. Kept small:
// at 0.09 the far lobe was wide enough to cover the whole frame, so there was
// no road, only daylight.
const SPEC_SPREAD_RANGE = 4000; // metres to full extra spread
const SPEC_SPREAD = 0.035; // added to roughness^2 at that range

// --- aerial perspective ----------------------------------------------------
// The water composites its OWN haze (mat.fog is off below) rather than taking
// scene.fogNode. Two reasons, and both are about the horizon line.
//
// scene.fogNode runs after the material, so it hazes the finished pixel — but
// its target is skyColor() evaluated along the view ray, which just below the
// line is by construction the same value as the sky just above it. Fog that
// converges exactly on the sky cannot draw a horizon, it can only dissolve into
// one, and every additive term the water adds on the way (glitter, foam, glow)
// then rides on top of that and pushes the last few hundred metres of sea
// ABOVE the sky it meets. Measured: sea 205 against sky 197.
//
// Owning the composite fixes both. The target is the sky just ABOVE the line
// (so it is the value the eye compares against) scaled by SEA_SINK, and it is
// applied last, after the specular and the foam, so nothing escapes it. The sea
// then lands a fixed few percent under the sky at every distance and the line
// is a clean step instead of a bloom band.
//
// Density and scale height mirror atmosphere.js so the water agrees with every
// other fogged object in the scene.
// Thinner than atmosphere.js's 1/2150, deliberately. That density is right for
// a sky dome and for spray a few metres away, but on water it puts a fifth of
// the sand-coloured haze band over sea that is only six hundred metres out, and
// a fifth of sand over green reads as a mudflat. At 1/3200 the mid-field keeps
// its own colour, the last kilometre still dissolves, and by the horizon the
// extinction is 99.8% — which is all the horizon line needs.
const HAZE_DENSITY = 1 / 3200; // extinction per metre at sea level
const HAZE_SCALE_H = 450; // metres for the haze to thin by 1/e
const HORIZON_LIFT = 0.004; // elevation the haze target is sampled at: just above the line
const SEA_SINK = 0.820; // ...and how far under the sky the sea is held
const FAR_SINK = 0.78; // extra darkening of the sea's own value with distance
const FAR_SINK_RANGE = 4000;

// A last, deliberate chroma push. ACES at exposure 1.2 desaturates hard, and
// the reflection is a large fraction of most pixels; without this the frame
// grades out cooler and greyer than any of the palette swatches suggest.
const SAT_BOOST = 1.12;

// The ocean surface: multi-cascade displacement (vertex) + water shading
// (fragment). Shading is manual (unlit MeshBasicNodeMaterial): the normal is
// mip-filtered against the pixel footprint and the detail it still loses becomes
// roughness, a roughness-aware Fresnel blends a relaxed, lobe-averaged sky
// reflection against a body whose value runs from near-black trough to lit
// crest on three independent axes (height, wave group, and the water mass' own
// wander), a measured thickness makes back-lit crest lips transmit, a GGX lobe
// draws the glitter path, and accumulated-Jacobian foam whitens breaking crests.
// A baked tiling noise texture, differenced for its gradient, adds capillary
// ripple below the finest cascade.
export function createOceanSurfaceMaterial(cascades, { lengthScales, shading, detailTex }) {
  const mat = new MeshBasicNodeMaterial();
  // Plane is authored in XY and remapped to XZ in positionNode, flipping the
  // winding — render both sides so it's visible from above too.
  mat.side = DoubleSide;
  // ...and the water composites its own aerial perspective, at the very end of
  // colorNode, so it has to opt out of scene.fogNode — see HAZE_DENSITY.
  mat.fog = false;
  // The mesh is a finite tile that gets recentred under the camera; adding its
  // offset here keeps every map lookup in true world space, so the wave field
  // stays put while the tile slides over it.
  const worldXZ = vec2(positionGeometry.x, positionGeometry.y).add(shading.originXZ);

  mat.positionNode = Fn(() => {
    const disp = vec3(0).toVar();
    cascades.forEach((c, i) => {
      disp.addAssign(texture(c.displacement, worldXZ.div(lengthScales[i])).level(0).xyz);
    });
    return vec3(positionGeometry.x.add(disp.x), disp.y, positionGeometry.y.add(disp.z));
  })();

  mat.colorNode = Fn(() => {
    const t = shading.time;
    const UP = vec3(0, 1, 0);

    // --- band-limited normal ------------------------------------------------
    // fwidth of the *sampling* coordinate is the true world footprint of this
    // pixel, grazing foreshortening included. Turn that into a mip level per
    // cascade and the tap becomes the correct average over the pixel — no
    // bilinear lottery, so no per-pixel speckle and no cascade-1 texel grid
    // printing as hard-edged cells across the mid-field. What is left of a
    // cascade once the mip chain has averaged it flat is then faded out of the
    // normal and banked as roughness: a Toksvig/LEAN trade, geometry below the
    // pixel becoming a wider lobe.
    const footprint = max(fwidth(worldXZ.x), fwidth(worldXZ.y)).max(1e-3).toVar();

    const d = vec4(0).toVar();
    const lostVar = float(0).toVar();
    cascades.forEach((c, i) => {
      const texelM = lengthScales[i] / c.N;
      const lod = log2(footprint.div(texelM)).max(0);
      const s = texture(c.derivatives, worldXZ.div(lengthScales[i])).level(lod).toVar();
      const w = saturate(float(lengthScales[i] / FEATURE_DIV).div(footprint)).toVar();
      d.addAssign(s.mul(w));
      lostVar.addAssign(s.x.mul(s.x).add(s.y.mul(s.y)).mul(float(1).sub(w.mul(w))));
    });
    const slopeX = d.x.div(float(1).add(d.z));
    const slopeZ = d.y.div(float(1).add(d.w));
    const N = normalize(vec3(slopeX.negate(), float(1), slopeZ.negate())).toVar();

    // Sub-grid capillary ripple, differenced out of the noise texture's HEIGHT
    // channels rather than read from its packed slope — see RIPPLE_TILE above
    // for why the packed slope is unusable. Three taps at one scale carry two
    // octaves, because A is the same field at twice B's frequency. Everything
    // here is capillary-scale on purpose: at a 17 m period this read as smooth
    // marbled swirls sliding over the surface — an oil slick, not water.
    // Ripples have to be smaller than the wave they ride on.
    const rippleGrad = (scale, drift) => {
      const uv = worldXZ.mul(scale).add(drift).toVar();
      const e = float(RIPPLE_STEP);
      const c0 = texture(detailTex, uv).toVar();
      const cx = texture(detailTex, uv.add(vec2(e, 0))).toVar();
      const cy = texture(detailTex, uv.add(vec2(0, e))).toVar();
      return vec2(
        cx.b.sub(c0.b).add(cx.a.sub(c0.a).mul(RIPPLE_FINE)),
        cy.b.sub(c0.b).add(cy.a.sub(c0.a).mul(RIPPLE_FINE)),
      ).toVar();
    };
    // Two octaves an octave-and-a-bit apart, each fading out at its own
    // footprint. One alone is not enough: at a metre from the lens the coarse
    // set is the only thing on the surface and it reads as slow marbled swirls
    // sliding over the water — an oil slick. The fine set is what a metre of
    // water in front of your face actually has on it.
    const fadeA = saturate(float(RIPPLE_FOOT).div(footprint)).toVar();
    const fadeB = saturate(float(RIPPLE_FOOT / RIPPLE_OCT).div(footprint)).toVar();
    const gradA = rippleGrad(RIPPLE_TILE, vec2(t.mul(0.035), t.mul(0.022)));
    const gradB = rippleGrad(RIPPLE_TILE * RIPPLE_OCT, vec2(t.mul(-0.09), t.mul(0.06)));
    const detail = gradA.mul(fadeA).add(gradB.mul(RIPPLE_OCT_AMP).mul(fadeB))
      .mul(RIPPLE_GAIN).mul(shading.detail).toVar();
    // The texture has mips so it self-filters; these fades are only so the slope
    // they lose lands in the roughness budget like everything else.
    lostVar.addAssign(shading.detail.mul(shading.detail)
      .mul(float(2).sub(fadeA).sub(fadeB)).mul(0.35));
    N.assign(normalize(N.add(vec3(detail.x.negate(), 0, detail.y.negate()))));

    // --- roughness ----------------------------------------------------------
    // Variances add, so roughness combines in quadrature: a floor that keeps a
    // minimum lobe width, the slope variance we just threw away, and a straight
    // footprint term so far water still goes broad where it happens to sample
    // flat (a calm trough at 3 km must not turn back into a mirror).
    const rough = min(sqrt(
      float(ROUGH_FLOOR * ROUGH_FLOOR)
        .add(lostVar.mul(ROUGH_FROM_VAR))
        .add(saturate(footprint.mul(ROUGH_FOOT_SCALE)).mul(ROUGH_FROM_FOOT)),
    ), float(ROUGH_MAX)).toVar();

    const V = normalize(cameraPosition.sub(positionWorld)).toVar();
    const viewDist = length(cameraPosition.sub(positionWorld)).toVar();
    const NoV = saturate(dot(N, V)).toVar();

    // The exact dielectric Fresnel — see fresnelDielectric. What was here was
    // Schlick with a roughness-dependent exponent, `mix(5, 3.4, rough)`, bending
    // the curve to fix an error Schlick makes at grazing; it overshot instead,
    // by 30% at 78.8 degrees, which is where a deck camera spends most of its
    // frame. Exact costs about nine more ALU and needs no bending.
    const fres = fresnelDielectric(NoV, float(ETA_AW)).toVar();
    // Kept before the art direction below touches it, because the sun glitter
    // has to ride on the REAL reflection coefficient. REFL_NEAR is a hold-back
    // on the *sky* — it exists so a deck camera keeps its water colour at
    // grazing angles — and multiplying the glitter by it as well meant the
    // brightest thing in the sky was reflected at a fifth strength precisely
    // where the path should be at its most intense. That is most of why looking
    // down the sun's azimuth produced a flat wash with no road in it.
    const fresSpec = fres.toVar();
    // Near water keeps only part of that mirror so the body colour survives the
    // grazing angles a deck-height camera sees everything at; the far field is
    // then released and pushed the other way, because at a kilometre the shading
    // normal is an average over square metres of slope and an averaged sheet is
    // flatter — and so more mirror-like — than the water actually is.
    const distF = saturate(viewDist.div(GRAZE_RANGE)).toVar();
    fres.mulAssign(mix(float(REFL_NEAR), float(1), distF));
    fres.assign(mix(fres, float(1), distF.mul(GRAZE_AMOUNT)));
    // ...but never all of it. A hard ceiling is the only thing that guarantees
    // the far band keeps its own hue, and it is what puts a colour difference
    // under the horizon line instead of a value difference alone.
    fres.assign(min(fres, float(REFL_CEIL)));

    // --- sky reflection -----------------------------------------------------
    // As roughness rises the mirror ray relaxes toward the flat-water direction,
    // which is what a wide lobe averages out to.
    const Rm = reflect(V.negate(), N);
    const R = normalize(mix(Rm, reflect(V.negate(), UP),
      saturate(rough.mul(REFL_RELAX)).mul(REFL_RELAX_MAX))).toVar();
    // A steep fold throws that ray *below* the horizon. Mirroring it back up
    // (abs) sends it to the zenith — and the zenith is a saturated indigo, which
    // is exactly the blue dashes that were printing on wave faces. Instead the
    // elevation is floored at the horizon while the azimuth is kept, and the ray
    // is rebuilt unit-length from the two, so a sub-horizon fold lands in the
    // same pale grazing band its neighbours already use and nothing degenerates
    // when the ray points straight down.
    // A flat floor is not enough on its own, though: it collapses EVERY
    // sub-horizon fold onto one elevation, and that elevation sits in the
    // narrow warm haze band, so a whole population of steep facets came back
    // carrying the same sand-orange and printed as olive dashes on green wave
    // faces. Folding the ray instead — max() against a shallow line through the
    // negatives — is continuous, keeps a floor at the line, and sends genuinely
    // downward rays up into the pale blue above the band where they belong.
    const azN = normalize(vec2(R.x, R.z).add(vec2(1e-4, 1e-4))).toVar();
    const elev = R.y.max(R.y.mul(-0.35).add(0.007)).toVar();
    const flat = sqrt(saturate(float(1).sub(elev.mul(elev)))).toVar();
    const refl = skyColor(vec3(azN.x.mul(flat), elev, azN.y.mul(flat)), shading).toVar();
    // Below the line there is no sky to reflect, only more water, which is far
    // darker than the haze band — fade the sample down rather than pasting a
    // bright horizon on a downward-facing fold.
    refl.mulAssign(float(1).sub(saturate(R.y.negate().mul(5)).mul(0.6)));
    // A rough facet cannot hold the mirror image of a point source: compress
    // reflected radiance above 1 in proportion to roughness, so the sky's sun
    // disc stops printing single-pixel sparks on distant water. Everything
    // below 1 (the gradient, and whatever the sky lane adds) passes untouched.
    const peak = max(max(refl.x, refl.y), refl.z);
    refl.mulAssign(float(1).div(float(1).add(max(peak.sub(1), 0).mul(rough.mul(GLINT_CLAMP)))));
    // A rough facet's "reflection" is not only the mirror: some of it is light
    // that entered, scattered a wave-width inside the water and came back out,
    // and that path is filtered by the sea. Biasing the rough end of the
    // reflection toward the water's own transmission is what stops the warm
    // horizon band printing as a tan film over mid-distance chop.
    refl.mulAssign(mix(vec3(1), REFL_FILTER, rough));
    // ...and a much harder one on whatever came out of the haze band — see
    // REFL_BAND. A hue shift alone is not enough there: the band is saturated
    // sand-orange, and any amount of it on green water grades to olive, so what
    // has to go is its CHROMA. Keyed on the reflected ray's own elevation, so it
    // releases within the band's own width and cannot touch the blue overhead.
    const bandF = float(1).sub(saturate(elev.mul(REFL_BAND_ELEV))).mul(0.4).add(0.6).toVar();
    const warm = saturate(refl.x.sub(refl.z).mul(REFL_WARM_GAIN)).toVar();
    const bandL = dot(refl, vec3(0.2126, 0.7152, 0.0722)).toVar();
    // ...but NOT the sun itself, and that exemption is the whole reason the
    // glitter road used to read as chrome under a golden sky. This term keys on
    // "is the reflection warm", and it cannot tell the two warm things in the
    // dome apart: the sand-coloured haze band, which genuinely does grade to
    // olive on green water and is what this was written for, and the solar disc,
    // which is warm because it is the light source. Pulling the second one 48%
    // toward REFL_BAND — a deliberately COOL silver — was repainting the sun.
    //
    // Magnitude separates them cleanly. The haze band never clears the HDR knee
    // in sky.js, so it arrives at a luminance under half; the reconstructed disc
    // arrives an order of magnitude above. Full strength below 1, released to
    // nothing by 3, so the band is treated exactly as before and the sun is left
    // the colour the sky says it is.
    const notSun = saturate(float(3.0).sub(bandL).mul(0.5)).toVar();
    refl.assign(mix(refl, REFL_BAND.mul(bandL), warm.mul(bandF).mul(REFL_BAND_DESAT).mul(notSun)));
    // ...and the same argument applies to its *chroma*. A single mirror tap of a
    // high-contrast sky is a point sample of something the facet actually
    // integrates over several degrees, so a chop ripple that happens to aim at
    // a hole between clouds prints the zenith's saturated indigo as an isolated
    // dash on a wave face. The integral over the lobe is a mixture, and mixtures
    // desaturate: pulling the sample toward its own luminance in proportion to
    // roughness is that, for free, and it takes the blue dashes with it while
    // leaving the value structure — which is the part that reads as water —
    // completely intact.
    // The grey it averages toward is a COOL grey, not a neutral one. Neutral is
    // the trap: luminance is 72% green, so desaturating a warm cloud lands on a
    // khaki that reads as dirt floating on the sea. Biasing the grey point blue
    // means a washed-out reflection can only ever be silver.
    // Capped, though. The cap is the difference between "a rough facet
    // integrates over its lobe" and "the ocean is grey": at full strength every
    // pixel past roughness 0.556 sat on the chroma floor, and ROUGH_MAX is
    // 0.60, so that was the entire sea.
    const reflL = dot(refl, vec3(0.2126, 0.7152, 0.0722));
    refl.assign(mix(refl, mix(REFL_GREY.mul(reflL), refl, float(REFL_CHROMA)),
      saturate(rough.mul(REFL_DESAT)).mul(REFL_DESAT_CAP)));

    // --- body: real value range from trough floor to crest lip ---------------
    // The depth proxy is height above mean sea level through a soft ramp:
    // x/(|x|+k) is monotonic and asymptotic, so however tall a crest gets the
    // colour keeps moving and never lands on a ceiling. Two crests of different
    // height are therefore two different colours, which is the whole point.
    const hN = positionWorld.y.div(WAVE_SCALE).toVar();
    const lift = hN.div(abs(hN).add(HEIGHT_SOFT)).mul(0.5).add(0.5).toVar();
    const facing = saturate(dot(N, UP)).toVar();

    // --- the wave form, read at two scales -----------------------------------
    // Two probes of the displacement map per wave-carrying cascade: one at the
    // pixel's own mip level, one at whatever level blurs it over about
    // CREST_RADIUS metres of sea. Their difference is how far this point stands
    // proud of the water around it — a lip detector that costs two taps and no
    // thresholds, positive at a crest, zero on a plateau, negative in a trough,
    // and continuous everywhere because it is a difference of two smooth fields.
    // The sharp tap of cascade 0 on its own is the swell: the blurred sea level
    // the chop rides on, and an axis of value independent of both height and
    // facing (see GROUP_SCALE).
    const swellH = float(0).toVar();
    const crestRel = float(0).toVar();
    cascades.forEach((c, i) => {
      if (i > 1) return; // the 14 m ripple has nothing to say about wave form
      const L = lengthScales[i];
      const texelM = L / c.N;
      const lod = log2(footprint.div(texelM)).max(0).toVar();
      const blur = Math.max(0, Math.log2(CREST_RADIUS / texelM));
      const here = texture(c.displacement, worldXZ.div(L)).level(lod).y.toVar();
      const wide = texture(c.displacement, worldXZ.div(L)).level(lod.max(float(blur))).y;
      crestRel.addAssign(here.sub(wide));
      if (i === 0) swellH.assign(here);
    });
    const gN = swellH.div(GROUP_SCALE).toVar();
    const group = gN.div(abs(gN).add(GROUP_SOFT)).mul(0.5).add(0.5).toVar();
    // ...and what is left once the swell is subtracted is the chop's own height,
    // which is high on a chop crest even when the whole group is in a hollow.
    // softPos() is a smooth max(x, 0): the ramps below all want "how far above
    // this reference", and a bare max() creases at zero, which prints as an
    // iso-contour drawn across the sea in exactly the way this file spends most
    // of its length trying to avoid. sqrt(x^2 + k^2) rounds the corner off over
    // a band of width k and costs one instruction.
    const softPos = (x, k) => x.add(sqrt(x.mul(x).add(k * k))).mul(0.5);
    const chopS = positionWorld.y.sub(swellH).div(1.6).toVar();
    const chopP = softPos(chopS, 0.30).toVar();
    const chop = chopP.div(chopP.add(1.0)).toVar();
    // Thickness, as one number reused by every term that wants "thin water":
    // how far this point stands proud of the sea within a few metres of it,
    // through the same asymptotic knee. Never clamped, so a lip is a run down
    // the wave face and not a plateau with a contour round it.
    // The knee came down from 0.40, and it was a measured floor rather than a
    // taste change. softPos(0, k) is k/2, so on dead-flat water crestP was 0.20
    // and thinP was 0.20/0.90 = 0.222 — a 22% "thinness" on water that stands
    // proud of nothing at all, i.e. the glow could never switch off anywhere.
    // The comment below claiming thinP "is zero in a trough" was wrong: it was
    // 0.09 in a trough and 0.22 on a plateau, which is most of why the effect
    // read as an even wash. 0.15 puts the flat-water floor at 0.097. The wide
    // knee was there to soften a lobe that had no directional gating of its own;
    // the entry cosine below now does that job properly.
    const crestN = crestRel.div(SSS_DEPTH).toVar();
    const crestP = softPos(crestN, 0.15).toVar();
    const thinP = crestP.div(crestP.add(0.70)).toVar();

    // The water mass carries its own colour, wandering on a scale far longer
    // than the waves — depth, turbidity, plankton. Without it every patch of sea
    // at a given height is the same value and the field reads as a height key
    // rather than a body of water.
    // Three octaves, not two. The longest is depth and turbidity; the middle one
    // varies within a wave group; the shortest exists because a deck camera
    // spends half the frame on ONE swell face ten metres away, where the other
    // two are constants — and a ten-metre patch of sea at one flat value is the
    // moulded-plastic look however good the normal on it is.
    const mass = texture(detailTex, worldXZ.mul(MASS_SCALE).add(vec2(t.mul(0.0016), t.mul(-0.0011)))).b
      .add(texture(detailTex, worldXZ.mul(MASS_NEAR).add(vec2(t.mul(-0.01), t.mul(0.008)))).b.mul(0.6))
      .add(texture(detailTex, worldXZ.mul(MASS_FINE).add(vec2(t.mul(0.02), t.mul(0.03)))).a.mul(0.30))
      .div(1.9).sub(0.5).mul(MASS_AMOUNT).toVar();
    // Asymptotic, NOT saturate(). This is the line that drew the paint-by-
    // numbers coastlines: saturate(mass * 1.2 + 0.18) pinned every point below
    // mass = -0.15 to exactly BODY_COLOR, so half the near field was one flat
    // colour with a one-pixel staircase where it started to move again. x/(|x|+k)
    // has the same shape and the same range and can never reach either end.
    const massT = mass.div(abs(mass).add(MASS_SOFT)).mul(0.5).add(0.5).toVar();
    // ...and the same wander picks the water TYPE, so a patch that is painted
    // as deeper, clearer water also transmits like it. One noise field, two
    // consistent consequences, instead of a colour swatch that knows nothing
    // about the transmittance sitting next to it. massT runs 0 = body toward
    // 1 = abyss, and deep water is the clear end, hence the argument order.
    const muWater = mix(MU_TURBID, MU_CLEAR, massT).toVar();
    // The deep anchor keeps some green in it: the palette's deep swatch is a
    // near-black navy, and letting the wander run all the way to it turned the
    // shadowed troughs slate-grey instead of the green-black a warm sea has.
    const abyss = mix(vec3(shading.deepColor), ABYSS_COLOR, float(0.65)).toVar();
    const deepBody = mix(BODY_COLOR, abyss, massT).toVar();
    // A second, independent axis: hue rather than value. Depth alone gives one
    // colour getting darker; a real sea also swings between blue-green over
    // deep water and a greener cast where it is shallower or richer, and having
    // that on its own frequency is what stops the wander reading as a single
    // grey-scale field tinted teal.
    const hue = texture(detailTex, worldXZ.mul(MASS_HUE).add(vec2(t.mul(-0.0022), t.mul(0.0017)))).a
      .sub(0.5).mul(2.2).toVar();
    deepBody.assign(mix(deepBody, BODY_BLUE,
      hue.div(abs(hue).add(0.9)).mul(0.5).add(0.5).mul(MASS_HUE_AMOUNT)));

    // trough floor -> body -> thin lit crest, as one continuous run.
    // Also asymptotic — see BODY_BIAS. The smoothstep this replaces reached its
    // ceiling at mean sea level, which made the whole upper half of the sea one
    // flat card and drew a horizontal contour under it.
    const hB = hN.add(BODY_BIAS).toVar();
    const rise = hB.div(abs(hB).add(BODY_SOFT)).mul(0.5).add(0.5).toVar();
    const body = mix(TROUGH_COLOR, deepBody, rise).toVar();
    // Thin water reads shallow where the water IS thin — which is the measured
    // quantity two lines up, not the surface's facing. Gating on facing had the
    // effect exactly inside out: a 75-degree overhanging lip, the one place the
    // whole cue exists for, ran the tint at 42% while flat water ran it at
    // 100%. Facing survives only as a weak trim, enough to stop a trough floor
    // lighting up, nothing like a gate.
    const shIn = lift.sub(0.46).mul(2.0).add(thinP.mul(1.6)).add(chop.mul(0.8)).toVar();
    const shP = softPos(shIn, 0.22).toVar();
    const shallow = shP.div(shP.add(1.0)).mul(SHALLOW_MAX)
      .mul(mix(float(0.30), float(1.0), thinP))
      .mul(float(0.85).add(facing.mul(0.15))).toVar();
    // The crest tint keeps the palette uniform in play so the GUI still bites,
    // pulled toward a more saturated turquoise than a mid teal swatch can give.
    body.assign(mix(body, mix(vec3(shading.scatterColor), CREST_COLOR, float(0.72)), shallow));

    // Down in a trough the water only sees a slot of sky between the walls
    // around it; on a crest it sees the whole dome. This is what stops the
    // frame collapsing to one mid-tone — it darkens body and reflection alike.
    const skyVis = float(SKY_VIS_MIN).add(saturate(lift.mul(1.15)).mul(float(1).sub(SKY_VIS_MIN))).toVar();
    // The wave group's own occlusion, on its own axis. Deep inside a swell
    // hollow the surrounding water cuts the sky down whatever this point's own
    // height and facing are doing, and a chop crest standing on the group wins
    // some of it back. Without a term that is independent of (height, facing),
    // every quantity in the value chain pins to a constant on a flat foreground
    // patch and the hero wave face comes out a featureless card.
    const groupOcc = float(1).sub(float(GROUP_DARK).mul(float(1).sub(group)))
      .add(chop.mul(CHOP_LIFT).mul(float(1).sub(group))).toVar();
    // Sky ambient (a flat sheet sees the whole dome, a steep face a slice of it)
    // and then a wrapped sun term: faces turned away fall into a cool shadow
    // instead of a hard terminator, and that split does most of the value range.
    const sunWrap = pow(saturate(dot(N, shading.sunDir).mul(0.55).add(0.45)), float(1.7)).toVar();
    // Thin water does not obey a lambert on its own surface normal — light
    // arrives through it from behind, so the more of it there is the less the
    // ambient and the terminator are allowed to darken it. Without this the
    // crest tint is mixed in at full strength and then immediately halved by
    // the two attenuations, because a lip is by definition the steepest, most
    // side-lit part of the wave: the brightest colour in the palette was
    // reachable only on flat water, where nothing is thin.
    const amb = mix(float(0.36).add(facing.mul(0.64)), float(0.72).add(facing.mul(0.28)), thinP).toVar();
    body.mulAssign(amb.mul(skyVis).mul(groupOcc));
    // The sunlit half is lifted mostly in value, not in hue: multiplying deep
    // green water by a saturated warm light turns it olive, and olive is the one
    // colour a tropical sea must never be. But half-white was too much of a
    // hedge — under a sun the sky measures at 1.00 / 0.83 / 0.58 the sunlit
    // water was being lit by 1.00 / 0.81 / 0.63, so the sea disagreed with its
    // own sky. The white share comes down to a bit over a quarter and the gain
    // goes up to exactly compensate: 1.26 is (luminance at 0.50 white) / (at
    // 0.28), so this is a pure hue change and the value chain above it is
    // untouched. Olive is still the thing to watch for on the sunlit shoulder
    // of a swell — if it appears, raise the white share, not the gain.
    body.mulAssign(mix(SHADOW_TINT, mix(vec3(shading.sunColor), vec3(1), float(0.28)).mul(1.26),
      saturate(sunWrap.add(thinP.mul(0.35)))));
    // Single scattering, added rather than multiplied — see SCATTER_COLOR. Every
    // other path to brightness in this file is either an attenuation of a dark
    // swatch or a reflection of the sky, so the frame's value histogram was
    // bimodal with a hole from 0.15 to 0.45 linear and the sea had no turquoise
    // anywhere. This is sun energy coming back OUT of the water: view-
    // independent, so it lights the whole sunlit shoulder of a swell, gated on
    // the measured thinness so a lip gets three times what a trough floor does,
    // and multiplied by the group occlusion so a hollow stays dark.
    const sunN = saturate(dot(N, shading.sunDir)).toVar();
    body.addAssign(SCATTER_COLOR.mul(vec3(shading.sunColor)).mul(sunN)
      .mul(mix(float(SCATTER_BASE), float(1), thinP)).mul(groupOcc).mul(SCATTER_GAIN));
    refl.mulAssign(float(0.55).add(skyVis.mul(0.5)));

    // --- subsurface scatter -------------------------------------------------
    // Light that entered the FAR side of the wave, crossed it, and came out
    // toward the eye. Three things have to line up and the old form only had
    // one and a half of them:
    //
    //   1. light has to get IN, through the face you cannot see;
    //   2. having scattered, it has to come out toward the CAMERA;
    //   3. the water it crossed has to be thin enough to let it.
    //
    // What was here was a single DICE-style wrap lobe,
    // pow(saturate(dot(V, -normalize(sunDir + N*0.9))), 1.6), doing all three at
    // once through one authored "distortion" constant — and it had the sign of
    // (1) inside out. Evaluated against this scene's own sun (az 122.8, el 11.2)
    // and a deck camera it returned 0.79 on the far face turned toward the sun,
    // the one surface whose light cannot reach the eye at all, 0.48 on flat
    // water, and 0.18 on the near-vertical backlit wall the whole effect exists
    // for. It was four and a half times brighter on the wrong facet than on the
    // right one, which is why the cue read as a green wash over the sunward half
    // of the sea rather than as light coming through a crest.
    //
    // Split into the three factors it always was:
    //
    // (1) The far face's outward normal is -N, so the cosine of incidence there
    // is -dot(N, sunDir). That one quantity replaces SSS_DISTORT entirely: it is
    // negative — clamped to zero — on every front-lit facet by construction, and
    // ~0.98 on a wall lit from behind. No constant to tune, and it cannot be
    // pointed the wrong way without the glow visibly vanishing.
    const cosIn = saturate(dot(N, shading.sunDir).negate()).toVar();
    // ...and only some of the sun gets in at all. At this sun's elevation the
    // interface rejects 31% of it before any of this starts, and at grazing
    // incidence on a steep face far more. This is the physical replacement for
    // sssStrength having to carry an entry term it knew nothing about.
    const Tin = float(1).sub(fresnelDielectric(cosIn, float(ETA_AW))).toVar();
    // (2) The exit lobe is the volume's phase function and NOTHING else. A phase
    // function is a property of the medium: it depends only on the angle between
    // where the light was going and where the eye is, and it has no opinion about
    // the surface normal. Folding N into it, as the old form did, is what made
    // the term fire on geometry rather than on light.
    const fwd = hgForward(SSS_G, saturate(dot(V, shading.sunDir.negate()))).toVar();
    // (3) Path length, in metres, and refracted. The view ray bends at the
    // surface, so the distance it travels through a sheet of thickness t is
    // t/cos(theta_t) — bounded at 1.50x by the critical angle, which is why this
    // is a real term and not an unbounded one. thinP is a proudness measure, so
    // the path SHORTENS as it rises: LIP_TIP is the thin end.
    const cosT = sqrt(float(1).sub(float(ETA2).mul(float(1).sub(NoV.mul(NoV))))).toVar();
    const lipD = mix(float(LIP_BASE), float(LIP_TIP), thinP).div(max(cosT, float(0.1))).toVar();
    // Diffusion transmittance over that path, not beam extinction. At a single-
    // scattering albedo near 0.9 most of the light arriving has scattered
    // several times, so it diffuses rather than travelling straight, and the
    // transport coefficient is the right one — see water.js. This is the term
    // that used to be a pair of authored swatches with a blend between them: a
    // thin lip lands on a warm near-white and a metre further down the face on a
    // jade green, off one exponential.
    const Tvol = exp(muWater.mul(lipD).negate()).toVar();
    // Thickness, measured rather than inferred — thinP, built above out of the
    // sharp-minus-blurred displacement, so it is largest exactly at a lip and
    // falls off continuously down the face over a metre or two. The height term
    // on the end is a gentle bias that keeps a chance ridge on the floor of a
    // deep trough from lighting up; it is a bias, not a gate, and it never clamps.
    const thin = thinP.mul(float(0.25).add(lift.mul(0.90))).toVar();
    // Input to the knee peaks a little under 2, where x/(x+1) is still climbing:
    // the glow stays a gradient over the whole face instead of clipping to a
    // flat patch of one green. The gain is unchanged from the old lobe on
    // purpose — hgForward is peak-normalised, so the number the knee sees still
    // tops out at the same place and the tuning carries over.
    const sss = fwd.mul(cosIn).mul(Tin).mul(thin).mul(shading.sssStrength).mul(SSS_GAIN).toVar();
    const glow = Tvol.mul(shading.sunColor).mul(sss.div(sss.add(1)).mul(SSS_MAX)).toVar();

    // One energy-conserving split: what is not reflected is transmitted, and
    // everything coming out of the water — body colour and subsurface glow
    // alike — is on the transmitted side of it.
    //
    // The glow used to be divided 45/55 between "under the Fresnel" and "on top
    // of it, at 1 - fres*0.5", with a component-wise body floor underneath the
    // lot. That existed because Schlick with the bent exponent reached 0.43 at
    // the grazing angles you actually view a backlit crest from, which killed
    // the glow at exactly the wrong moment; exact Fresnel is 0.31 there, so
    // there is nothing left to compensate for. The floor goes with it — REFL_CEIL
    // already guarantees the sea keeps 14% of its own colour in the mix.
    const water = mix(body.add(glow), refl, fres).toVar();
    // A deliberate chroma push, on the water and nothing else — see SAT_BOOST.
    // It sits here rather than on the finished pixel because foam and glitter
    // are white by intent, and extrapolating a white away from its own luminance
    // only ever finds a colour cast to exaggerate.
    const waterL = dot(water, vec3(0.2126, 0.7152, 0.0722));
    water.assign(max(mix(vec3(waterL), water, float(SAT_BOOST)), vec3(0)));

    // --- sun glitter --------------------------------------------------------
    // One GGX lobe against the sun, rolled off through a knee instead of
    // clamped: the peak saturates smoothly however narrow the lobe gets, so
    // there are no fireflies, and because roughness grows with distance the
    // path widens and softens toward the horizon on its own.
    const Hv = normalize(V.add(shading.sunDir));
    // The lobe rides on a roughness that keeps widening with distance, so the
    // path broadens toward the horizon rather than breaking up into whichever
    // far facets happen to line up.
    const roughSpec = min(max(sqrt(rough.mul(rough)
      .add(saturate(viewDist.div(SPEC_SPREAD_RANGE)).mul(SPEC_SPREAD))),
    float(SPEC_ROUGH_MIN)), float(SPEC_ROUGH_MAX)).toVar();
    const a2 = roughSpec.mul(roughSpec).mul(roughSpec).mul(roughSpec).toVar();
    const NoH = saturate(dot(N, Hv)).toVar();
    const den = NoH.mul(NoH).mul(a2.sub(1)).add(1);
    const ggx = a2.div(den.mul(den).mul(PI)).mul(saturate(dot(N, shading.sunDir))).toVar();
    const specTint = mix(vec3(shading.sunColor), vec3(1), float(SPEC_WHITE));
    water.addAssign(specTint.mul(ggx.div(ggx.add(SPEC_KNEE)).mul(SPEC_MAX)).mul(fresSpec));

    // foam appearance lives in foamShading.js
    const foam = foamShading({ cascades, lengthScales, worldXZ, shading, detailTex, N });
    const surface = mix(water, foam.color, foam.coverage).toVar();

    // A crest can swallow a deck-height camera. The sheet is DoubleSide, so that
    // frame is drawn from underneath — and looking UP through water is the one
    // place in this whole scene where refraction is a real, visible thing rather
    // than a bent ray with nothing behind it to bend.
    //
    // Snell's window. Everything above the surface — the entire 180-degree sky —
    // is compressed into a cone of half-angle asin(1/1.34) = 48.27 degrees
    // directly overhead. The zenith lands at the middle of it, the horizon lands
    // exactly on its rim, and the bottom ten degrees of sky squeeze into the last
    // one degree before that rim. Outside the cone there is no sky at all: past
    // the critical angle the surface is a perfect mirror and you see the water
    // column reflected back at you.
    //
    // This costs one sky tap and no plumbing whatsoever, and that is worth
    // saying out loud because it is the reason this is the refraction worth
    // having: skyColor() is a pure function of a direction into a baked
    // panorama, so a refracted ray can be evaluated exactly, in closed form,
    // with no render target, no framebuffer copy, no second pass and no depth
    // buffer. Screen-space refraction here would sample the clear colour and
    // return a flat constant — the ocean draws before the sky dome.
    //
    // What was here instead was a crossfade to a flat swatch with a glow bolted
    // on. It had no Fresnel at all, and the obvious fix — Schlick on the
    // water-side cosine — returns 0.025 at the critical angle where the true
    // answer is exactly 1.0, so it would have produced no window edge and no
    // total internal reflection.
    // Is the camera actually IN the water? dot(N, V) < 0 on its own is not that
    // test and never was: it fires on any facet tilting a few degrees away from
    // a grazing sight line, which at deck height is most of the horizon band, so
    // the old path was quietly painting the far field with a swatch meant for a
    // camera inside a wave. A debug pass showed it covering large contiguous
    // areas of open water in the deck framing.
    //
    // The honest test is a height comparison, and the height field is already on
    // the GPU. shading.originXZ is the ocean tile's origin, which main.js snaps
    // to the camera's own XZ every frame, so sampling the displacement there is
    // sampling the water under the lens. Two taps at an address that is constant
    // across the whole frame — no CPU readback, no async latency, no uniform to
    // keep in sync. Cascade 2 is 24 cm of ripple and cannot decide this.
    const camWaterY = float(0).toVar();
    cascades.forEach((c, i) => {
      if (i > 1) return;
      camWaterY.addAssign(texture(c.displacement, shading.originXZ.div(lengthScales[i])).level(0).y);
    });
    // Ramped over ~30 cm rather than switched, so breaking the surface is a
    // dissolve and not a cut. Approximate by design: the sample is at the
    // undisplaced patch coordinate, so choppiness puts it up to a metre or two
    // off in XZ, which is nothing next to a wave that is swallowing the camera.
    const submerged = saturate(camWaterY.sub(cameraPosition.y).mul(3).add(0.5)).toVar();
    const under = saturate(dot(N, V).negate().mul(8)).mul(submerged).toVar();
    const Nb = N.negate().toVar(); // the sheet's normal as seen from below: into the air
    const ciUp = saturate(dot(Nb, V)).toVar();
    const Fup = fresnelDielectric(ciUp, float(N_WATER)).toVar();
    // refract() takes the incident direction — pointing AT the surface, hence
    // -V — and returns exactly vec3(0) under total internal reflection. That
    // zero is load-bearing and dangerous in equal measure: normalize(vec3(0)) is
    // NaN, and NaN survives multiplication by the zero weight it is about to get,
    // so the epsilon is required rather than defensive.
    const up = refract(V.negate(), Nb, float(N_WATER)).toVar();
    const window = skyColor(normalize(up.add(vec3(0, 1e-5, 0))), shading).toVar();
    // Inside the cone, sky; outside it, the mirrored water column. Fup does the
    // switch on its own and reaches exactly 1 at the rim, so the edge of the
    // window is drawn by the physics rather than by a threshold.
    const underLit = mix(window, deepBody.mul(float(0.30).add(facing.mul(0.5))), Fup).toVar();
    surface.assign(mix(surface, underLit, under));

    // --- aerial perspective -------------------------------------------------
    // The whole of it, composited here — see HAZE_DENSITY for why the water
    // opts out of scene.fogNode and does this itself.
    //
    // Two deliberate asymmetries make the horizon a line rather than a
    // dissolve. The target is the sky sampled just ABOVE the line rather than
    // along the view ray (which points below it), because "the value the sea
    // has to sit under" is the value the eye is comparing it with. And it is
    // scaled by SEA_SINK, so however thick the haze gets the far water lands a
    // measurable few percent darker than the sky it meets — a 1-pixel step,
    // which is what a horizon is. Being last, after the glitter and the foam,
    // is the other half: additive energy that escapes the haze is exactly what
    // used to bloom the last twenty pixels of sea brighter than the sky.
    const hazeDir = normalize(vec3(V.x.negate(), float(HORIZON_LIFT), V.z.negate()));
    const hazeCol = skyColor(hazeDir, shading).mul(SEA_SINK).toVar();
    // Beer-Lambert over true world distance, density falling off with altitude,
    // sampled at the ray's midpoint — the same model atmosphere.js applies to
    // everything else in the scene, so the water agrees with the spray and the
    // dome instead of drifting away from them.
    const midY = positionWorld.y.add(cameraPosition.y).mul(0.5).max(0);
    const tau = viewDist.mul(HAZE_DENSITY).mul(exp(midY.div(-HAZE_SCALE_H)));
    const ext = float(1).sub(exp(tau.negate())).toVar();
    // ...and the sea loses a little of its own value on top, so the band right
    // under the line reads as sea in shadow rather than as haze that happens to
    // be teal.
    const sink = mix(float(1), float(FAR_SINK), saturate(viewDist.div(FAR_SINK_RANGE)));
    const aerial = mix(surface.mul(sink), hazeCol, ext).toVar();
    // ...and none of that applies with the lens under the water. The extinction
    // along an underwater ray is the sea's, not the atmosphere's, and the two
    // are four orders of magnitude apart: air thins by 1/e over 3.2 km, water
    // over about three METRES. Compositing the sky's haze onto a submerged
    // frame put a sheet of cream-coloured sky across the bottom half of it,
    // which is what a `trough` capture that happened to be swallowed by a crest
    // came back with. Same Beer-Lambert, same muWater the crest glow uses, so
    // the underwater gloom and the light through a lip agree about what water
    // this is. Nothing beyond a few metres survives — which is correct, and is
    // also why this needs no volumetric anything.
    const seaExt = exp(muWater.mul(viewDist).negate()).toVar();
    return mix(aerial, mix(deepBody.mul(0.35), surface, seaExt), submerged);
  })();

  return mat;
}
