import { params } from './ocean/params.js';

function clone(value) {
  if (Array.isArray(value)) return value.map(clone);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, clone(item)]));
  }
  return value;
}

function merge(target, source) {
  for (const [key, value] of Object.entries(source ?? {})) {
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      target[key] = merge(
        target[key] && typeof target[key] === 'object' ? target[key] : {},
        value,
      );
    } else {
      target[key] = clone(value);
    }
  }
  return target;
}

const PRESET_OVERRIDES = {
  ocean: {},
  lake_calm: {
    lengthScales: [96, 18, 4],
    boundaryFactor: 5,
    depth: 14,
    lambda: 0.34,
    chopFalloff: 0.5,
    chopFloor: 0.08,
    chopLean: 0.1,
    timeScale: 0.32,
    local: {
      scale: 0.035,
      windSpeed: 3.2,
      windDirection: 35,
      fetch: 4200,
      spreadBlend: 0.38,
      swell: 0.12,
      peakEnhancement: 2.2,
      shortWavesFade: 0.18,
      tailFalloff: 1.4,
      tailFloor: 0.28,
    },
    swell: {
      scale: 0.006,
      windCoupling: 0.25,
      windSpeed: 4.0,
      windDirection: 18,
      fetch: 9000,
      spreadBlend: 0.65,
      swell: 0.25,
      peakEnhancement: 2.0,
      shortWavesFade: 0.75,
      tailFalloff: 1.9,
      tailFloor: 0.12,
    },
    foamThreshold: 0.08,
    foamScale: 1.2,
    foamDecay: 2.5,
    foamSpread: 0.7,
    foamBright: 0.72,
    foamMilk: 0.1,
    detailStrength: 0.055,
    palette: 0,
  },
};

export const POSEIDON_PRESET_NAMES = Object.freeze(Object.keys(PRESET_OVERRIDES));

export function createPoseidonConfig(preset = 'ocean', overrides = {}) {
  if (!(preset in PRESET_OVERRIDES)) {
    throw new Error(`Unknown Poseidon preset ${JSON.stringify(preset)}.`);
  }
  return merge(merge(clone(params), PRESET_OVERRIDES[preset]), overrides);
}

export function patchPoseidonConfig(config, patch) {
  return merge(config, patch);
}
