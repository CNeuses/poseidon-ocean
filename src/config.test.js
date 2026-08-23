import { describe, expect, it } from 'vitest';
import { createPoseidonConfig, POSEIDON_PRESET_NAMES } from './config.js';

describe('Poseidon presets', () => {
  it('keeps the upstream ocean as the reference preset', () => {
    const ocean = createPoseidonConfig('ocean');
    expect(ocean.N).toBe(256);
    expect(ocean.lengthScales).toEqual([1024, 144, 24]);
    expect(ocean.lambda).toBe(2.2);
  });

  it('makes bounded lakes materially calmer and slower than oceans', () => {
    const ocean = createPoseidonConfig('ocean');
    const lake = createPoseidonConfig('lake_calm');
    expect(POSEIDON_PRESET_NAMES).toEqual(['ocean', 'lake_calm']);
    expect(lake.local.scale).toBeLessThan(ocean.local.scale / 5);
    expect(lake.swell.scale).toBeLessThan(ocean.swell.scale / 10);
    expect(lake.lambda).toBeLessThan(ocean.lambda / 3);
    expect(lake.timeScale).toBeLessThan(ocean.timeScale / 2);
  });

  it('returns independent mutable configurations', () => {
    const first = createPoseidonConfig('lake_calm');
    const second = createPoseidonConfig('lake_calm');
    first.local.windSpeed = 99;
    expect(second.local.windSpeed).toBe(3.2);
  });
});
