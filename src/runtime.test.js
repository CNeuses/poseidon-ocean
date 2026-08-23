import { describe, expect, it } from 'vitest';
import { createPoseidonRadialGeometry, createPoseidonShadingState } from './runtime.js';

describe('public runtime helpers', () => {
  it('creates a deterministic radial surface without scene ownership', () => {
    const grid = createPoseidonRadialGeometry({ rings: 3, sectors: 8, spacing: 1, soften: 2 });
    expect(grid.vertexCount).toBe(25);
    expect(grid.geometry.getAttribute('position').count).toBe(25);
    expect(grid.geometry.index.count).toBe(120);
    grid.geometry.dispose();
  });

  it('maps a Z-up world direction into Poseidons internal Y-up frame', () => {
    const state = createPoseidonShadingState({ upAxis: 'z', sunDirection: [0, 0, 1] });
    expect(state.uniforms.sunDir.value).toMatchObject({ x: 0, y: 1, z: 0 });
    state.setOrigin(12, 34);
    expect(state.uniforms.originXZ.value).toMatchObject({ x: 12, y: 34 });
  });
});
