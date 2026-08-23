import type { BufferGeometry, Material, Texture, Vector2, Vector3, WebGPURenderer } from 'three/webgpu';
import type { Node } from 'three/tsl';

export type PoseidonUpAxis = 'y' | 'z';
export type PoseidonPreset = 'ocean' | 'lake_calm';

export interface PoseidonSpectrumConfig {
  scale: number;
  windSpeed: number;
  windDirection: number;
  fetch: number;
  spreadBlend: number;
  swell: number;
  peakEnhancement: number;
  shortWavesFade: number;
  tailFalloff: number;
  tailFloor: number;
  windCoupling?: number;
}

export interface PoseidonSimulationConfig {
  N: number;
  cascades: number;
  lengthScales: number[];
  boundaryFactor: number;
  g: number;
  depth: number;
  lambda: number;
  chopFalloff: number;
  chopFloor: number;
  chopLean: number;
  local: PoseidonSpectrumConfig;
  swell: PoseidonSpectrumConfig;
  timeScale: number;
  foamThreshold: number;
  foamScale: number;
  foamDecay: number;
  foamSpread: number;
  foamBright: number;
  foamRelief: number;
  foamMilk: number;
  detailStrength: number;
  palette: number;
  [key: string]: unknown;
}

export interface PoseidonSimulation {
  readonly cascades: unknown[];
  readonly config: PoseidonSimulationConfig;
  readonly disposed: boolean;
  readonly elapsedSeconds: number;
  step(deltaSeconds: number, timeScale?: number): void;
  update(patch: Partial<PoseidonSimulationConfig>): Promise<boolean>;
  readbackH0(cascade?: number): Promise<Float32Array>;
  dispose(): void;
}

export interface PoseidonShadingState {
  readonly upAxis: PoseidonUpAxis;
  readonly uniforms: Record<string, { value: unknown }>;
  setOrigin(x: number, horizontalY: number): void;
  setTime(seconds: number): void;
  setSunDirection(direction: [number, number, number]): void;
}

export interface PoseidonSurfaceResource {
  material: Material;
  shadingState: PoseidonShadingState;
  dispose(): void;
}

export const POSEIDON_PRESET_NAMES: readonly PoseidonPreset[];
export function createPoseidonConfig(
  preset?: PoseidonPreset,
  overrides?: Partial<PoseidonSimulationConfig>,
): PoseidonSimulationConfig;
export function createPoseidonSimulation(
  renderer: WebGPURenderer,
  options?: { preset?: PoseidonPreset; config?: Partial<PoseidonSimulationConfig> },
): Promise<PoseidonSimulation>;
export function createPoseidonShadingState(options?: {
  upAxis?: PoseidonUpAxis;
  sunDirection?: [number, number, number];
  sunIntensity?: number;
  palette?: number;
  colors?: Record<string, number | string>;
  [key: string]: unknown;
}): PoseidonShadingState;
export function createPoseidonSpectralSurfaceMaterial(
  simulation: PoseidonSimulation,
  options?: {
    upAxis?: PoseidonUpAxis;
    shadingState?: PoseidonShadingState;
    shading?: Record<string, unknown>;
    displacementMask?: Node;
    environmentColor?: (direction: Node) => Node;
    surfaceElevation?: Node | number;
    detailTextureSize?: number;
  },
): PoseidonSurfaceResource;
export function createPoseidonShoreMask(options?: {
  attributeName?: string;
  fadeDistanceMeters?: number;
}): Node;
export function createPoseidonRadialGeometry(options?: {
  rings?: number;
  sectors?: number;
  spacing?: number;
  soften?: number;
}): {
  geometry: BufferGeometry;
  outerRadius: number;
  vertexCount: number;
  innerSpacing: number;
};
export function validatePoseidonFft(
  renderer: WebGPURenderer,
  resolution?: number,
): Promise<{ pass: boolean; err1: number; err2: number }>;
