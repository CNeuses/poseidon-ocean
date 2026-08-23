import { Color, Vector2, Vector3 } from 'three/webgpu';
import { attribute, float, mix, positionGeometry, sin, smoothstep, uniform } from 'three/tsl';
import { createPoseidonConfig, patchPoseidonConfig } from './config.js';
import { makeDetailTexture } from './ocean/detailTexture.js';
import { validateFFT } from './ocean/fft.js';
import { Ocean } from './ocean/Ocean.js';
import { createRadialGrid } from './ocean/oceanGrid.js';
import { createOceanSurfaceMaterial } from './ocean/oceanSurfaceMaterial.js';

const STRUCTURAL_KEYS = new Set(['N', 'cascades', 'lengthScales', 'boundaryFactor']);

function assertRenderer(renderer) {
  if (!renderer || typeof renderer.compute !== 'function' || typeof renderer.computeAsync !== 'function') {
    throw new Error('Poseidon needs an initialized Three.js WebGPURenderer.');
  }
}

function validateConfig(config) {
  if (!Number.isInteger(config.N) || config.N < 8 || (config.N & (config.N - 1)) !== 0) {
    throw new Error('Poseidon N must be a power of two of at least 8.');
  }
  if (!Array.isArray(config.lengthScales) || config.lengthScales.length < config.cascades) {
    throw new Error('Poseidon needs one length scale per cascade.');
  }
}

export async function createPoseidonSimulation(renderer, options = {}) {
  assertRenderer(renderer);
  const config = createPoseidonConfig(options.preset ?? 'ocean', options.config ?? {});
  validateConfig(config);
  const ocean = new Ocean(renderer, config);
  await ocean.updateInitialSpectrum();

  let elapsed = 0;
  let updateVersion = 0;
  let rebuildQueue = Promise.resolve(false);

  const runtime = {
    get cascades() { return ocean.cascades; },
    get config() { return config; },
    get disposed() { return ocean.disposed; },
    get elapsedSeconds() { return elapsed; },
    step(deltaSeconds, timeScale = config.timeScale ?? 1) {
      if (ocean.disposed) return;
      const dt = Math.max(0, Math.min(Number(deltaSeconds) || 0, 0.1)) * timeScale;
      elapsed += dt;
      ocean.evolve(elapsed, dt);
    },
    update(patch) {
      if (ocean.disposed) {
        return Promise.reject(new Error('Cannot update a disposed Poseidon simulation.'));
      }
      for (const key of STRUCTURAL_KEYS) {
        if (Object.prototype.hasOwnProperty.call(patch, key)) {
          return Promise.reject(
            new Error(`Poseidon ${key} is structural; create a new simulation to change it.`),
          );
        }
      }
      patchPoseidonConfig(config, patch);
      ocean.lambda.value = config.lambda;
      ocean.foamDecay.value = config.foamDecay;
      ocean.foamSpread.value = config.foamSpread;
      const version = ++updateVersion;
      rebuildQueue = rebuildQueue.catch(() => false).then(async () => {
        if (version !== updateVersion || ocean.disposed) return false;
        await ocean.updateInitialSpectrum();
        return version === updateVersion && !ocean.disposed;
      });
      return rebuildQueue;
    },
    readbackH0(cascade = 0) {
      return ocean.readbackH0(cascade);
    },
    dispose() {
      updateVersion++;
      ocean.dispose();
    },
  };
  return runtime;
}

function internalDirection(direction, upAxis) {
  const [x, y, z] = direction;
  return upAxis === 'z' ? new Vector3(x, z, y) : new Vector3(x, y, z);
}

export function createPoseidonShadingState(options = {}) {
  const upAxis = options.upAxis ?? 'y';
  if (upAxis !== 'y' && upAxis !== 'z') {
    throw new Error("Poseidon upAxis must be 'y' or 'z'.");
  }
  const colors = {
    sun: 0xfffbf2,
    horizon: 0xc2d5eb,
    zenith: 0x356ace,
    ambient: 0x5d86d5,
    deep: 0x07283a,
    scatter: 0x38b9ad,
    foam: 0xf2f4eb,
    ...(options.colors ?? {}),
  };
  const shading = {
    sunDir: uniform(internalDirection(options.sunDirection ?? [0.48, 0.64, 0.6], upAxis).normalize()),
    sunColor: uniform(new Color(colors.sun).multiplyScalar(options.sunIntensity ?? 1.45)),
    horizon: uniform(new Color(colors.horizon)),
    zenith: uniform(new Color(colors.zenith)),
    ambient: uniform(new Color(colors.ambient)),
    deepColor: uniform(new Color(colors.deep)),
    scatterColor: uniform(new Color(colors.scatter)),
    palette: uniform(options.palette ?? 1),
    sssStrength: uniform(options.sssStrength ?? 1),
    foamColor: uniform(new Color(colors.foam)),
    foamThreshold: uniform(options.foamThreshold ?? 0.32),
    foamScale: uniform(options.foamScale ?? 2.5),
    foamBright: uniform(options.foamBright ?? 0.88),
    foamRelief: uniform(options.foamRelief ?? 0.18),
    foamMilk: uniform(options.foamMilk ?? 0.45),
    detail: uniform(options.detailStrength ?? 0.1),
    time: uniform(0),
    originXZ: uniform(new Vector2()),
    hazeWater: uniform(options.hazeWater ?? 1 / 9000),
    hazeAir: uniform(options.hazeAir ?? 1 / 6500),
    specBoost: uniform(options.specBoost ?? 12),
  };
  return {
    upAxis,
    uniforms: shading,
    setOrigin(x, horizontalY) {
      shading.originXZ.value.set(x, horizontalY);
    },
    setTime(seconds) {
      shading.time.value = seconds;
    },
    setSunDirection(direction) {
      shading.sunDir.value.copy(internalDirection(direction, upAxis)).normalize();
    },
  };
}

export function createPoseidonSpectralSurfaceMaterial(simulation, options = {}) {
  if (!simulation || simulation.disposed) {
    throw new Error('Poseidon surface needs a live simulation.');
  }
  const shadingState = options.shadingState ?? createPoseidonShadingState({
    ...options.shading,
    upAxis: options.upAxis ?? 'y',
    foamThreshold: simulation.config.foamThreshold,
    foamScale: simulation.config.foamScale,
    foamBright: simulation.config.foamBright,
    foamRelief: simulation.config.foamRelief,
    foamMilk: simulation.config.foamMilk,
    detailStrength: simulation.config.detailStrength,
    palette: simulation.config.palette,
  });
  const detailTexture = makeDetailTexture(options.detailTextureSize ?? 512);
  const material = createOceanSurfaceMaterial(simulation.cascades, {
    lengthScales: simulation.config.lengthScales,
    shading: shadingState.uniforms,
    detailTex: detailTexture,
    upAxis: options.upAxis ?? shadingState.upAxis,
    displacementMask: options.displacementMask ?? float(1),
    opacity: options.opacity,
    shoreFoamMask: options.shoreFoamMask,
    environmentColor: options.environmentColor,
    surfaceElevation: options.surfaceElevation ?? float(0),
  });
  let disposed = false;
  return {
    material,
    shadingState,
    dispose() {
      if (disposed) return;
      disposed = true;
      material.dispose();
      detailTexture.dispose();
    },
  };
}

export function createPoseidonShoreMask(options = {}) {
  const shoreDistance = attribute(options.attributeName ?? 'waterShoreDistance', 'float');
  return smoothstep(float(0), float(options.fadeDistanceMeters ?? 3), shoreDistance);
}

/** Blend the spectral surface with the scene according to authored water depth. */
export function createPoseidonDepthOpacity(options = {}) {
  const depth = attribute(options.depthAttributeName ?? 'waterDepth', 'float').max(0);
  const clarity = attribute(options.clarityAttributeName ?? 'waterClarity', 'float').clamp(0, 1);
  const shallowOpacity = mix(float(0.56), float(0.16), clarity);
  const deepOpacity = mix(float(0.96), float(0.76), clarity);
  const extinctionDepth = mix(float(1.4), float(8), clarity);
  const depthBlend = smoothstep(float(0.08), extinctionDepth, depth);
  return mix(shallowOpacity, deepOpacity, depthBlend).clamp(0, 1);
}

/** Produce a broken-up foam band from the same shore-distance contract as displacement. */
export function createPoseidonShoreFoamMask(options = {}) {
  const distance = attribute(options.distanceAttributeName ?? 'waterShoreDistance', 'float');
  const strength = attribute(options.strengthAttributeName ?? 'waterFoamStrength', 'float')
    .clamp(0, 1);
  const fadeDistance = float(options.fadeDistanceMeters ?? 1.8);
  const band = float(1).sub(smoothstep(float(0.04), fadeDistance, distance));
  const broad = sin(positionGeometry.x.mul(1.37).add(positionGeometry.y.mul(1.91)))
    .mul(0.5).add(0.5);
  const lace = sin(positionGeometry.x.mul(4.13).sub(positionGeometry.y.mul(2.77)))
    .mul(0.5).add(0.5);
  return band.mul(mix(float(0.48), float(1), broad.mul(0.7).add(lace.mul(0.3))))
    .mul(strength).clamp(0, 1);
}

export function createPoseidonRadialGeometry(options = {}) {
  return createRadialGrid(options);
}

export function validatePoseidonFft(renderer, resolution = 32) {
  assertRenderer(renderer);
  return validateFFT(renderer, resolution);
}
