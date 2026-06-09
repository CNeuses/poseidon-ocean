import {
  Scene, PerspectiveCamera, WebGPURenderer, Color, Vector3, PlaneGeometry, Mesh, FogExp2, MathUtils,
} from 'three/webgpu';
import { uniform } from 'three/tsl';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import WebGPU from 'three/addons/capabilities/WebGPU.js';
import { params } from './ocean/params.js';
import { Ocean } from './ocean/Ocean.js';
import { validateFFT } from './ocean/fft.js';
import { createOceanSurfaceMaterial } from './ocean/oceanSurfaceMaterial.js';
import { makeDetailTexture } from './ocean/detailTexture.js';
import { createSpray } from './ocean/spray.js';
import { createSkyDome } from './ocean/sky.js';
import { createDebugView, spectrumDebugMaterial } from './ocean/debugView.js';
import { createGUI } from './gui.js';
import { createHUD } from './util/hud.js';

const hud = createHUD();

async function main() {
  if (WebGPU.isAvailable() === false) {
    hud.error('WebGPU is not available. Use Chrome/Edge 113+ or Safari 18+ — this build has no WebGL fallback.');
    return;
  }

  const c = params.colors;
  const scene = new Scene();
  scene.fog = new FogExp2(new Color(c.skyHorizon).getHex(), 0.0045);

  const camera = new PerspectiveCamera(55, innerWidth / innerHeight, 0.5, 30000);
  camera.position.set(0, 16, 68);

  const renderer = new WebGPURenderer({ antialias: true, trackTimestamp: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.setClearColor(new Color(c.skyHorizon), 1);
  document.body.appendChild(renderer.domElement);

  await renderer.init();
  if (!renderer.backend.isWebGPUBackend) {
    hud.error('Renderer fell back to WebGL2 — this project targets WebGPU only.');
    return;
  }
  const device = renderer.backend.device;
  if (device && typeof device.addEventListener === 'function') {
    device.addEventListener('uncapturederror', (e) => hud.error('WebGPU validation: ' + (e.error?.message ?? String(e.error))));
  }

  // FFT isolation test (hard gate)
  const fftTest = await validateFFT(renderer, params.N);
  const fftStr = `FFT self-test: ${fftTest.pass ? 'PASS ✓' : 'FAIL ✗'} (impulse=${fftTest.err1.toExponential(1)}, freq=${fftTest.err2.toExponential(1)})`;
  if (!fftTest.pass) hud.error(fftStr + ' — IFFT not matching analytic result.');

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 0, -20);
  controls.maxPolarAngle = Math.PI * 0.495;

  // shared shading uniforms (sky dome + ocean reflection use the same values)
  const shading = {
    sunDir: uniform(new Vector3()),
    sunColor: uniform(new Color(c.sun)),
    horizon: uniform(new Color(c.skyHorizon)),
    zenith: uniform(new Color(c.skyZenith)),
    deepColor: uniform(new Color(c.deep)),
    scatterColor: uniform(new Color(c.scatter)),
    sssStrength: uniform(params.sssStrength),
    foamColor: uniform(new Color(c.foam)),
    foamThreshold: uniform(params.foamThreshold),
    foamScale: uniform(params.foamScale),
    detail: uniform(params.detailStrength),
    time: uniform(0),
  };
  function updateSun() {
    const az = MathUtils.degToRad(params.sunAzimuth);
    const el = MathUtils.degToRad(params.sunElevation);
    shading.sunDir.value.set(Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az)).normalize();
  }
  updateSun();

  scene.add(createSkyDome(shading));

  // FFT-ocean simulation
  const ocean = new Ocean(renderer, params);
  await ocean.updateInitialSpectrum();

  // shaded ocean surface — ~300k-vert tiled plane (smaller extent keeps the
  // triangles small within budget; fog hides the edge)
  const detailTex = makeDetailTexture();
  const oceanSurfaceMat = createOceanSurfaceMaterial(ocean.cascades, { lengthScales: params.lengthScales, shading, detailTex });
  const oceanMesh = new Mesh(new PlaneGeometry(400, 400, 900, 900), oceanSurfaceMat);
  oceanMesh.frustumCulled = false;
  scene.add(oceanMesh);

  // ballistic spray particles off breaking crests (temporarily disabled)
  const ENABLE_SPRAY = false;
  const spray = ENABLE_SPRAY
    ? createSpray(renderer, { cascades: ocean.cascades, lengthScales: params.lengthScales, N: params.N, shading })
    : null;
  if (spray) scene.add(spray.mesh);
  const windVec = new Vector3();

  // debug views
  const debug = createDebugView();
  const spectrumMats = ocean.cascades.map((cc) => spectrumDebugMaterial(cc.h0, params.N, { exposure: 0.12 }));
  const heightHeatmap = spectrumDebugMaterial(ocean.cascades[0].DyDxz, params.N, { exposure: 0.5 });

  createGUI(params, { ocean, shading, updateSun, spray });

  let view = 'fft';
  addEventListener('keydown', (e) => {
    const k = e.key.toLowerCase();
    if (k === 'f') view = 'fft';
    else if (k === '1') view = 'spectrum0';
    else if (k === '2' && ocean.cascades[1]) view = 'spectrum1';
    else if (k === '3' && ocean.cascades[2]) view = 'spectrum2';
    else if (k === '5') view = 'height';
    else if (k === '=' || k === '+') ocean.lambda.value = Math.min(ocean.lambda.value + 0.1, 3);
    else if (k === '-' || k === '_') ocean.lambda.value = Math.max(ocean.lambda.value - 0.1, 0);
  });

  addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });

  // render loop + timing
  let elapsed = 0;
  let last = performance.now();
  let emaWall = 16.7;
  let gpuMs = -1;
  let hudAccum = 1;
  let resolving = false;

  renderer.setAnimationLoop(() => {
    const now = performance.now();
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    elapsed += dt * params.timeScale;

    ocean.evolve(elapsed, dt * params.timeScale); // foam dissipation tracks time scale
    shading.time.value = elapsed;
    if (spray) {
      const wd = MathUtils.degToRad(params.local.windDirection);
      windVec.set(Math.cos(wd), 0, Math.sin(wd)).multiplyScalar(params.local.windSpeed * 0.3);
      spray.update(dt, camera.position, windVec);
    }
    controls.update();

    if (view === 'fft') {
      renderer.render(scene, camera);
    } else {
      debug.mesh.material =
        view === 'spectrum0' ? spectrumMats[0]
          : view === 'spectrum1' ? spectrumMats[1]
            : view === 'spectrum2' ? spectrumMats[2]
              : heightHeatmap;
      renderer.render(debug.scene, debug.camera);
    }

    emaWall = emaWall * 0.9 + dt * 1000 * 0.1;
    hudAccum += dt;
    if (hudAccum >= 0.25) {
      hudAccum = 0;
      if (renderer.backend.trackTimestamp && !resolving) {
        resolving = true;
        renderer.resolveTimestampsAsync('render').then(() => { gpuMs = renderer.info.render.timestamp; }).catch(() => {}).finally(() => { resolving = false; });
      }
      const gpuTxt = gpuMs >= 0 ? `${gpuMs.toFixed(2)} ms GPU/render` : 'GPU ms n/a';
      hud.set(
        `WebGPU · ${(1000 / emaWall).toFixed(0)} fps · ${emaWall.toFixed(2)} ms wall · ${gpuTxt}\n` +
        `step 6 · foam · N=${params.N} · ${ocean.cascades.length} cascades · choppiness λ=${ocean.lambda.value.toFixed(2)} (+/-)\n` +
        `view: ${view}   (F = ocean, 5 = height map, 1/2/3 = spectra)\n` +
        fftStr,
      );
    }
  });
}

main().catch((e) => hud.error(String(e?.stack ?? e)));
