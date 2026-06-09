import { OrthographicCamera, Scene, Mesh, PlaneGeometry, MeshBasicNodeMaterial } from 'three/webgpu';
import { Fn, uv, uint, vec3, floor, length, log, clamp, storage } from 'three/tsl';

// A fullscreen quad in an ortho scene, used to inspect a storage buffer as a
// heatmap (point-sampled). Lets step 2 be visually verified before the FFT
// exists: a correct spectrum shows a wind-oriented lobe centered on DC.
export function createDebugView() {
  const scene = new Scene();
  const camera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const mesh = new Mesh(new PlaneGeometry(2, 2), new MeshBasicNodeMaterial());
  scene.add(mesh);
  return { scene, camera, mesh };
}

// Heatmap of |field.xy| for an N*N complex storage buffer (vec2 or vec4).
export function spectrumDebugMaterial(buffer, N, { exposure = 0.12 } = {}) {
  const mat = new MeshBasicNodeMaterial();
  // Wrap the SAME GPU buffer in a separate read-only node. toReadOnly() mutates
  // the node it's called on, so reusing `buffer` here would flip the live
  // compute buffer to read-only and break the kernels that write to it.
  const type = buffer.value.itemSize >= 4 ? 'vec4' : 'vec2';
  const field = storage(buffer.value, type, buffer.value.count).toReadOnly();
  mat.colorNode = Fn(() => {
    const cell = floor(uv().mul(N));
    const x = uint(clamp(cell.x, 0, N - 1));
    const y = uint(clamp(cell.y, 0, N - 1));
    const idx = y.mul(uint(N)).add(x);
    const mag = length(field.element(idx).xy);
    const v = clamp(log(mag.mul(1000).add(1)).mul(exposure), 0, 1);
    return vec3(v.mul(0.7).add(0.05), v.mul(0.85).add(0.05), v.mul(1.0).add(0.08));
  })();
  return mat;
}
