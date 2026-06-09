import { Sprite, SpriteNodeMaterial, NormalBlending, Vector2, Vector3 } from 'three/webgpu';
import {
  Fn, instanceIndex, instancedArray, uniform, float, vec2, vec3, vec4, ivec2,
  hash, textureLoad, floor, cos, sin, sqrt, max, saturate, length, uv, If,
} from 'three/tsl';

const TAU = Math.PI * 2;

// GPU ballistic spray. A pool of particles lives in storage buffers. Each frame
// a compute pass either (dead particle) tries to respawn at a randomly-sampled
// breaking crest — read straight from the foam/turbulence in the cascade
// displacement maps — or (live particle) integrates ballistically and ages.
// Spray is wind-dominated: it's ripped off the crest and blown downwind, with
// only a small upward kick, then carried by wind while gravity pulls it down.
// Rendered as instanced soft additive billboards; dead particles get zero scale
// so they cost nothing.
export function createSpray(renderer, { cascades, lengthScales, N, shading, count = 24000 }) {
  const posLife = instancedArray(count, 'vec4'); // xyz position, w = remaining life
  const velLife = instancedArray(count, 'vec4'); // xyz velocity, w = total life

  const u = {
    dt: uniform(1 / 60),
    seed: uniform(0),
    camXZ: uniform(new Vector2()),
    wind: uniform(new Vector3()),
    emitRadius: uniform(130),
    breakThreshold: uniform(0.85), // only the hardest-breaking crests spawn spray
    emitChance: uniform(0.3), // per-frame spawn probability at a break
    burst: uniform(2.5), // upward kick (small — wind dominates)
    size: uniform(0.45), // droplet size (m)
    opacity: uniform(0.55),
  };

  // surface height + breaking strength at a world xz, read from the foam maps
  const sampleOcean = (xz) => {
    const h = float(0).toVar();
    const brk = float(-10).toVar();
    cascades.forEach((c, idx) => {
      const uvc = xz.div(lengthScales[idx]);
      const texel = ivec2(uvc.sub(floor(uvc)).mul(N));
      const d = textureLoad(c.displacement, texel);
      h.addAssign(d.y);
      if (idx < cascades.length - 1) brk.assign(max(brk, shading.foamThreshold.sub(d.w)));
    });
    return { h, brk };
  };

  const kernel = Fn(() => {
    const i = instanceIndex;
    const pl = posLife.element(i);
    const vl = velLife.element(i);
    const pos = pl.xyz.toVar();
    const life = pl.w.toVar();
    const v = vl.xyz.toVar();
    const total = vl.w.toVar();

    If(life.lessThanEqual(0), () => {
      const s = float(i).mul(2.17).add(u.seed);
      const r1 = hash(s);
      const r2 = hash(s.add(11.0));
      const r3 = hash(s.add(23.0));
      const r4 = hash(s.add(37.0));
      const ang = r1.mul(TAU);
      const rad = sqrt(r2).mul(u.emitRadius);
      const xz = u.camXZ.add(vec2(cos(ang), sin(ang)).mul(rad));
      const oc = sampleOcean(xz);
      If(oc.brk.greaterThan(u.breakThreshold).and(r3.lessThan(u.emitChance)), () => {
        pos.assign(vec3(xz.x, oc.h.add(0.3), xz.y));
        // small upward kick + strong downwind drift, both varied
        const up = u.burst.mul(float(0.5).add(oc.brk.mul(0.4))).mul(float(0.6).add(r4.mul(0.8)));
        const downwind = u.wind.mul(float(1.0).add(r2.mul(1.8)));
        const jit = vec3(r1.sub(0.5), r3.mul(0.4), r4.sub(0.5)).mul(2.0);
        v.assign(vec3(0, up, 0).add(downwind).add(jit));
        total.assign(float(0.8).add(r2.mul(1.4)));
        life.assign(total);
      }).Else(() => {
        life.assign(0);
      });
    }).Else(() => {
      v.y.addAssign(float(-9.8).mul(u.dt));
      v.addAssign(u.wind.mul(0.35).mul(u.dt));
      v.mulAssign(max(float(1).sub(float(0.6).mul(u.dt)), 0));
      pos.addAssign(v.mul(u.dt));
      life.subAssign(u.dt);
    });

    pl.assign(vec4(pos, life));
    vl.assign(vec4(v, total));
  })().compute(count);

  // render: instanced soft additive sprites; zero scale when dead (no overdraw)
  const pa = posLife.toAttribute();
  const va = velLife.toAttribute();
  const lifeFrac = saturate(pa.w.div(max(va.w, float(0.001))));
  const alive = saturate(pa.w.mul(1000));

  const material = new SpriteNodeMaterial();
  material.positionNode = pa.xyz;
  material.scaleNode = u.size.mul(float(0.6).add(saturate(float(1).sub(lifeFrac)).mul(1.1))).mul(alive);
  const soft = saturate(float(1).sub(length(uv().sub(0.5)).mul(2)));
  material.colorNode = vec4(shading.foamColor, soft.mul(soft).mul(lifeFrac).mul(u.opacity));
  material.transparent = true;
  material.depthWrite = false;
  material.blending = NormalBlending;

  const mesh = new Sprite(material);
  mesh.count = count;
  mesh.frustumCulled = false;

  let frame = 0;
  const update = (dt, camPos, windVec) => {
    frame = (frame + 1) % 100000;
    u.dt.value = Math.min(dt, 0.05);
    u.seed.value = frame * 1.7;
    u.camXZ.value.set(camPos.x, camPos.z);
    u.wind.value.copy(windVec);
    renderer.compute(kernel);
  };

  return { mesh, update, uniforms: u };
}
