# Poseidon

Real-time Tessendorf-style ocean in [Three.js](https://threejs.org/) (WebGPURenderer + TSL). The inverse FFT runs in WebGPU compute shaders.

## What's in it

- Stockham butterfly IFFT on the GPU, with precomputed twiddle/index buffers
- 3 wave cascades (250 / 17 / 5 m) on disjoint wavenumber bands, so you get swell and ripples without visible tiling
- JONSWAP/Horvath directional spectrum: wind sea + swell, TMA depth correction, Donelan-Banner spreading
- Choppy horizontal displacement, normals from the slope FFTs
- Foam from the displacement Jacobian, with build/decay accumulation so whitecaps linger a bit before fading
- Exact dielectric Fresnel (n = 1.34) both ways across the interface, so a
  submerged camera gets a real Snell's window — the whole sky inside a 48.3°
  cone — with total internal reflection outside it
- Subsurface scatter built from the three factors it actually has: entry Fresnel
  on the wave's far face, a Henyey-Greenstein forward lobe for the exit, and
  diffusion transmittance over a refracted path, so a thin lip goes warm and a
  thick one goes jade off one exponential
- Sun glitter (GGX with a knee), accumulated-Jacobian whitecaps, self-composited
  aerial perspective
- lil-gui panel for live tuning (wind, choppiness, foam, sun, colors)

## Run

```bash
npm install
npm run dev
```

Open the printed URL in a WebGPU-capable browser (Chrome/Edge 113+, Safari 18+). WebGPU only, no WebGL fallback.

Camera is an Unreal-style free look: hold right mouse to look, `WASD` to fly,
`Q`/`E` down/up, shift to boost, mouse wheel sets speed.

Views: `F` ocean, `5` height map, `1/2/3` cascade spectra. `+`/`-` choppiness.

## Credits

Spectrum and FFT techniques adapted from [gasgiant/FFT-Ocean](https://github.com/gasgiant/FFT-Ocean) (MIT), based on Tessendorf 2001 (*Simulating Ocean Water*) and Horvath 2015 (*Empirical Directional Wave Spectra for Computer Graphics*).

Water optics (`src/ocean/water.js`) use the Jerlov coastal-water inherent optical properties tabulated by Solonenko & Mobley 2015, [*Inherent optical properties of Jerlov water types*](https://doi.org/10.1364/AO.54.005392), Appl. Opt. 54(17):5392.
