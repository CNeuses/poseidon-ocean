# Poseidon — Real-time FFT Ocean (WebGPU)

A real-time, Tessendorf-style **spectral ocean** built in [Three.js](https://threejs.org/)
with the **WebGPURenderer + TSL**, where the inverse FFT runs entirely in
**WebGPU compute shaders**.

## Features

- **GPU butterfly IFFT** (Stockham, precomputed twiddle/index buffer), validated
  in isolation against an analytic impulse/frequency before wiring to the ocean.
- **3 wave cascades** (250 / 17 / 5 m) over disjoint wavenumber bands for swell +
  ripples without visible tiling.
- **Horvath / JONSWAP directional spectrum** — wind-sea + swell, with TMA depth
  correction, Donelan-Banner spreading and short-wave fade.
- **Choppy** horizontal displacement, **fold-aware normals** from the slope FFTs.
- **Foam** from the displacement **Jacobian**, accumulated with build/decay so
  whitecaps linger and dissipate; bubbly texture + sun-lit shading.
- **Shading**: Fresnel sky reflection, reflected-sun glitter, subsurface scatter,
  sub-grid detail noise, depth-based water color.
- **lil-gui** panel for live tuning (wind, choppiness, foam, sun, colors).
- Optional **GPU ballistic spray-particle** system (disabled by default).

## Run

```bash
npm install
npm run dev
```

Open the printed URL in a **WebGPU-capable** browser (Chrome/Edge 113+, Safari 18+).
This targets WebGPU only — there is no WebGL fallback.

Views (keys): `F` ocean · `5` height map · `1/2/3` cascade spectra.

## Credits

Spectrum and FFT techniques adapted from
[gasgiant/FFT-Ocean](https://github.com/gasgiant/FFT-Ocean) (MIT), implementing
Tessendorf 2001 (*Simulating Ocean Water*) and Horvath 2015 (*Empirical
Directional Wave Spectra for Computer Graphics*).
