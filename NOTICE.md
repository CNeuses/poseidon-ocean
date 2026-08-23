# Attribution and package boundary

`@fantasai/poseidon-ocean` is maintained as a fork of Owen Yuwono's MIT-licensed
[Poseidon](https://github.com/owenyuwono/poseidon), starting at upstream commit
`671053b812fcbffe8ecc4668eaa6ab7ffeb63287`.

The spectral simulation and water shader retain Poseidon's original history and
the upstream `LICENSE`. Spectrum/FFT portions additionally acknowledge the MIT
`gasgiant/FFT-Ocean` work cited by the upstream project.

The root package export contains simulation, material, geometry, configuration,
validation, and disposal APIs only. Demo sky images, GUI, camera controls, DOM,
and render-loop ownership are not part of the published package files.
