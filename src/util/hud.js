// Minimal on-screen HUD + error overlay. Because this build is judged visually
// in the browser (no headless run), the frame time and any WebGPU validation
// errors need to be visible on the page itself.
export function createHUD() {
  const hud = document.createElement('div');
  hud.id = 'hud';
  hud.textContent = 'initializing…';
  const err = document.createElement('div');
  err.id = 'err';
  document.body.append(hud, err);

  return {
    set(text) {
      hud.textContent = text;
    },
    error(msg) {
      err.style.display = 'block';
      err.textContent += msg + '\n';
      console.error('[FFT-ocean] ' + msg);
    },
  };
}
