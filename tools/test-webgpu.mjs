import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { launch } from 'puppeteer-core';

const capture = process.argv.includes('--capture');
const port = 5174;
const baseUrl = `http://127.0.0.1:${port}`;
const vite = spawn(
  process.execPath,
  ['node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', String(port), '--strictPort'],
  { cwd: process.cwd(), stdio: ['ignore', 'pipe', 'pipe'] },
);

async function waitForServer() {
  for (let attempt = 0; attempt < 80; attempt++) {
    try {
      const response = await fetch(baseUrl);
      if (response.ok) return;
    } catch {
      // Vite is still starting.
    }
    await new Promise((resolveWait) => setTimeout(resolveWait, 125));
  }
  throw new Error('Timed out starting the Poseidon validation fixture.');
}

const chrome = [
  process.env.CHROME_PATH,
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  process.env.LOCALAPPDATA && `${process.env.LOCALAPPDATA}/Google/Chrome/Application/chrome.exe`,
].find((candidate) => candidate && existsSync(candidate));

if (!chrome) {
  vite.kill();
  throw new Error('Chrome not found; set CHROME_PATH to run the WebGPU gate.');
}

let browser;
try {
  await waitForServer();
  browser = await launch({
    executablePath: chrome,
    headless: true,
    args: [
      '--headless=new',
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan',
      '--use-angle=default',
      '--no-sandbox',
      '--window-size=1600,900',
    ],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 900, deviceScaleFactor: 1 });
  const errors = [];
  page.on('pageerror', (error) => errors.push(String(error)));
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  const url = new URL(baseUrl);
  url.searchParams.set('shot', '1');
  url.searchParams.set('preset', 'deck');
  url.searchParams.set('t', capture ? '40' : '0.2');
  await page.goto(url.href, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    'window.__shotReady === true && window.__poseidonValidation?.fft',
    { timeout: 180000, polling: 200 },
  );
  const validation = await page.evaluate(() => window.__poseidonValidation);
  if (!validation.backendWebGPU || !validation.fft.pass) {
    throw new Error(`Poseidon WebGPU validation failed: ${JSON.stringify(validation)}`);
  }
  if (errors.length) {
    throw new Error(`Poseidon fixture logged errors: ${errors.slice(0, 6).join(' | ')}`);
  }
  if (capture) {
    const output = resolve('shots/golden/deck.png');
    await mkdir(resolve('shots/golden'), { recursive: true });
    await page.screenshot({ path: output, type: 'png' });
    console.log(`Captured ${output}`);
  }
  console.log(JSON.stringify(validation));
} finally {
  await browser?.close();
  vite.kill();
}
