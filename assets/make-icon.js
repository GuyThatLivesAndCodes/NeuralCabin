// Writes a minimal 256x256 PNG with a white hexagon on black. No deps.
// PNG output via a tiny hand-rolled encoder.

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const W = 256, H = 256;
const pixels = Buffer.alloc(W * H * 4);

// Fill with black, fully opaque
for (let i = 0; i < W * H; i++) {
  pixels[i * 4 + 0] = 10;
  pixels[i * 4 + 1] = 10;
  pixels[i * 4 + 2] = 10;
  pixels[i * 4 + 3] = 255;
}

// Draw a filled hexagon centered, white
function drawHex(cx, cy, r) {
  const verts = [];
  for (let i = 0; i < 6; i++) {
    const a = Math.PI / 6 + i * Math.PI / 3;
    verts.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  }
  const minX = Math.floor(Math.min(...verts.map(v => v[0])));
  const maxX = Math.ceil(Math.max(...verts.map(v => v[0])));
  const minY = Math.floor(Math.min(...verts.map(v => v[1])));
  const maxY = Math.ceil(Math.max(...verts.map(v => v[1])));
  function inside(x, y) {
    let in_ = false;
    for (let i = 0, j = verts.length - 1; i < verts.length; j = i++) {
      const [xi, yi] = verts[i], [xj, yj] = verts[j];
      if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) in_ = !in_;
    }
    return in_;
  }
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      if (x < 0 || y < 0 || x >= W || y >= H) continue;
      if (inside(x + 0.5, y + 0.5)) {
        const i = (y * W + x) * 4;
        pixels[i] = 245; pixels[i + 1] = 245; pixels[i + 2] = 245; pixels[i + 3] = 255;
      }
    }
  }
}

drawHex(W / 2, H / 2, 90);

// Tiny PNG encoder
function crc32(buf) {
  let c;
  const table = [];
  for (let n = 0; n < 256; n++) {
    c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    table[n] = c;
  }
  let crc = 0xFFFFFFFF;
  for (let i = 0; i < buf.length; i++) crc = table[(crc ^ buf[i]) & 0xFF] ^ (crc >>> 8);
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length, 0);
  const typeBuf = Buffer.from(type);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])), 0);
  return Buffer.concat([len, typeBuf, data, crc]);
}

const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
const ihdr = Buffer.alloc(13);
ihdr.writeUInt32BE(W, 0); ihdr.writeUInt32BE(H, 4);
ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;

// Add filter byte per row
const raw = Buffer.alloc(H * (W * 4 + 1));
for (let y = 0; y < H; y++) {
  raw[y * (W * 4 + 1)] = 0;
  pixels.copy(raw, y * (W * 4 + 1) + 1, y * W * 4, (y + 1) * W * 4);
}
const idat = zlib.deflateSync(raw);

const out = Buffer.concat([
  sig,
  chunk('IHDR', ihdr),
  chunk('IDAT', idat),
  chunk('IEND', Buffer.alloc(0))
]);

fs.writeFileSync(path.join(__dirname, 'icon.png'), out);
console.log('Wrote icon.png', out.length, 'bytes');
