#!/usr/bin/env node
'use strict';

// Bump package.json's "version" field before each installer build.
//
// Why: app.getVersion() reads from package.json, and the installer artifact
// filename uses ${version}, so a single bump here flows everywhere — the
// in-app status bar, the about screen, and the NSIS setup file's name all
// pick up the new number without any other edit.
//
// Default behavior (no args): bump the PATCH digit (1.0.4 → 1.0.5). This is
// what the build scripts call before electron-builder runs.
//
// Manual overrides (handy from the CLI):
//   node scripts/bump-version.js patch    → 1.0.5  (default)
//   node scripts/bump-version.js minor    → 1.1.0  (resets patch)
//   node scripts/bump-version.js major    → 2.0.0  (resets minor + patch)
//   node scripts/bump-version.js set 1.2.3 → set explicitly
//   node scripts/bump-version.js --dry    → print what would change, write nothing
//
// Exits 0 on success, non-zero on bad input. Intended to be called from the
// "version:bump" npm script and chained ahead of electron-builder.

const fs = require('fs');
const path = require('path');

const pkgPath = path.join(__dirname, '..', 'package.json');

function parseArgs(argv) {
  const args = argv.slice(2);
  const dry = args.includes('--dry');
  const positional = args.filter(a => !a.startsWith('--'));
  let mode = positional[0] || 'patch';
  let explicit = null;
  if (mode === 'set') {
    explicit = positional[1];
    if (!explicit || !/^\d+\.\d+\.\d+$/.test(explicit)) {
      console.error('bump-version: "set" requires a SemVer like 1.2.3');
      process.exit(2);
    }
  } else if (!['patch', 'minor', 'major'].includes(mode)) {
    console.error(`bump-version: unknown mode "${mode}" (expected patch|minor|major|set)`);
    process.exit(2);
  }
  return { mode, explicit, dry };
}

function bump(version, mode) {
  const m = /^(\d+)\.(\d+)\.(\d+)/.exec(version);
  if (!m) throw new Error(`bump-version: existing version "${version}" is not SemVer`);
  let [, major, minor, patch] = m;
  major = +major; minor = +minor; patch = +patch;
  if (mode === 'patch') patch += 1;
  else if (mode === 'minor') { minor += 1; patch = 0; }
  else if (mode === 'major') { major += 1; minor = 0; patch = 0; }
  return `${major}.${minor}.${patch}`;
}

function main() {
  const { mode, explicit, dry } = parseArgs(process.argv);
  const raw = fs.readFileSync(pkgPath, 'utf-8');
  const pkg = JSON.parse(raw);
  const before = pkg.version;
  const after = explicit ? explicit : bump(before, mode);
  if (after === before) {
    console.log(`bump-version: no change (${before})`);
    return;
  }
  pkg.version = after;
  if (dry) {
    console.log(`bump-version (dry): ${before} → ${after}`);
    return;
  }
  // Preserve trailing newline + 2-space indent that npm/editors expect.
  const ending = raw.endsWith('\n') ? '\n' : '';
  fs.writeFileSync(pkgPath, JSON.stringify(pkg, null, 2) + ending);
  console.log(`bump-version: ${before} → ${after}`);
}

main();
