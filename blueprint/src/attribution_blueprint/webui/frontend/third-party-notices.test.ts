// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import { afterEach, describe, expect, it } from 'vitest'
import { build } from 'vite'
import { thirdPartyNotices } from './third-party-notices'

const roots: string[] = []

afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true })
})

function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'attribution-notices-'))
  roots.push(root)
  const dependency = path.join(root, 'node_modules', 'fixture-dependency')
  fs.mkdirSync(dependency, { recursive: true })
  fs.writeFileSync(path.join(dependency, 'package.json'), JSON.stringify({
    name: 'fixture-dependency', version: '1.0.0', license: 'MIT', main: 'index.js',
  }))
  fs.writeFileSync(path.join(dependency, 'index.js'), 'export const value = 1')
  fs.writeFileSync(path.join(root, 'entry.js'), "export { value } from 'fixture-dependency'")
  const source = path.join(root, 'THIRD_PARTY_NOTICES.md')
  fs.writeFileSync(source, '# Source notices\n\nCopied component copyright and permission notice.\n')
  const licenses = path.join(root, 'licenses-supplement')
  fs.mkdirSync(licenses)
  const options = {
    configUrl: pathToFileURL(path.join(root, 'vite.config.ts')).href,
    sourceNotices: pathToFileURL(source),
    licenseDirectory: pathToFileURL(`${licenses}/`),
  }
  async function bundle(overrides: Partial<Parameters<typeof thirdPartyNotices>[0]> = {}) {
    const result = await build({
      root, configFile: false, publicDir: false, logLevel: 'silent',
      plugins: [thirdPartyNotices({ ...options, ...overrides })],
      build: { write: false, lib: { entry: path.join(root, 'entry.js'), formats: ['es'] } },
    })
    const output = Array.isArray(result) ? result[0].output : 'output' in result ? result.output : []
    const notices = output.find(file => file.fileName === 'assets/THIRD_PARTY_NOTICES.txt')
    if (!notices || notices.type !== 'asset') throw new Error('Notice file missing from built assets')
    return String(notices.source)
  }
  return { root, dependency, licenses, bundle }
}

describe('third-party notices in distributed builds', () => {
  it('ships copied-source notices, upstream NOTICE, and embedded-code licenses together', async () => {
    const { dependency, bundle } = fixture()
    fs.writeFileSync(path.join(dependency, 'LICENSE'), 'Upstream permission notice.\n')
    fs.writeFileSync(path.join(dependency, 'NOTICE'), 'Upstream attribution that must travel with the binary.\n')
    fs.mkdirSync(path.join(dependency, 'licenses'))
    fs.writeFileSync(path.join(dependency, 'licenses', 'LICENSE-embedded'), 'Embedded component copyright and redistribution conditions.\n')
    const text = await bundle()
    expect(text).toContain('Copied component copyright and permission notice.')
    expect(text).toContain('## Vite and Rolldown runtime helpers')
    expect(text).toContain('Copyright (c) 2024-present VoidZero Inc. & Contributors')
    expect(text).toContain('## fixture-dependency@1.0.0')
    expect(text).toContain('Upstream permission notice.')
    expect(text).toContain('Upstream attribution that must travel with the binary.')
    expect(text).toContain('Embedded component copyright and redistribution conditions.')
  })

  it('includes a CSS/font package even when it contributes no JavaScript', async () => {
    const { root, dependency, bundle } = fixture()
    fs.writeFileSync(path.join(root, 'entry.js'), 'export const value = 1')
    fs.writeFileSync(path.join(dependency, 'OFL.txt'), 'Font copyright and redistribution conditions.\n')
    expect(await bundle({ assetPackages: ['fixture-dependency'] }))
      .toContain('Font copyright and redistribution conditions.')
  })

  it('includes icon collections that a bundled package inlines', async () => {
    const { root, dependency, bundle } = fixture()
    fs.writeFileSync(path.join(dependency, 'LICENSE'), 'Upstream permission notice.\n')
    const manifest = path.join(dependency, 'package.json')
    fs.writeFileSync(manifest, JSON.stringify({
      ...JSON.parse(fs.readFileSync(manifest, 'utf8')),
      dependencies: { '@iconify-json/fixture': '1.0.0' },
    }))
    const icons = path.join(root, 'node_modules', '@iconify-json', 'fixture')
    fs.mkdirSync(icons, { recursive: true })
    fs.writeFileSync(path.join(icons, 'package.json'), JSON.stringify({ name: '@iconify-json/fixture', version: '1.0.0' }))
    fs.writeFileSync(path.join(icons, 'LICENSE'), 'Icon set permission notice.\n')
    const text = await bundle()
    expect(text).toContain('## @iconify-json/fixture@1.0.0')
    expect(text).toContain('Icon set permission notice.')
  })

  it('refuses to distribute a package that has only a license identifier', async () => {
    const { bundle } = fixture()
    await expect(bundle()).rejects.toThrow('No license text found for fixture-dependency@1.0.0')
  })

  it('allows a preserved upstream license only for the explicitly reviewed version', async () => {
    const { licenses, bundle } = fixture()
    fs.writeFileSync(path.join(licenses, 'fixture.txt'), 'Complete upstream permission notice.\n')
    const text = await bundle({ licenseSupplements: { 'fixture-dependency@1.0.0': 'fixture.txt' } })
    expect(text).toContain('Complete upstream permission notice.')
    await expect(bundle({ licenseSupplements: { 'fixture-dependency@2.0.0': 'fixture.txt' } }))
      .rejects.toThrow('No license text found for fixture-dependency@1.0.0')
  })
})
