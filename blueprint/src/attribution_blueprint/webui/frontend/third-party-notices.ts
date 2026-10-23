// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import type { Plugin } from 'vite'

const legalFile = /^(?:licen[cs]e|copying|notice|ofl)(?:[._-].*)?$/i
const licenseText = /^(?:licen[cs]e|copying|ofl)/i
// Packages offered under a choice of licenses, and the license this distribution uses.
const licenseChoices: Record<string, string> = {
  dompurify: 'Apache-2.0',
  jszip: 'MIT',
}
// Vite and Rolldown inject runtime helpers that have no module in the bundle graph.
const runtimeNotices = new URL('./third-party-licenses/vite-rolldown-runtime.txt', import.meta.url)

function packageDirectory(moduleId: string): string | undefined {
  if (moduleId.startsWith('\0')) return
  const filename = moduleId.split('?')[0]
  if (!path.isAbsolute(filename) || !fs.existsSync(filename)) return
  let directory = path.dirname(filename)
  while (directory.includes(`${path.sep}node_modules${path.sep}`)) {
    const manifest = path.join(directory, 'package.json')
    if (fs.existsSync(manifest) && JSON.parse(fs.readFileSync(manifest, 'utf8')).name) {
      return directory
    }
    directory = path.dirname(directory)
  }
}

// Slidev inlines icon data from @iconify-json collections into JS and CSS.
function iconCollections(directory: string): string[] {
  const manifest = path.join(directory, 'package.json')
  const { dependencies = {} } = JSON.parse(fs.readFileSync(manifest, 'utf8'))
  const require = createRequire(manifest)
  return Object.keys(dependencies)
    .filter(name => name.startsWith('@iconify-json/'))
    .map(name => path.dirname(require.resolve(`${name}/package.json`)))
}

function legalFiles(directory: string): string[] {
  return fs.readdirSync(directory, { withFileTypes: true })
    .sort((a, b) => a.name.localeCompare(b.name))
    .flatMap((entry) => {
      if (entry.isFile() && legalFile.test(entry.name)) return [path.join(directory, entry.name)]
      // Some packages, including ECharts, keep licenses for embedded code here.
      if (entry.isDirectory() && /^licen[cs]es$/i.test(entry.name)) {
        return legalFiles(path.join(directory, entry.name))
      }
      return []
    })
}

/**
 * Emit the repository notices together with the license and NOTICE files of
 * every npm package whose code or assets ship in the build.
 */
export function thirdPartyNotices(options: {
  configUrl: string
  sourceNotices: URL
  // Packages processed entirely by CSS plugins, which leave no module in the graph.
  assetPackages?: string[]
  // Upstream texts for exact package versions, as files in licenseDirectory.
  // Use them for packages published without a license file or with extra notices.
  licenseDirectory?: URL
  licenseSupplements?: Record<string, string>
  fileName?: string
}): Plugin {
  return {
    name: 'attribution-third-party-notices',
    apply: 'build',
    generateBundle(_output, bundle) {
      const directories = new Set<string>()
      const moduleIds = [
        ...Object.values(bundle).flatMap(output => output.type === 'chunk' ? output.moduleIds : []),
        // CSS and font modules can be inlined without remaining in a JS chunk.
        ...[...this.getModuleIds()].filter(id => /\.(?:css|woff2?|ttf|otf)(?:\?|$)/.test(id)),
      ]
      for (const moduleId of moduleIds) {
        const directory = packageDirectory(moduleId)
        if (directory) directories.add(fs.realpathSync(directory))
      }
      for (const name of options.assetPackages ?? []) {
        const root = path.dirname(fileURLToPath(options.configUrl))
        directories.add(fs.realpathSync(path.join(root, 'node_modules', name)))
      }
      for (const directory of [...directories]) {
        for (const collection of iconCollections(directory)) directories.add(fs.realpathSync(collection))
      }

      const read = (file: string) => {
        this.addWatchFile(file)
        return fs.readFileSync(file, 'utf8').trim()
      }
      const sections = [
        'Relative links in the first section refer to files in the NVIDIA Attribution source repository.\n',
        read(fileURLToPath(options.sourceNotices)),
        '\n# Bundled dependency licenses\n\nThis build includes code or assets from the components below.',
        `\n## Vite and Rolldown runtime helpers\n\n${read(fileURLToPath(runtimeNotices))}`,
      ]
      const packages = [...directories]
        .map(directory => ({
          directory,
          metadata: JSON.parse(fs.readFileSync(path.join(directory, 'package.json'), 'utf8')),
        }))
        .map(pkg => ({ ...pkg, id: `${pkg.metadata.name}@${pkg.metadata.version}` }))
        .filter((pkg, index, all) => all.findIndex(other => other.id === pkg.id) === index)
        .sort((a, b) => a.id.localeCompare(b.id))

      const missing: string[] = []
      for (const { directory, metadata, id } of packages) {
        const files = legalFiles(directory)
        const supplementName = options.licenseSupplements?.[id]
        const supplement = supplementName && options.licenseDirectory
          ? path.join(fileURLToPath(options.licenseDirectory), supplementName)
          : undefined
        if (!supplement && !files.some(file => licenseText.test(path.basename(file)))) missing.push(id)
        sections.push(`\n## ${id}\n\nLicense: ${metadata.license ?? 'see license text below'}.`)
        const choice = licenseChoices[metadata.name]
        if (choice) {
          sections.push(`\nThis distribution uses ${metadata.name} under ${choice}; other license alternatives reproduced below are not selected.`)
        }
        for (const file of files) sections.push(`\n### ${path.relative(directory, file)}\n\n${read(file)}`)
        if (supplement) sections.push(`\n### Upstream license text\n\n${read(supplement)}`)
      }
      if (missing.length) {
        this.error(`No license text found for ${missing.join(', ')}; add their upstream licenses before distributing the build`)
      }
      this.emitFile({
        type: 'asset',
        fileName: options.fileName ?? 'assets/THIRD_PARTY_NOTICES.txt',
        source: `${sections.join('\n')}\n`,
      })
    },
  }
}
