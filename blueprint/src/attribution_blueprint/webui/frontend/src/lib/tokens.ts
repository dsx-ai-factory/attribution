// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { OutcomeKey } from '@/interfaces/attribution'

// The only way from the tokens in src/styles/globals.css into renderers that
// cannot resolve CSS themselves (ECharts paints on a canvas). DOM markup,
// SVG included, keeps writing var(--…) directly.

function readToken(name: string): string {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  if (!value) throw new Error(`CSS token ${name} is not defined`)
  return value
}

const colors = new Map<string, string>()
const hex = (byte: number) => byte.toString(16).padStart(2, '0')

// zrender parses hex, rgb() and hsl() only, while most tokens are oklch, so the
// token is painted on a 1×1 canvas and read back as #rrggbb (#rrggbbaa when
// translucent). Tokens are fixed for the page's lifetime, so each is read once.
export function readColor(name: string): string {
  const cached = colors.get(name)
  if (cached) return cached
  const value = readToken(name)
  if (!CSS.supports('color', value)) throw new Error(`CSS token ${name} is not a color: ${value}`)
  const canvas = document.createElement('canvas')
  canvas.width = 1
  canvas.height = 1
  const context = canvas.getContext('2d')
  if (!context) throw new Error(`No 2D canvas context to resolve ${name}`)
  context.fillStyle = value
  context.fillRect(0, 0, 1, 1)
  const [r, g, b, a] = context.getImageData(0, 0, 1, 1).data
  const color = `#${hex(r)}${hex(g)}${hex(b)}${a === 255 ? '' : hex(a)}`
  colors.set(name, color)
  return color
}

export function readFont(name: string): string {
  return readToken(name)
}

export function outcomeColor(outcome: OutcomeKey): string {
  return readColor(`--outcome-${outcome}`)
}

// The two chart faces: sans for chrome and labels, mono for numbers.
export function chartFonts() {
  return { sans: readFont('--app-font-sans'), mono: readFont('--app-font-mono') }
}

// An ECharts theme started from the house dark theme (lepton
// metric-chart-panel/themes.ts, tooltip chrome from its chart-utils.ts) and
// tuned for Attribution: NVIDIA Sans at the default 400 weight instead of 500;
// muted chrome text (house #555 axis text is 2.51:1); one hairline for
// the axis and gridlines (card border), no ticks, no category gridlines; mono
// for value axes and bar values; a translucent hover band because the axis
// pointer paints above the bars. No `color` palette: series colors only ever
// come from outcomeColor(), never from rank.
export function chartTheme() {
  const { sans, mono } = chartFonts()
  const foreground = readColor('--foreground')
  const grid = readColor('--chart-grid')
  const label = { color: readColor('--foreground-muted'), fontFamily: sans, fontSize: 12 }
  return {
    textStyle: label,
    categoryAxis: {
      axisLine: { show: true, lineStyle: { color: grid } },
      axisTick: { show: false },
      axisLabel: label,
      splitLine: { show: false },
    },
    valueAxis: {
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { ...label, fontFamily: mono },
      nameTextStyle: label,
      splitLine: { show: true, lineStyle: { color: grid, type: 'dashed' as const } },
    },
    bar: { label: { color: foreground, fontFamily: mono, fontSize: 12 } },
    tooltip: {
      // A tooltip stays inside its chart, never wider than it, its lines
      // wrapping: the chart sits in its card inside the page's gutter, so the
      // tooltip keeps clear of the window's edge (floatingEdge) and of the
      // card's clipping edge. The house puts it on <body> instead, where a
      // narrow window could push it past the edge.
      confine: true,
      extraCssText: 'max-width: 100%; white-space: normal; overflow-wrap: anywhere;',
      backgroundColor: readColor('--background'),
      borderColor: readColor('--border'),
      borderWidth: 1,
      padding: [8, 12],
      shadowColor: 'transparent',
      textStyle: { color: foreground, fontFamily: sans, fontSize: 12 },
      // 4% white over the card composites to ≈ --background-active, the house hover surface.
      axisPointer: { shadowStyle: { color: foreground, opacity: 0.04 } },
    },
  }
}
