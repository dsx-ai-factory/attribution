// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'
import type { ECharts, EChartsOption } from 'echarts'
import { chartTheme } from '@/lib/tokens'

// Chart chrome (fonts, axis and tooltip colors) comes from the token theme,
// registered once before the first chart as house charts register theirs, so
// an option carries only its data, layout and series colors.
const THEME = 'attribution'
let themeRegistered = false

function registerThemeOnce() {
  if (themeRegistered) return
  echarts.registerTheme(THEME, chartTheme())
  themeRegistered = true
}

export function useECharts(
  ref: React.RefObject<HTMLDivElement | null>,
  option: EChartsOption,
  onEvents?: {
    mouseover?: (params: echarts.ECElementEvent) => void
    mouseout?: (params: echarts.ECElementEvent) => void
    click?: (params: echarts.ECElementEvent) => void
  },
) {
  const chartRef = useRef<ECharts | null>(null)
  const handlersRef = useRef(onEvents)

  useEffect(() => {
    handlersRef.current = onEvents
  }, [onEvents])

  useEffect(() => {
    if (!ref.current) return
    registerThemeOnce()
    const chart = echarts.init(ref.current, THEME)
    chartRef.current = chart
    const resize = () => chart.resize()
    window.addEventListener('resize', resize)
    const ro = new ResizeObserver(() => chart.resize())
    ro.observe(ref.current)

    if (handlersRef.current?.mouseover) {
      chart.on('mouseover', (p) => handlersRef.current?.mouseover?.(p as echarts.ECElementEvent))
    }
    if (handlersRef.current?.mouseout) {
      chart.on('mouseout', (p) => handlersRef.current?.mouseout?.(p as echarts.ECElementEvent))
    }
    if (handlersRef.current?.click) {
      chart.on('click', (p) => handlersRef.current?.click?.(p as echarts.ECElementEvent))
    }
    return () => {
      window.removeEventListener('resize', resize)
      ro.disconnect()
      chart.dispose()
      chartRef.current = null
    }
  }, [ref])

  // Each option is the whole chart, so it replaces the last one instead of
  // merging into it: a merge keeps series the new data no longer has and adds
  // new ones after the old, out of the option's (and a stack's) order. Series
  // keep their views by name, so their changes still animate.
  useEffect(() => {
    chartRef.current?.setOption(option, { notMerge: true })
  }, [option])

  return chartRef
}
