import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'
import type { ECharts, EChartsOption } from 'echarts'

export const ECHART_COMMON = {
  textStyle: {
    fontFamily: 'JetBrains Mono Variable, ui-monospace, monospace',
  },
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
    const chart = echarts.init(ref.current)
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

  useEffect(() => {
    chartRef.current?.setOption(option, { notMerge: false })
  }, [option])

  return chartRef
}
