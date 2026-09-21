// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import { BaseEdge, EdgeLabelRenderer, getSmoothStepPath, type Edge, type EdgeProps } from '@xyflow/react'
import { EDGE_LABEL_HEIGHT, type TraceEdgeData } from './layout'

function roundedPath(points: { x: number; y: number }[]) {
  let path = `M ${points[0].x} ${points[0].y}`
  for (let index = 1; index < points.length - 1; index++) {
    const previous = points[index - 1]
    const point = points[index]
    const next = points[index + 1]
    const incomingLength = Math.hypot(point.x - previous.x, point.y - previous.y)
    const outgoingLength = Math.hypot(next.x - point.x, next.y - point.y)
    const radius = Math.min(12, incomingLength / 2, outgoingLength / 2)
    if (radius === 0) {
      path += ` L ${point.x} ${point.y}`
      continue
    }
    // Measured handles have subpixel offsets. Unit vectors keep each tangent
    // on its segment instead of amplifying those offsets into a full radius.
    path += ` L ${point.x - (point.x - previous.x) / incomingLength * radius} ${point.y - (point.y - previous.y) / incomingLength * radius}`
    path += ` Q ${point.x} ${point.y} ${point.x + (next.x - point.x) / outgoingLength * radius} ${point.y + (next.y - point.y) / outgoingLength * radius}`
  }
  return `${path} L ${points.at(-1)!.x} ${points.at(-1)!.y}`
}

export function TraceEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, markerEnd, style, label, data }: EdgeProps<Edge<TraceEdgeData>>) {
  const [defaultPath, defaultLabelX, defaultLabelY] = getSmoothStepPath({
    sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, borderRadius: 12,
  })
  const path = data ? roundedPath([
    { x: sourceX, y: sourceY },
    ...data.points,
    { x: targetX, y: targetY },
  ]) : defaultPath
  const labelX = data?.labelX ?? defaultLabelX
  const labelY = data?.labelY ?? defaultLabelY
  return (
    <>
      <BaseEdge id={id} path={path} markerEnd={markerEnd} style={style} />
      {label != null && (
        <EdgeLabelRenderer>
          <div
            data-slot="decision-edge-label"
            className="pointer-events-none absolute box-border overflow-hidden rounded-sm border border-border bg-background-secondary px-2 py-0.5 text-center font-sans text-[10px] leading-[14px] text-ellipsis whitespace-nowrap text-foreground-secondary shadow-sm"
            title={data?.edgeType}
            style={{ width: data?.labelWidth, height: EDGE_LABEL_HEIGHT, transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}
          >
            {label}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  )
}
