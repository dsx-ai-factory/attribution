// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { BaseEdge, EdgeLabelRenderer, type Edge, type EdgeProps } from '@xyflow/react'
import { EDGE_LABEL_HEIGHT, type TracePathData } from './layout'

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
    // Unit vectors keep each tangent on its segment, a radius from the
    // corner and never past the segment's middle.
    path += ` L ${point.x - (point.x - previous.x) / incomingLength * radius} ${point.y - (point.y - previous.y) / incomingLength * radius}`
    path += ` Q ${point.x} ${point.y} ${point.x + (next.x - point.x) / outgoingLength * radius} ${point.y + (next.y - point.y) / outgoingLength * radius}`
  }
  return `${path} L ${points.at(-1)!.x} ${points.at(-1)!.y}`
}

// A piece of a route (TracePathData), through its own corners from end to
// end, with the labels on it. Its ends at steps are the layout's points on
// the steps' borders, not where React Flow measures their handles on screen:
// that measurement puts an end 3px off the border (a handle's outer edge) and
// scales with any transform around the canvas as it mounts (a dialog zooming
// in drew every end 95% of the way). A label names relationships on the
// traced chain only (the view passes no other), so it wears the trace color.
// The 20px layout box holds one 16px line between 1px borders and 1px
// padding. Like the piece, a label is part of the picture: assistive
// technology reads each relationship by its name.
export function TraceEdge({ id, markerEnd, style, data }: EdgeProps<Edge<TracePathData>>) {
  const path = roundedPath(data!.points)
  return (
    <>
      <BaseEdge id={id} path={path} markerEnd={markerEnd} style={style} />
      {data!.labels.length > 0 && (
        <EdgeLabelRenderer>
          {data!.labels.map((label) => (
            <div
              key={`${label.x} ${label.y}`}
              data-slot="decision-edge-label"
              aria-hidden="true"
              className="pointer-events-none absolute box-border overflow-hidden rounded-sm border border-graph-trace/50 bg-background-secondary px-2 py-px text-center text-xs leading-4 text-ellipsis whitespace-nowrap text-graph-trace"
              style={{ width: label.width, height: EDGE_LABEL_HEIGHT, transform: `translate(-50%, -50%) translate(${label.x}px, ${label.y}px)` }}
            >
              {label.text}
            </div>
          ))}
        </EdgeLabelRenderer>
      )}
    </>
  )
}
