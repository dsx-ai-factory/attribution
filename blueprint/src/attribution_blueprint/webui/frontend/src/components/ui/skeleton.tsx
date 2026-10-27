// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react"

import { cn } from "@/lib/utils"

type SkeletonHeight = "4" | "5" | "6" | "7" | "16" | "40"
type SkeletonWidth = "1/2" | "2/3" | "3/4"

const heightMap: Record<SkeletonHeight, string> = {
  "4": "h-4",
  "5": "h-5",
  "6": "h-6",
  "7": "h-7",
  "16": "h-16",
  "40": "h-40",
}

const widthMap: Record<SkeletonWidth, string> = {
  "1/2": "w-1/2",
  "2/3": "w-2/3",
  "3/4": "w-3/4",
}

// The house Skeleton. Its background-active block is decorative (1.10:1 on the
// card, as in house) and pulses while content loads. Its height and width
// presets are the ones the placeholders here use; the other house presets are
// dropped, as nothing here uses them.
function Skeleton({
  className,
  height,
  width,
  ...props
}: React.ComponentProps<"div"> & {
  height?: SkeletonHeight
  width?: SkeletonWidth
}) {
  return (
    <div
      data-slot="skeleton"
      className={cn(
        "animate-pulse rounded-md bg-background-active",
        height && heightMap[height],
        width && widthMap[width],
        className
      )}
      {...props}
    />
  )
}

export { Skeleton }
