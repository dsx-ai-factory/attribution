// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useLayoutEffect, type RefObject } from 'react'

/**
 * Tells the page how tall a page's own sticky band under the app bar is, for
 * as long as the band is mounted: the root's --page-sticky-height, which
 * globals.css adds to the page's scroll padding, follows the band's height as
 * it wraps or resizes. So keyboard focus, scrollIntoView and anchors bring
 * content in below the band instead of behind it.
 */
export function useStickyBand(ref: RefObject<HTMLElement | null>) {
  useLayoutEffect(() => {
    const band = ref.current
    if (!band) return
    const root = document.documentElement
    const measure = () => root.style.setProperty('--page-sticky-height', `${band.offsetHeight}px`)
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(band)
    return () => {
      observer.disconnect()
      root.style.removeProperty('--page-sticky-height')
    }
  }, [ref])
}
