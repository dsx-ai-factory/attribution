// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef, useState } from 'react'
import { copyText } from '@/lib/clipboard'

// What a copy control's polite live region says about its last copy.
const ANNOUNCEMENTS = { copied: 'Copied', failed: 'Copy failed' } as const

/**
 * Copy state for a copy control: `copied` turns true only after a copy that
 * succeeded, so the control never confirms text that is not on the clipboard,
 * and `announcement` is what the control's polite live region says about the
 * last copy: "Copied", or "Copy failed" when neither clipboard path took the
 * text. Both turn back after `resetAfterMs`.
 */
export function useCopyToClipboard(resetAfterMs: number) {
  const [outcome, setOutcome] = useState<keyof typeof ANNOUNCEMENTS | null>(null)
  const timer = useRef<number | undefined>(undefined)
  const mounted = useRef(false)

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      window.clearTimeout(timer.current)
    }
  }, [])

  const copy = useCallback(
    async (text: string) => {
      const ok = await copyText(text)
      // The control may have unmounted while the copy was pending.
      if (mounted.current) {
        window.clearTimeout(timer.current)
        setOutcome(ok ? 'copied' : 'failed')
        timer.current = window.setTimeout(() => setOutcome(null), resetAfterMs)
      }
      return ok
    },
    [resetAfterMs],
  )

  return { copied: outcome === 'copied', announcement: outcome === null ? '' : ANNOUNCEMENTS[outcome], copy }
}
