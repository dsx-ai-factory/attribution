// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Copies text and resolves whether it reached the clipboard. The async
 * Clipboard API exists only in secure contexts, so an instance served over
 * plain HTTP falls back to the legacy execCommand path; callers confirm a copy
 * only on true.
 */
export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    // Selecting the temporary field moves focus into it, and removing it would
    // leave focus on the page; it goes back to the control that asked to copy.
    const focused = document.activeElement
    const area = document.createElement('textarea')
    area.value = text
    area.style.position = 'fixed'
    area.style.opacity = '0'
    document.body.appendChild(area)
    area.select()
    try {
      return document.execCommand('copy')
    } catch {
      return false
    } finally {
      area.remove()
      if (focused instanceof HTMLElement) focused.focus({ preventScroll: true })
    }
  }
}
