// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react'

/**
 * Titles the browser tab "{subject} · {page} · Attribution": the most specific
 * part first, so tabs stay distinguishable when the browser truncates them.
 * A page whose record is not shown (loading, missing, failed) passes no subject.
 */
export function useDocumentTitle(page: string, subject?: string) {
  const title = subject ? `${subject} · ${page} · Attribution` : `${page} · Attribution`
  useEffect(() => {
    document.title = title
  }, [title])
}
