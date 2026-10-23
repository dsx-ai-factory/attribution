// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const ENTITIES: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
}

/**
 * Escapes recorded text for an HTML string. Chart tooltips are built as HTML,
 * and symptom, node, rule and user names come from recorded data, so every
 * value interpolated into one passes through here.
 */
export function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (ch) => ENTITIES[ch])
}
