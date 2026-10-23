// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import dayjs from 'dayjs'
import utc from 'dayjs/plugin/utc'

dayjs.extend(utc)

// The one way the app writes a time for people: the full 24-hour date and
// time in UTC. TimeHover shows it with a muted " UTC" mark; text the app
// writes for people (copied evidence, its Markdown download) spells the mark
// out. Machine formats (JSON, YAML, CSV, the log download) keep the recorded
// string. A recorded time without a zone is UTC, as the server reads it
// (parseUtc), never the reader's own zone.
export const STAMP_FORMAT = 'YYYY-MM-DD HH:mm:ss'

/** A recorded time as an instant: its own zone when it names one, UTC when it does not. */
export function parseUtc(date: string): dayjs.Dayjs {
  return dayjs.utc(date)
}

/** A recorded time as the app writes it in text, e.g. "2026-09-21 00:30:01 UTC"; "Unavailable", as the UI says, when it does not parse. */
export function utcStamp(date: string): string {
  const instant = parseUtc(date)
  return instant.isValid() ? `${instant.format(STAMP_FORMAT)} UTC` : 'Unavailable'
}
