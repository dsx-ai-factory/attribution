// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { thirdPartyNotices } from '../../blueprint/src/attribution_blueprint/webui/frontend/third-party-notices'

// Slidev merges this config into its Vite build.
export default {
  plugins: [thirdPartyNotices({
    configUrl: import.meta.url,
    sourceNotices: new URL('../../THIRD_PARTY_NOTICES.md', import.meta.url),
    licenseDirectory: new URL('./third-party-licenses/', import.meta.url),
    licenseSupplements: {
      '@iconify-json/carbon@1.2.26': 'iconify-json-carbon.txt',
      '@iconify-json/ph@1.2.2': 'iconify-json-ph.txt',
      '@iconify-json/svg-spinners@1.2.4': 'iconify-json-svg-spinners.txt',
      'dompurify@3.4.16': 'babel-helpers.txt',
    },
  })],
}
