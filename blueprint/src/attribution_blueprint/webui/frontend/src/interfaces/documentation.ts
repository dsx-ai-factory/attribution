// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface SymptomDocumentation {
  id: string
  title: string
  doc_url: string
}

export interface SymptomDocumentationCatalog {
  symptoms: Record<string, SymptomDocumentation>
}
