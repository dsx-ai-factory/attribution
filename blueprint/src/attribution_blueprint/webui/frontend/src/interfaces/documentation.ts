// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary

export interface SymptomDocumentation {
  id: string
  title: string
  doc_url: string
}

export interface SymptomDocumentationCatalog {
  symptoms: Record<string, SymptomDocumentation>
}
