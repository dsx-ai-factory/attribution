<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Public symptom contract fixture

`public-symptom-contract.json` is synthetic document metadata for the
[dsx-ai-factory/attribution](https://github.com/dsx-ai-factory/attribution)
coverage contract. It does not depend on any bundled rule, contain tenant
guidance, or include private manifest data. Its repeated-character commit SHA
is a placeholder, not provenance or an attestation.

Schema version 1 retains an empty `dynamic` list for consumer compatibility.
Specific runtime mapping examples are tested within attribution-lib. Coverage
CI validates all documentable declarations produced by the loaded rules.
