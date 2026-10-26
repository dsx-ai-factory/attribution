<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Public symptom contract fixture

`public-symptom-contract.json` is a trimmed output of the public
`.ci/symptom_doc_check.py snapshot` producer at source commit
`d5c1d1dcd86e352ae20a88a2de1051d24c1bc022`.
The rule and catalog source is NVIDIA/attribution (Apache-2.0).

Only the `detected-bad-nic-state` and `irisc-not-responding` declarations
and public document metadata are retained; the seven runtime examples are
unchanged. There is no tenant guidance or private manifest data.

The public test compares these records with fresh producer output. Keep any
consumer's copy identical when deliberately updating the contract. The recorded
SHA is fixture provenance, not the current PR's identity or an attestation.
