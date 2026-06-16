# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Embedded copy of nvidia-xid-analyzer v2.3.

Only the parts attribution-lib uses are embedded: the analyzer module and the
XID resolution data it reads. That data (catalog, NVL5 decode rules, and
parsing policy) ships as ``xid_catalog_data.json`` rather than an upstream
vendor catalog source. The upstream visualizer, ingest, and
BMC helper scripts are intentionally not included, and the code is not a
separate distribution; it ships as part of attribution-lib.
"""
