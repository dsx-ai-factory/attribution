# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import threading
from datetime import datetime, timedelta

from loguru import logger

from app.attributor import AttributorId, AttributorInput, AttributorWrapper
from app.exceptions import DuplicateAttributorError, UnknownAttributorIdError


class AttributorManager:
    """A simple collection of attribution attributors to help the service track them."""

    attributors: dict[AttributorId, AttributorWrapper]
    lock: threading.Lock
    expiration_period: timedelta

    def __init__(self, expiration_period_hours: int = 24):
        """Constructor for Attribution Attributor Manager.

        Args:
            expiration_period_hours: Number of hours after which inactive attributors will be removed
        """
        self.attributors = {}
        self.lock = threading.Lock()
        self.expiration_period = timedelta(hours=expiration_period_hours)
        logger.info(f"AttributorManager initialized with expiration period: {expiration_period_hours} hours")

    def get_attributor(self, id: AttributorId) -> AttributorWrapper:
        """Retrieve attributor by id, or raise UnknownAttributorId."""
        with self.lock:
            try:
                attributor = self.attributors[id]
                logger.debug(f"Retrieved attributor with id: {id}")
                return attributor
            except KeyError:
                logger.warning(f"Attempted to retrieve unknown attributor id: {id}")
                raise UnknownAttributorIdError.new(id).to_http_exception()

    def add_attributor(self, info: AttributorInput) -> AttributorWrapper:
        """Add a new attributor, computing id from info. Raise DuplicateAttributor if id is already in manager."""
        with self.lock:
            new_attributor = AttributorWrapper.new(info)
            if new_attributor.id in self.attributors:
                logger.warning(f"Attempted to add duplicate attributor with id: {new_attributor.id}")
                raise DuplicateAttributorError.new(new_attributor.id).to_http_exception()
            self.attributors[new_attributor.id] = new_attributor
            logger.info(
                f"Added new attributor with id: {new_attributor.id}. Total attributors: {len(self.attributors)}"
            )
            return new_attributor

    def delete_attributor(self, id: AttributorId):
        """Remove attributor by id, unless we don't have it, in which case we raise UnknownAttributor."""
        with self.lock:
            try:
                with self.attributors[id].lock:
                    del self.attributors[id]
                logger.info(f"Deleted attributor with id: {id}. Remaining attributors: {len(self.attributors)}")
            except KeyError:
                logger.warning(f"Attempted to delete unknown attributor id: {id}")
                raise UnknownAttributorIdError.new(id).to_http_exception()

    def cleanup_expired_attributors(self):
        """Remove attributors that have been inactive for longer than the expiration period."""
        with self.lock:
            now = datetime.now()
            expired_ids = [
                id
                for id, attributor in self.attributors.items()
                if now - attributor.last_access_time > self.expiration_period
            ]

            if expired_ids:
                logger.info(f"Found {len(expired_ids)} expired attributors to clean up: {expired_ids}")
                for id in expired_ids:
                    with self.attributors[id].lock:
                        del self.attributors[id]
                logger.info(
                    f"Cleaned up {len(expired_ids)} expired attributors. Remaining attributors: {len(self.attributors)}"
                )
            else:
                logger.debug(
                    f"No expired attributors found during cleanup. Current attributors: {len(self.attributors)}"
                )
