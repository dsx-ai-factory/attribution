# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from contextlib import ExitStack
from datetime import datetime, timedelta

from loguru import logger

from app.attributor import AttributorId, AttributorInput, AttributorWrapper
from app.exceptions import DuplicateAttributorError, UnknownAttributorIdError


class AttributorManager:
    """Own wrappers until resource cleanup succeeds, even after API retirement.

    Retire under manager -> wrapper locks: remove the active entry, retain it in
    pending cleanup, then close. Wrapper._closed rejects waiting requests while
    EventManager._closed records actual resource release. Pending keys are object
    identities because replacements and rejected duplicates can share an API ID.
    Failed closes stay owned and retry on later passes; one failure must not stop
    other objects being cleaned. Shutdown uses the same path for every object.
    """

    attributors: dict[AttributorId, AttributorWrapper]
    lock: threading.Lock
    expiration_period: timedelta

    def __init__(self, expiration_period_hours: int = 24):
        """Constructor for Attribution Attributor Manager.

        Args:
            expiration_period_hours: Number of hours after which inactive attributors will be removed
        """
        self.attributors = {}
        # Key by object identity: a replacement may reuse the same attributor ID.
        self._pending_cleanup: dict[int, AttributorWrapper] = {}
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
            new_attributor = AttributorWrapper(info)
            if new_attributor.id in self.attributors:
                logger.warning(f"Attempted to add duplicate attributor with id: {new_attributor.id}")
                raise DuplicateAttributorError.new(new_attributor.id).to_http_exception()
            # Own before acquiring resources. Failed setup, deletion and expiry
            # all retire the same wrapper through the same retry queue.
            key = id(new_attributor)
            self._pending_cleanup[key] = new_attributor
            with ExitStack() as cleanup:
                cleanup.callback(self._close_attributor, new_attributor, preserve_error=True)
                new_attributor.initialize(info)
                self.attributors[new_attributor.id] = new_attributor
                del self._pending_cleanup[key]
                cleanup.pop_all()  # The active entry now owns the initialized wrapper.
            logger.info(
                f"Added new attributor with id: {new_attributor.id}. Total attributors: {len(self.attributors)}"
            )
            return new_attributor

    def delete_attributor(self, id: AttributorId):
        """Remove attributor by id, unless we don't have it, in which case we raise UnknownAttributor."""
        with self.lock:
            try:
                with self.attributors[id].lock:
                    self._close_attributor(self.attributors.pop(id))
                logger.info(f"Deleted attributor with id: {id}. Remaining attributors: {len(self.attributors)}")
            except KeyError:
                logger.warning(f"Attempted to delete unknown attributor id: {id}")
                raise UnknownAttributorIdError.new(id).to_http_exception()

    def _close_attributor(self, attributor: AttributorWrapper, *, preserve_error: bool = False) -> None:
        """Caller holds manager lock and, for published wrappers, wrapper lock.

        Enqueue before close; dequeue only on success. Never rely on a destructor
        or a log message to retain responsibility for unfinished cleanup.
        During setup unwind, log close failures but preserve the setup exception.
        """
        key = id(attributor)
        self._pending_cleanup[key] = attributor
        try:
            attributor.close()
        except Exception as error:
            logger.warning(
                f"Failed to close attributor {attributor.id}; retained for cleanup retry: "
                f"{type(error).__name__}: {error}"
            )
            if not preserve_error:
                raise
        else:
            del self._pending_cleanup[key]

    def cleanup_expired_attributors(self):
        """Retry retired attributors and expire inactive ones, attempting every cleanup."""
        self._cleanup_attributors()

    def close(self) -> None:
        """Attempt cleanup of every active and retired attributor on shutdown."""
        self._cleanup_attributors(close_all=True)

    def _cleanup_attributors(self, *, close_all: bool = False) -> None:
        with self.lock:
            first_error = None
            now = datetime.now()
            candidates = list(self._pending_cleanup.values()) + [
                wrapper
                for wrapper in self.attributors.values()
                if close_all or now - wrapper.last_access_time > self.expiration_period
            ]
            for wrapper in candidates:
                with wrapper.lock:
                    if self.attributors.get(wrapper.id) is wrapper:
                        # A request may have refreshed it before we acquired its lock.
                        if not close_all and datetime.now() - wrapper.last_access_time <= self.expiration_period:
                            continue
                        self.attributors.pop(wrapper.id)
                    try:
                        self._close_attributor(wrapper)
                    except Exception as error:
                        if first_error is None:
                            first_error = error
            if first_error is not None:
                raise first_error
