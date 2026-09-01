# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Hook system for NVIDIA Attribution

This introduces hook mechanisms for NVIDIA Attribution.

There are two: multihooks, which allow for multiple subscribers and which only modify arguments in-place, and singlehooks, which return a value and which allow only one subscriber.

## Multihooks:
The basic scheme is that code that wants to allow hooks does the following:

```
from attribution_lib.hooks import multihooks

def my_func():
    item0 = foo()
    item1 = bar()
    multihooks.SomeName(item0, item1) # item0 and item1 will be passed to each hook function associated with `SomeName`. They may be modified in place.
```

Then, other code can do the following:
```
from attribution_lib.hooks import multihooks

@multihooks.SomeName.hook(priority=5)
def myhook(item0, item1):
    item0["key"] = 2
    item1 += 3
```

`myhook` will be called anywhere `hooks.SomeName()` is called. They will run in order of priority (equal priorities will run in unspecified order with respect to each other.)

## Singlehooks:
The basic scheme is that code that wants to allow hooks does the following:

```
from attribution_lib.hooks import singlehooks

@singlehooks.SomeName.set_default
def SomeName_default(item0, item1):
    return 0 # dummy return value

def my_func():
    item0 = foo()
    item1 = bar()
    return singlehooks.SomeName(item0, item1) # item0 and item1 will be passed to each hook function associated with `SomeName`. They may be modified in place.
```

Then, other code can do the following:
```
from attribution_lib.hooks import singlehooks

@singlehooks.SomeName.hook()
def myhook(item0, item1):
    # could alter item0, item1
    return item0 + item1
```


If no default nor hooks are registered, it is an error when the hook is called (raises `SingleHookNoDefault`).

If the hook is installed more than once, it is an error, unless `@singlehooks.SomeName.hook(replace_ok=True)` is set, in which case the latter will supercede the former.

Some foreward-thinking thoughts:

1. This mechanism allows for any number of call sites and hooks. This is flexible but could be dangerous.
2. Hooks with side-effects are probably a bad idea, in most cases. Caveat emptor!
3. Right now, hooks are are picked up as part of GlobalConfig(), so it should happen the first time that is accessed. There are cases where this might fall over, so be watchful
"""

import inspect
from collections.abc import Callable
from typing import Any, Optional

from loguru import logger


class MultiHookSpec:
    """Multiple subscribers, in-place modification"""

    def __init__(self, name: str):
        self.name = name
        self.hook_type = "MultiHook"
        self._hooks: list[dict[str, Any]] = []

    def hook(self, priority: int = 0, name: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register multiple hooks with priorities"""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._hooks.append(
                {"function": func, "priority": priority, "name": name or getattr(func, "__name__", "<unknown>")}
            )
            self._hooks.sort(key=lambda x: x["priority"], reverse=True)
            func_name = getattr(func, "__name__", "<unknown>")
            logger.info(f"MultiHook registered: {func_name} for {self.name} (priority: {priority})")
            return func

        return decorator

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Execute all hooks (in-place modification)"""
        for hook_info in self._hooks:
            hook_info["function"](*args, **kwargs)

    def get_info(self) -> dict[str, Any]:
        return {"type": "MultiHook", "name": self.name, "subscriber_count": len(self._hooks)}


class SingleHookNoDefault(Exception):
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return f"SingleHookNoDefault: '{self.name}' - no default function set."


class SingleHookIllegalReplacement(Exception):
    def __init__(self, hook_name: str, current_hook_name: str, current_hook_source: str) -> None:
        self.hook_name = hook_name
        self.current_hook_name = current_hook_name
        self.current_hook_source = current_hook_source

    def __str__(self) -> str:
        return f"SingleHookIllegalReplacement: '{self.hook_name}' already has subscriber '{self.current_hook_name}' from `{self.current_hook_source}`"


class SingleHookSpec:
    """Single subscriber, return values, requires default"""

    def __init__(self, name: str):
        self.name = name
        self.hook_type = "SingleHook"
        self.default_func: Optional[Callable[..., Any]] = None
        self.current_hook: Optional[Callable[..., Any]] = None
        self._hook_source: Optional[dict[str, Any]] = None

    def set_default(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Set the default behavior - REQUIRED for SingleHook"""
        self.default_func = func
        func_name = getattr(func, "__name__", "<unknown>")
        logger.info(f"SingleHook default set: {func_name} for {self.name}")
        return func

    def hook(self, replace_ok: bool = False) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a single hook with replacement control"""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            # Check for existing hook and replacement policy
            if self.current_hook is not None:
                if not replace_ok:
                    raise SingleHookIllegalReplacement(
                        self.name, getattr(self.current_hook, "__name__", "<unknown>"), self._format_hook_source()
                    )
                else:
                    old_name = getattr(self.current_hook, "__name__", "<unknown>")
                    new_name = getattr(func, "__name__", "<unknown>")
                    logger.info(f"SingleHook replacing: {old_name} -> {new_name} for {self.name}")

            # Track where this hook was registered (for error messages only)
            frame = inspect.currentframe()
            if frame and frame.f_back and frame.f_back.f_back:
                caller_frame = frame.f_back.f_back
                self._hook_source = {
                    "filename": caller_frame.f_code.co_filename,
                    "function": caller_frame.f_code.co_name,
                    "line": caller_frame.f_lineno,
                }

            self.current_hook = func
            func_name = getattr(func, "__name__", "<unknown>")
            logger.info(f"SingleHook registered: {func_name} for {self.name}")
            return func

        return decorator

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the hook or default behavior and return result"""

        # Execute hook or default
        if self.current_hook:
            return self.current_hook(*args, **kwargs)
        else:
            if self.default_func is None:
                raise SingleHookNoDefault(self.name)
            return self.default_func(*args, **kwargs)

    def _format_hook_source(self) -> str:
        """Format hook source for error messages"""
        if self._hook_source:
            filename = self._hook_source["filename"].split("/")[-1]
            return f"{filename}:{self._hook_source['line']} in {self._hook_source['function']}()"
        return "unknown location"

    def has_hook(self) -> bool:
        """Check if a custom hook is installed"""
        return self.current_hook is not None

    def has_default(self) -> bool:
        """Check if a default function is set"""
        return self.default_func is not None

    def clear_hook(self) -> None:
        """Remove the custom hook, reverting to default behavior"""
        if self.current_hook:
            old_hook = getattr(self.current_hook, "__name__", "<unknown>")
            self.current_hook = None
            self._hook_source = None
            logger.info(f"SingleHook cleared: {old_hook} removed from {self.name} (reverted to default)")
        else:
            logger.info(f"SingleHook {self.name} already using default (no custom hook to clear)")

    def get_info(self) -> dict[str, Any]:
        return {
            "type": "SingleHook",
            "name": self.name,
            "has_default": self.has_default(),
            "has_custom_hook": self.has_hook(),
            "current_hook": getattr(self.current_hook, "__name__", "<unknown>") if self.current_hook else None,
            "default_func": getattr(self.default_func, "__name__", "<unknown>") if self.default_func else None,
        }


class MultiHookManager:
    """Manager for MultiHooks"""

    def __init__(self) -> None:
        self._specs: dict[str, MultiHookSpec] = {}

    def __getattr__(self, name: str) -> MultiHookSpec:
        """Auto-create MultiHook specs"""
        if name.startswith("_"):
            raise AttributeError(f"No attribute '{name}'")

        if name not in self._specs:
            self._specs[name] = MultiHookSpec(name)
            logger.info(f"Created MultiHook spec: {name}")

        return self._specs[name]


class SingleHookManager:
    """Manager for SingleHooks"""

    def __init__(self) -> None:
        self._specs: dict[str, SingleHookSpec] = {}

    def __getattr__(self, name: str) -> SingleHookSpec:
        """Auto-create SingleHook specs"""
        if name.startswith("_"):
            raise AttributeError(f"No attribute '{name}'")

        if name not in self._specs:
            self._specs[name] = SingleHookSpec(name)
            logger.info(f"Created SingleHook spec: {name}")

        return self._specs[name]


multihooks = MultiHookManager()
singlehooks = SingleHookManager()
