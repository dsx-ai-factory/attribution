# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import itertools as it
import re
from collections.abc import Iterable


class ParseError(RuntimeError):
    """Generic parser exception class."""

    pass


class ParseBase:
    """Basic skeleton for parsing."""

    def __init__(self, in_str: str, pos: int):
        """Parser contains a string and its current position in that string (rather than actually split up strings and risk string-interning hell."""
        self.str = in_str
        self.pos = pos

    def comp_char(self, char: str) -> bool:
        """Compare self.string at self.pos and return if it is char; return false if we've run off the end of str. Perhaps we should accept an "end" param instead of relying on string end."""
        if self.pos >= len(self.str):
            return False
        return self.str[self.pos] == char


class HostPhrase(ParseBase):
    # """Parses HostPhrase := ([A-Z,a-z,_,-,0-9]*\[ExpanderExp\])*"""

    __hostexp__ = re.compile(r"[\w-]+")

    def parse(self) -> tuple[Iterable[str], int]:
        """Tries to parse the above expression, and throws an exception if it fails.
        Returns result and new position if it suceeds."""
        prod = []
        while self.pos < len(self.str):
            if self.comp_char("["):
                bracket_pos = self.pos
                self.pos += 1
                if self.pos >= len(self.str):
                    raise InvalidSyntaxError("Unclosed bracket at end of input", bracket_pos)
                try:
                    exp1, self.pos = ExpanderExp(self.str, self.pos).parse()
                except HostlistError:
                    # Preserve the position from nested errors
                    raise
                if not self.comp_char("]"):
                    raise InvalidSyntaxError("Missing closing bracket", bracket_pos)
                self.pos += 1
                prod.append(exp1)
            else:
                m = self.__hostexp__.match(self.str, self.pos)
                if m is None:
                    break
                prod.append(it.repeat(m.group(0), 1))
                self.pos = m.end()
        if len(prod) == 0:
            raise InvalidSyntaxError("Empty hostname", self.pos)
        return ("".join(x) for x in it.product(*prod)), self.pos


class ExpanderExp(ParseBase):
    """Parses ExpanderExp := (ExpanderSub)+(,ExpanderSub)*"""

    def parse(self) -> tuple[Iterable[str], int]:
        """Tries to parse the above expression, and throws an exception if it fails.
        Returns result and new position if it suceeds."""
        subs = []
        sub, self.pos = ExpanderSub(self.str, self.pos).parse()
        subs.append(sub)
        while self.comp_char(","):
            comma_pos = self.pos
            self.pos += 1
            if self.pos >= len(self.str):
                raise InvalidSyntaxError("Unexpected end of input after comma", comma_pos)
            sub, self.pos = ExpanderSub(self.str, self.pos).parse()
            subs.append(sub)
        return it.chain(*subs), self.pos


def prefix_count(numstr: str) -> int:
    """Return the estimated prefix length to use; if just '0' or we don't start with 0, that's 0. Otherwise, length of string."""
    if numstr == "0" or numstr[0] != "0":
        return 0
    else:
        return len(numstr)


class HostlistError(ParseError):
    """Base exception class for hostlist parsing errors"""

    def __init__(self, message: str, position: int):
        self.position = position
        super().__init__(f"{message} (at position {position})")


class InvalidRangeError(HostlistError):
    """Raised when a range expression is invalid (e.g. [5-3] or [01-2])"""

    pass


class InvalidSyntaxError(HostlistError):
    """Raised when the hostlist syntax is invalid (e.g. unclosed brackets)"""

    pass


class DuplicateHostError(HostlistError):
    """Raised when duplicate hosts are found in the expansion"""

    pass


class ExpanderSub(ParseBase):
    """Parses ExpanderSub := <NON-NEGATIVE INT>|<NON-NEGATIVE INT>-<NON-NEGATIVE INT>"""

    __rangeexp__ = re.compile(r"(\d+)-(\d+)")
    __digitsexp__ = re.compile(r"(\d+)")

    def parse(self) -> tuple[Iterable[str], int]:
        start_pos = self.pos
        if (m := self.__rangeexp__.match(self.str, self.pos)) is not None:
            low_str, high_str = m.groups()
            if not (low_str.isdigit() and high_str.isdigit()):
                raise InvalidSyntaxError("Range bounds must be integers", start_pos)

            low, high = int(low_str), int(high_str)
            if low >= high:
                raise InvalidRangeError(f"Invalid range [{low_str}-{high_str}]: start must be less than end", start_pos)

            padding = prefix_count(low_str)
            return (str(x).zfill(padding) for x in range(low, high + 1)), m.end()

        elif (m := self.__digitsexp__.match(self.str, self.pos)) is not None:
            return it.repeat(m.group(1), 1), m.end()

        raise InvalidSyntaxError("Expected number or range", start_pos)


class HostExpr(ParseBase):
    """Parses HostExpr := HostPhrase[.HostPhrase]*"""

    def parse(self) -> tuple[Iterable[str], int]:
        """Tries to parse the above expression, and throws an exception if it fails.
        Returns expanded result and new position if it suceeds."""
        prod = []
        host, self.pos = HostPhrase(self.str, self.pos).parse()
        prod.append(host)
        while self.comp_char("."):
            dot_pos = self.pos
            prod.append(".")
            self.pos += 1
            if self.pos >= len(self.str):
                raise InvalidSyntaxError("Unexpected end of input after dot", dot_pos)
            host1, self.pos = HostPhrase(self.str, self.pos).parse()
            prod.append(host1)
        return ("".join(x) for x in it.product(*prod)), self.pos


class HostList(ParseBase):
    """Parses comma-delimited HostExpr, expands, and checks for duplicates"""

    def parse(self) -> tuple[list[str], int]:
        """Tries to parse comma-delimnted HostExpr; returns result and updated position."""
        prod = []
        hosts, self.pos = HostExpr(self.str, self.pos).parse()
        prod.append(hosts)
        while self.comp_char(","):
            comma_pos = self.pos
            self.pos += 1
            if self.pos >= len(self.str):
                raise InvalidSyntaxError("Unexpected end of input after comma", comma_pos)
            hosts, self.pos = HostExpr(self.str, self.pos).parse()
            prod.append(hosts)
        all_hosts = list(it.chain(*prod))

        # Check for duplicates
        unique_hosts = set(all_hosts)
        if len(all_hosts) != len(unique_hosts):
            duplicates = [h for h in unique_hosts if all_hosts.count(h) > 1]
            # For duplicates, we'll use position 0 since it's a post-processing error
            raise DuplicateHostError(f"Duplicate hosts found: {', '.join(duplicates)}", 0)

        return all_hosts, self.pos


def parse_hostlist(hostlist: str) -> list[str]:
    """
    Parse a hostlist expression and return expanded list of hosts.

    Args:
        hostlist: A string containing the hostlist expression (e.g. "node[1-3],gpu[01-02]")

    Returns:
        A list of expanded hostnames

    Raises:
        InvalidSyntaxError: When the hostlist syntax is invalid, includes position
        InvalidRangeError: When a range expression is invalid, includes position
        DuplicateHostError: When duplicate hosts are found
    """
    if not hostlist:
        raise InvalidSyntaxError("Empty hostlist expression", 0)

    result, pos = HostList(hostlist, 0).parse()
    if pos < len(hostlist):
        raise InvalidSyntaxError("Unexpected characters", pos)

    return result
