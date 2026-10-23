#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""XID parsing and resolution helpers."""

import json
import logging
import pathlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Union
from zoneinfo import ZoneInfo

import pandas as pd

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

XID_CATALOG_DATA = SCRIPT_DIR / "xid_catalog_data.json"

_XID_DATA_CACHE = None

XID_CATALOG = None
NVL5_XID_DECODING_RULES = None
STICKY_XIDS = None
XID_POLICY = None
XID_WATCHLIST = []


def _restore_missing(value):
    """Keep JSON null values unset when rebuilding catalog records."""

    return value


def load_xid_catalog_data():
    """Load and cache XID catalog data."""
    global _XID_DATA_CACHE
    if _XID_DATA_CACHE is None:
        with open(XID_CATALOG_DATA, encoding="utf-8") as data_file:
            _XID_DATA_CACHE = json.load(data_file)
    return _XID_DATA_CACHE


def normalized_string(s):
    if s is None or pd.isna(s) or not isinstance(s, str):
        return str(s)

    return re.sub(r"\s+", "", s)


def get_resolution_for_xid154(input_string):
    """Extract the dynamic recovery action from an XID 154 message.

    Args:
        input_string: Driver message containing the recovery action transition.

    Returns:
        The action in the final parenthesized group, or None when unavailable.
    """

    match = re.search(r"\bto\s+0x[0-9a-fA-F]+\s+\((?P<action>[^)]*)\)", input_string)
    if match is None:
        return None
    return match.group("action")


def extract_xid94_poison_victim(message: str) -> str | None:
    """Extract the normalized poison victim from an XID 94 message.

    Args:
        message: Driver message that may contain XID 94 containment details.

    Returns:
        ``CE``, ``SM``, or ``GSP`` when declared by the message; otherwise
        ``None``.
    """

    is_nvrm_xid94 = re.search(r"NVRM: Xid \([^)]*\):\s*94,", message) is not None
    is_redfish_xid94 = (
        re.search(r"Driver Event Message has detected errors of type\s+'[^']*\bXID\s+94\b", message) is not None
    )
    if not is_nvrm_xid94 and not is_redfish_xid94:
        return None

    victim_match = re.search(r"\bContained:\s*(?P<victim>[^.(]+?)\s*(?:\([^)]*\))?\.\s*RST:", message)
    if victim_match is None:
        return None

    normalized = " ".join(victim_match.group("victim").split()).casefold()
    return {
        "ce": "CE",
        "ce user channel": "CE",
        "sm": "SM",
        "gsp": "GSP",
    }.get(normalized)


def ContentOfColumn(header, row, col_name):
    """Find and return the content of a column by matching normalized column names."""
    for index, name in enumerate(header):
        if normalized_string(col_name) in normalized_string(name):
            value = row[index]

            if isinstance(value, (float, int)):
                return str(value)
            return value
    return ""


def policy_xidlist_2_set(xidlist):
    xidset = set()
    global XID_WATCHLIST

    if not xidlist or pd.isna(xidlist):
        return xidset

    if isinstance(xidlist, (float, int)):
        xidlist = str(xidlist)

    xidlist = xidlist.replace("\n", "").replace("\r", "")

    for xid in re.split(r",\s*(?=\d+\{|\d+\s*,|\d+$|\d+\s*$)", xidlist):
        logging.debug(f"\nProcessing Rule Fragment xid: {xid}")

        if not xid or xid == "":
            continue

        if pd.isna(xid) or xid.lower() == "nan":
            continue

        if xid == "*":
            xidset.add("*")
            continue

        xid = xid.strip()

        xid_ctx = ""
        xid_drv = ""
        xid_name = xid
        xid_subcode = ""
        xid_intrinfo = ""
        xid_errorstatus = ""
        xid_num = 0

        entry_pattern = re.compile(r"(?P<number>\d+|\*)(?:\{(?P<properties>.*?)\})?")
        match = entry_pattern.match(xid)
        if match:
            xid_num = int(match.group("number")) if match.group("number") != "*" else 0
            if match.group("properties") is not None:
                try:
                    properties = "{" + match.group("properties") + "}"
                    properties_dict = json.loads(properties)

                    if "context" in properties_dict:
                        xid_ctx = properties_dict["context"]
                    if "driver" in properties_dict:
                        xid_drv = properties_dict["driver"]
                    if "errorstatus" in properties_dict:
                        xid_errorstatus = properties_dict["errorstatus"]
                    if "subcode" in properties_dict:
                        xid_subcode = properties_dict["subcode"]
                    if "intrinfo" in properties_dict:
                        xid_intrinfo = properties_dict["intrinfo"]
                    if "rate" in properties_dict:
                        del properties_dict["rate"]
                        properties = json.dumps(properties_dict)
                    xid_name = str(xid_num) + properties
                except json.JSONDecodeError as e:
                    logging.warning(f"JSON parsing error in entry '{xid}': {e}")

                    xid_name = str(xid_num)
        else:
            logging.warning(f"Failed to parse entry {xid}")

            continue

        xidset.add(xid_name)

        logging.debug(
            f"\nAdding XIDEntry to XID_WATCHLIST: name={xid_name}, number={xid_num}, context={xid_ctx}, driver={xid_drv}, subcode={xid_subcode}, intrinfo={xid_intrinfo}, errorstatus={xid_errorstatus}\n"
        )
        XID_WATCHLIST.append(
            XIDEntry(
                name=xid_name,
                number=xid_num,
                context=xid_ctx,
                driver=xid_drv,
                subcode=xid_subcode,
                intrinfo=xid_intrinfo,
                errorstatus=xid_errorstatus,
            )
        )

    return xidset


@dataclass
class XIDEntry:
    """Parsed XID error entry."""

    name: str
    number: int
    context: str
    driver: str
    timestamp: Optional[Union[datetime, str]] = None
    machine: str = "na"
    source: Optional[str] = None

    pcie_bdf: Optional[str] = None
    subcode: Optional[str] = None
    severity: Optional[str] = None
    cross_contain: Optional[str] = None
    injected: Optional[str] = None
    link: Optional[str] = None
    intrinfo: Optional[str] = None
    errorstatus: Optional[str] = None
    nvl5_port_down_reason: Optional[int] = None
    nvl5_error_code: Optional[int] = None
    vbios_version: Optional[str] = None
    mnemonic: Optional[str] = None
    poison_victim: Optional[str] = None

    resolution: Optional[str] = None
    investigatory_action: Optional[str] = None
    comments: Optional[str] = None
    decoded_xid_string: Optional[str] = None

    job_failure_severity: Optional[str] = None
    hw_sw_classification: Optional[str] = None

    def parse_timestamp(self):
        """Parse timestamp string into a datetime object with proper timezone."""
        if self.timestamp is None or self.timestamp == "":
            self.timestamp = ensure_timezone_local(datetime.fromtimestamp(0))
            return

        if isinstance(self.timestamp, datetime):
            if self.timestamp.tzinfo is None:
                self.timestamp = ensure_timezone_local(self.timestamp)
            return

        timestamp_str = str(self.timestamp).replace("0000000", "000000")

        if re.search(r"[+-]\d{2}$", timestamp_str):
            timestamp_str += "00"

        formats = [
            "%Y-%m-%d %H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S%z",
            "%a %b %d %H:%M:%S %Y",
            "%a %b  %d %H:%M:%S %Y",
            "%a %b %d %H:%M:%S %Z %Y",
        ]

        parsed = False
        for fmt in formats:
            try:
                self.timestamp = datetime.strptime(timestamp_str, fmt)
                parsed = True
                break
            except ValueError:
                continue

        uptime_match = re.fullmatch(r"\d+(?:\.\d+)?", timestamp_str)
        if parsed is False and uptime_match is not None:
            try:
                timestamp_seconds = float(timestamp_str) if "." in timestamp_str else float(timestamp_str) / 1000000
                self.timestamp = datetime.fromtimestamp(timestamp_seconds)
                parsed = True
            except (ValueError, OverflowError):
                pass

        if not parsed:
            try:
                from dateutil import parser as dateutil_parser

                self.timestamp = dateutil_parser.parse(timestamp_str)
                parsed = True
            except (ImportError, ValueError):
                pass

            if not parsed and re.search(r"[A-Za-z]{3}\s+\s+\d{1}\s", timestamp_str):
                try:
                    fixed_ts = re.sub(r"([A-Za-z]{3})\s+\s+(\d{1}\s)", r"\1 \2", timestamp_str)
                    self.timestamp = datetime.strptime(fixed_ts, "%a %b %d %H:%M:%S %Y")
                    parsed = True
                except ValueError:
                    pass

        if parsed is False and timestamp_str != "":
            logging.debug(f"Date string '{timestamp_str}' could not be parsed. Using the Unix epoch instead.")
            self.timestamp = datetime.fromtimestamp(0)

        if self.timestamp is not None and self.timestamp.tzinfo is None:
            self.timestamp = ensure_timezone_local(self.timestamp)

    @classmethod
    def from_message(cls, message=None, timestamp=None, driver="unknown", machine="na", vbios_version=None):
        """Extract XID information from a message and create an XIDEntry object."""
        if not message:
            return None

        poison_victim = extract_xid94_poison_victim(message)

        if not timestamp and message:
            if "NVRM: Xid" in message:
                full_ts_match = re.search(
                    r"^\[\s*([A-Za-z]{3}\s+[A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4})\s*\]", message
                )
                if full_ts_match:
                    timestamp = full_ts_match.group(1).strip()

                if not timestamp:
                    timestamp_match = re.search(r"^\[\s*([0-9.]+)\s*\]", message)
                    if timestamp_match:
                        timestamp = timestamp_match.group(1)

                if not timestamp:
                    syslog_match = re.search(r"^([A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})", message)
                    if syslog_match:
                        timestamp = syslog_match.group(1)

            elif "Driver Event Message has detected errors of type" in message:
                redfish_match = re.search(r"\[\s*([^\]]+)\s*\](?:\[[^\]]*\]){0,2} XID", message)
                if redfish_match:
                    timestamp = redfish_match.group(1).strip()

        pattern_nvl5_redfish = re.compile(
            r"Driver Event Message has detected errors of type \'\[(?P<timestamp>[^\]]+)\](?:\[[^\]]*\]){0,2} XID (?P<xid_number>\d+)\s+(?P<subcode>\w+)\s+(?P<severity>\w+)\s+(?P<cross_contain>\w+)\s+(?P<injected>\w+)\s+Link\s+(?P<link>-?\d+)\s+\((?P<intrinfo>0x[0-9a-fA-F]+)\s+(?P<errorstatus>0x[0-9a-fA-F]+)"
        )
        match = pattern_nvl5_redfish.search(message)
        if match:
            logging.debug("Matched pattern_nvl5_redfish")
            xid_number = int(match.group("xid_number"))
            xid_entry = cls(
                name="",
                number=xid_number,
                context=message,
                driver=driver,
                timestamp=timestamp if timestamp else match.group("timestamp"),
                machine=machine,
                subcode=match.group("subcode"),
                severity=match.group("severity"),
                cross_contain=match.group("cross_contain"),
                injected=match.group("injected"),
                link=match.group("link"),
                intrinfo=int(match.group("intrinfo"), 16) if match.group("intrinfo") else None,
                errorstatus=int(match.group("errorstatus"), 16) if match.group("errorstatus") else None,
                vbios_version=vbios_version,
                poison_victim=poison_victim,
            )
            xid_entry.parse_timestamp()
            return xid_entry

        pattern_nvl5_dmesg = re.compile(
            r"NVRM: Xid \(PCI:(?P<pcie_bdf>[^)]+)\): (?P<xid_number>\d+)(?:, pid=\d+, name=(?:[^,]*))?, (?P<subcode>\w+)\s+(?P<severity>\w+)\s+(?P<cross_contain>\w+)\s+(?P<injected>\w+)\s+Link\s+(?P<link>-?\d+)\s+\((?P<intrinfo>0x[0-9a-fA-F]+)\s+(?P<errorstatus>0x[0-9a-fA-F]+)"
        )
        match = pattern_nvl5_dmesg.search(message)
        if match:
            logging.debug("Matched pattern_nvl5_dmesg")
            xid_number = int(match.group("xid_number"))
            xid_entry = cls(
                name="",
                number=xid_number,
                context=message,
                driver=driver,
                timestamp=timestamp if timestamp else None,
                machine=machine,
                pcie_bdf=match.group("pcie_bdf"),
                subcode=match.group("subcode"),
                severity=match.group("severity"),
                cross_contain=match.group("cross_contain"),
                injected=match.group("injected"),
                link=match.group("link"),
                intrinfo=int(match.group("intrinfo"), 16) if match.group("intrinfo") else None,
                errorstatus=int(match.group("errorstatus"), 16) if match.group("errorstatus") else None,
                vbios_version=vbios_version,
                poison_victim=poison_victim,
            )
            xid_entry.parse_timestamp()
            return xid_entry

        pattern_redfish = re.compile(
            r"Driver Event Message has detected errors of type \'\[(?P<timestamp>[^\]]+)\](?:\[[^\]]*\]){0,2} XID (?P<xid_number>\d+)(?:\s+[^\']*)?"
        )
        match = pattern_redfish.search(message)
        if match:
            logging.debug("Matched pattern_redfish")
            xid_number = int(match.group("xid_number"))
            xid_entry = cls(
                name="",
                number=xid_number,
                context=message,
                driver=driver,
                timestamp=timestamp if timestamp else match.group("timestamp"),
                machine=machine,
                poison_victim=poison_victim,
            )
            xid_entry.parse_timestamp()
            return xid_entry

        pattern_dmesg = re.compile(
            r"NVRM: Xid \(PCI:(?P<pcie_bdf>[^)]+)\): (?P<xid_number>\d+), pid=(?:\'[^\']*\'|[^,]*), name=[^,]*, (.*)"
        )
        match = pattern_dmesg.search(message)
        if match:
            logging.debug("Matched pattern_dmesg")
            xid_number = int(match.group("xid_number"))
            xid_entry = cls(
                name="",
                number=xid_number,
                context=message,
                driver=driver,
                timestamp=timestamp if timestamp else None,
                machine=machine,
                pcie_bdf=match.group("pcie_bdf"),
                poison_victim=poison_victim,
            )
            xid_entry.parse_timestamp()
            return xid_entry

        pattern_dmesg_simple = re.compile(r"NVRM: Xid \(PCI:(?P<pcie_bdf>[^)]+)\): (?P<xid_number>\d+)")
        match = pattern_dmesg_simple.search(message)
        if match:
            logging.debug("Matched pattern_dmesg_simple")
            xid_number = int(match.group("xid_number"))
            xid_entry = cls(
                name="",
                number=xid_number,
                context=message,
                driver=driver,
                timestamp=timestamp if timestamp else None,
                machine=machine,
                pcie_bdf=match.group("pcie_bdf"),
                vbios_version=vbios_version,
                poison_victim=poison_victim,
            )
            xid_entry.parse_timestamp()
            return xid_entry

        return None

    def lookup_resolution(self):
        """Lookup resolution information for this XID entry."""
        global XID_CATALOG, NVL5_XID_DECODING_RULES

        try:
            logging.debug(f"Looking up resolution for XID: {self.number}")

            if XID_CATALOG and str(self.number) in XID_CATALOG:
                logging.debug("1st level lookup of xid in XID_CATALOG")
                catalog_entry = XID_CATALOG[str(self.number)]
                self.resolution = catalog_entry.get("recovery_action", "Unknown")

                self.mnemonic = catalog_entry.get("mnemonic", "")

                self.decoded_xid_string = str(self.number)

                if catalog_entry.get("investigatory_action"):
                    self.investigatory_action = catalog_entry.get("investigatory_action")

                if catalog_entry.get("decode_workflow") == "XID_154":
                    extracted_action = get_resolution_for_xid154(self.context)
                    if extracted_action is None:
                        self.resolution = "No resolution found for XID_154"
                    else:
                        self.resolution = extracted_action
                elif catalog_entry.get("decode_workflow") == "WORKFLOW_NVLINK5_ERR" and NVL5_XID_DECODING_RULES:
                    logging.debug(f"2nd level lookup of xid {self.number} in NVL5_XID_DECODING_RULES")

                    if hasattr(self, "subcode") and hasattr(self, "errorstatus") and hasattr(self, "intrinfo"):
                        logging.debug("XID entry has required NVL5 attributes")

                        self._extract_port_down_reason_and_error_code_from_intrinfo()

                        if int(self.number) in NVL5_XID_DECODING_RULES:
                            logging.debug(f"Found XID {self.number} in NVL5 rules")

                            for nvl5_decoding_rule in NVL5_XID_DECODING_RULES[int(self.number)]:
                                logging.debug(f"Checking NVL5 rule: {nvl5_decoding_rule}")
                                try:
                                    if (
                                        self.subcode in nvl5_decoding_rule["sheet_subcode"]
                                        and self.errorstatus in nvl5_decoding_rule["sheet_error_status_hex"]
                                        and does_xid_intrinfo_match_rule(
                                            nvl5_decoding_rule["sheet_intrinfo_binary"], self.intrinfo
                                        )
                                    ):
                                        logging.debug(f"Found matching NVL5 rule: {nvl5_decoding_rule}")
                                        self.resolution = nvl5_decoding_rule["recovery_action"]

                                        if nvl5_decoding_rule.get("investigatory_action"):
                                            self.investigatory_action = nvl5_decoding_rule["investigatory_action"]

                                        if nvl5_decoding_rule["sheet_comments"]:
                                            self.comments = nvl5_decoding_rule["sheet_comments"]

                                        self._determine_job_failure_severity(nvl5_decoding_rule)

                                        self._determine_hw_sw_classification(nvl5_decoding_rule)

                                        break

                                    else:
                                        pattern = r"Ch 00000[0-9]{3}$"
                                        if re.search(pattern, self.context) and self.number == 145:
                                            self.resolution = "IGNORE"
                                            self.investigatory_action = "IGNORE"
                                            self.comments = "Channel retirement message; no direct action needed"
                                            self.job_failure_severity = "Job-Non-Fatal"
                                            break

                                except Exception as e:
                                    logging.debug(f"Error matching NVL5 rule: {e}")
                                    continue
                        else:
                            logging.debug(f"XID {self.number} not found in NVL5 rules")
                    else:
                        logging.debug("XID entry missing required NVL5 attributes")
            else:
                logging.debug(f"XID {self.number} not found in catalog")

            if 144 <= self.number <= 150 and self.job_failure_severity is None:
                self.job_failure_severity = "Job-Non-Fatal"

            return self

        except Exception as e:
            logging.error(f"Error looking up resolution for XID: {e}")
            return self

    def _determine_job_failure_severity(self, nvl5_decoding_rule):
        """Determine Job Failure Severity based on NVL5 XID decoding rule."""
        global XID_CATALOG

        if not (144 <= self.number <= 150):
            return

        severity = nvl5_decoding_rule.get("sheet_severity", "").lower()
        local_remote = nvl5_decoding_rule.get("sheet_local_remote", "").lower()

        fatal_exclusions = ["fatal**", "non-fatal", "link fatal", "link fatal?"]
        if "fatal" in severity and not any(exclusion in severity for exclusion in fatal_exclusions):
            self.job_failure_severity = "Job-Fatal"
            logging.debug("Setting Job-Fatal due to 'fatal' in Severity column")
            return

        if "fatal**" in severity:
            self.job_failure_severity = "Possibly-Job-Fatal"
            logging.debug(
                "Setting Possibly-Job-Fatal due to 'fatal**' in Severity column. May or may not be Job-Fatal."
            )
            return

        non_fatal_terms = ["non-fatal", "link fatal", "link fatal?"]
        if any(term in severity for term in non_fatal_terms):
            if "appcrash" in local_remote:
                self.job_failure_severity = "Job-Fatal"
                logging.debug("Setting Job-Fatal due to 'AppCrash' in Local/Remote column")
                return

            if local_remote and XID_CATALOG:
                xid_numbers = re.findall(r"\b\d+\b", local_remote)

                if xid_numbers:
                    logging.debug(f"Found potential XID references: {xid_numbers}")
                    for xid_num in xid_numbers:
                        if xid_num in XID_CATALOG:
                            catalog_entry = XID_CATALOG[xid_num]
                            job_fatal_value = str(catalog_entry.get("job_fatal") or "").lower()

                            if job_fatal_value == "yes":
                                self.job_failure_severity = "Job-Fatal"
                                logging.debug(f"Setting Job-Fatal due to referenced XID {xid_num}")
                                return

        if "non-fatal*" in severity and self.context and "fatal" in self.context.lower():
            self.job_failure_severity = "Job-Fatal"
            logging.debug("Setting Job-Fatal due to 'fatal' in context with 'non-fatal*'")
            return

        self.job_failure_severity = "Job-Non-Fatal"
        logging.debug("Default case: Setting Job-Non-Fatal")

    def _determine_hw_sw_classification(self, nvl5_decoding_rule):
        """Determine HW/SW Classification based on NVL5 XID decoding rule."""

        if not (144 <= self.number <= 150):
            return

        hw_sw_classification = nvl5_decoding_rule.get("sheet_hw_sw_classification", "")

        if hw_sw_classification:
            self.hw_sw_classification = hw_sw_classification
            logging.debug(f"Setting HW/SW Classification to '{hw_sw_classification}' for XID {self.number}")
        else:
            self.hw_sw_classification = ""
            logging.debug(f"No HW/SW Classification found for XID {self.number}, setting to empty string")

    def _extract_port_down_reason_and_error_code_from_intrinfo(self):
        """Extract port down reason and error code from intrinfo."""
        if self.intrinfo is not None:
            self.nvl5_port_down_reason = (self.intrinfo >> 20) & 0x3F

            self.nvl5_error_code = self.intrinfo & 0x7F

            self.decoded_xid_string = f"{self.number}.{self.nvl5_port_down_reason:X}h"
        else:
            logging.debug(f"No intrinfo found for XID Message: {self.context}")
            self.nvl5_port_down_reason = 0
            self.nvl5_error_code = 0


@dataclass
class XIDBurst:
    """Class to represent a burst (group of XIDs within 3 minutes of each other)."""

    xids: set[XIDEntry]
    xid_list: list[XIDEntry]
    decoded_xid_set: set[str]
    count: int
    sources: list[str]
    is_interruption: bool
    category: str
    sub_category: str
    start_time: datetime
    end_time: datetime
    vbios_version: str
    driver_version: str

    @classmethod
    def from_xid_list(cls, xid_list):
        """Create a XIDBurst object from a list of XIDEntry objects."""
        if not xid_list:
            return None
        first_xid = xid_list[0]
        last_xid = xid_list[-1]
        return cls(
            xids=set(xid.name for xid in xid_list),
            xid_list=xid_list,
            decoded_xid_set=set(
                xid.decoded_xid_string
                for xid in xid_list
                if hasattr(xid, "decoded_xid_string") and xid.decoded_xid_string
            ),
            count=len(xid_list),
            sources=list(xid.source for xid in xid_list if xid.source is not None),
            is_interruption=False,
            category="",
            sub_category="",
            start_time=first_xid.timestamp,
            end_time=last_xid.timestamp,
            vbios_version=first_xid.vbios_version if first_xid.vbios_version else "",
            driver_version=first_xid.driver if first_xid.driver else "",
        )


@dataclass
class NodeEntry:
    """Class to read in and hold data for each node."""

    name: str
    interruptions_count: int = 0
    xid_entries: list[XIDEntry] = field(default_factory=list)
    xid_bursts: list[XIDBurst] = field(default_factory=list)
    sw_upgrades_driver: list[tuple[datetime, str]] = field(default_factory=list)
    fw_upgrades_vbios: list[tuple[datetime, str]] = field(default_factory=list)

    def update_version_history(self, timestamp, vbios_version=None, driver_version=None):
        """Update version history when new information is detected."""
        if vbios_version and timestamp:
            self.fw_upgrades_vbios.append((timestamp, vbios_version))

            self.fw_upgrades_vbios.sort(key=lambda x: x[0])

            i = 0
            while i < len(self.fw_upgrades_vbios) - 1:
                if self.fw_upgrades_vbios[i][1] == self.fw_upgrades_vbios[i + 1][1]:
                    del self.fw_upgrades_vbios[i + 1]
                else:
                    i += 1

        if driver_version and timestamp:
            self.sw_upgrades_driver.append((timestamp, driver_version))

            self.sw_upgrades_driver.sort(key=lambda x: x[0])

            i = 0
            while i < len(self.sw_upgrades_driver) - 1:
                if self.sw_upgrades_driver[i][1] == self.sw_upgrades_driver[i + 1][1]:
                    del self.sw_upgrades_driver[i + 1]
                else:
                    i += 1


@dataclass
class FleetData:
    """Fleet-level XID analysis state."""

    nodes: list[NodeEntry] = field(default_factory=list)

    driver_versions: set[str] = field(default_factory=set)
    vbios_versions: set[str] = field(default_factory=set)
    node_names: set[str] = field(default_factory=set)
    interruption_categories: set[str] = field(default_factory=set)
    interruption_sub_categories: set[str] = field(default_factory=set)

    total_interruptions_count: int = 0

    _unique_xid_entries: set = field(default_factory=set)

    def _get_or_create_node(self, node_name):
        """Get an existing node or create a new one if it doesn't exist."""
        for node in self.nodes:
            if node.name == node_name:
                return node

        new_node = NodeEntry(name=node_name)
        self.nodes.append(new_node)
        return new_node

    def _add_xid_entry(self, xid_entry, drop_duplicates):
        """Add a XIDEntry to the fleet data, avoiding duplicates."""
        if not xid_entry:
            return

        if drop_duplicates:
            key = (xid_entry.machine, xid_entry.timestamp, xid_entry.number, xid_entry.context)
            if key in self._unique_xid_entries:
                return
            self._unique_xid_entries.add(key)

        node = self._get_or_create_node(xid_entry.machine)
        node.xid_entries.append(xid_entry)

        if not xid_entry.vbios_version and xid_entry.timestamp:
            xid_entry.vbios_version = self._lookup_version_for_timestamp(node, xid_entry.timestamp, "vbios")

        if (
            not xid_entry.driver
            or xid_entry.driver == "unknown"
            or (xid_entry.driver and xid_entry.driver.strip() == "")
        ) and xid_entry.timestamp:
            xid_entry.driver = self._lookup_version_for_timestamp(node, xid_entry.timestamp, "driver")

        node.update_version_history(xid_entry.timestamp, xid_entry.vbios_version, xid_entry.driver)

        if xid_entry.driver:
            self.driver_versions.add(xid_entry.driver)
        if xid_entry.vbios_version:
            self.vbios_versions.add(xid_entry.vbios_version)
        if xid_entry.machine:
            self.node_names.add(xid_entry.machine)

    def _lookup_version_for_timestamp(self, node, timestamp, version_type):
        """Look up the version (VBIOS or driver) that was active at a specific timestamp for a node."""
        if not timestamp:
            return None

        if version_type == "vbios":
            version_history = node.fw_upgrades_vbios
        elif version_type == "driver":
            version_history = node.sw_upgrades_driver
        else:
            raise ValueError(f"Invalid version_type: {version_type}. Must be 'vbios' or 'driver'")

        if not version_history:
            return None

        active_version = None
        for version_timestamp, version in version_history:
            if version_timestamp <= timestamp:
                active_version = version
            else:
                break

        return active_version

    def tag_xids(self):
        """Tag XIDs based on the watchlist (step 7 in expected flow)."""
        global XID_WATCHLIST

        for node in self.nodes:
            for xid_entry in node.xid_entries:
                if xid_entry.name != "":
                    continue

                for xid_wanted in XID_WATCHLIST:
                    if xid_wanted.context and xid_wanted.context not in xid_entry.context:
                        continue
                    if xid_wanted.driver and xid_entry.driver and xid_entry.driver not in xid_wanted.driver:
                        continue
                    if xid_wanted.subcode and xid_entry.subcode and xid_entry.subcode not in xid_wanted.subcode:
                        continue
                    if (
                        xid_wanted.errorstatus
                        and xid_entry.errorstatus
                        and str(xid_entry.errorstatus) not in xid_wanted.errorstatus
                    ):
                        continue
                    if xid_wanted.intrinfo and xid_entry.intrinfo:
                        match_found = False
                        for pattern in xid_wanted.intrinfo.split(";"):
                            if does_xid_intrinfo_match_rule(pattern.strip(), xid_entry.intrinfo):
                                match_found = True
                                break
                        if not match_found:
                            continue
                    if xid_entry.number == xid_wanted.number:
                        xid_entry.name = xid_wanted.name
                        break

                if xid_entry.name == "":
                    xid_entry.name = str(xid_entry.number)

    def find_resolutions(self):
        """Find resolutions for XIDs (step 6 in expected flow)."""

        for node in self.nodes:
            for xid_entry in node.xid_entries:
                xid_entry.lookup_resolution()

    def group_xids_into_bursts(self):
        """Group XIDs into bursts (step 8 in expected flow)."""

        for node in self.nodes:
            node_bursts = SplitListToBursts(node.xid_entries)

            node.xid_bursts = node_bursts

    def match_bursts_to_patterns(self):
        """Match bursts to patterns (step 9 in expected flow)."""
        global XID_POLICY
        self.total_interruptions_count = 0

        for node in self.nodes:
            for burst in node.xid_bursts:
                if burst is None:
                    continue

                for pattern in XID_POLICY.XIDPolicyEntries:
                    if pattern.Matches(burst):
                        burst.category = pattern.CATEGORY
                        burst.sub_category = pattern.SUBCATEGORY

                        self.interruption_categories.add(pattern.CATEGORY)
                        self.interruption_sub_categories.add(pattern.SUBCATEGORY)
                        if "Yes" in pattern.JOBDISRUPTION:
                            self.total_interruptions_count += 1
                            node.interruptions_count += 1

                            burst.is_interruption = True
                        break


@dataclass
class XIDBurstPattern:
    RAWROW: list = None
    PRIORITY: str = "0"
    CATEGORY: str = ""
    SUBCATEGORY: str = ""
    JOBDISRUPTION: str = ""
    NEEDSOUTCAST: str = ""
    REPEATERATE: str = ""
    XIDS_MUST_EQUAL: str = ""
    XIDS_MUST_INTERSECT: str = ""
    XIDS_MUST_NOT_INTERSECT: str = ""
    XIDS_MAY_INTERSECT: str = ""

    SET_XIDS_MUST_EQUAL: set = field(default_factory=set)
    SET_XIDS_MUST_INTERSECT: set = field(default_factory=set)
    SET_XIDS_MUST_NOT_INTERSECT: set = field(default_factory=set)
    SET_XIDS_MAY_INTERSECT: set = field(default_factory=set)

    def __post_init__(self):
        if self.RAWROW is None:
            return

        self.PRIORITY = ContentOfColumn(xid_policy_header, self.RAWROW, "Priority\nOrder")
        self.CATEGORY = ContentOfColumn(xid_policy_header, self.RAWROW, "Category")
        self.SUBCATEGORY = ContentOfColumn(xid_policy_header, self.RAWROW, "Sub-Category")
        self.JOBDISRUPTION = ContentOfColumn(xid_policy_header, self.RAWROW, "Job\nDisruption")
        self.NEEDSOUTCAST = ContentOfColumn(xid_policy_header, self.RAWROW, "Needs\nOutcast")
        self.REPEATERATE = ContentOfColumn(xid_policy_header, self.RAWROW, "Occurrence\n(Count, Period, VMs impacted)")
        self.XIDS_MUST_EQUAL = ContentOfColumn(xid_policy_header, self.RAWROW, "XIDs\nPresent")
        self.XIDS_MUST_INTERSECT = ContentOfColumn(xid_policy_header, self.RAWROW, "XIDs\nPartially Present")
        self.XIDS_MAY_INTERSECT = ContentOfColumn(xid_policy_header, self.RAWROW, "XIDs\nOptionally Present")
        self.XIDS_MUST_NOT_INTERSECT = ContentOfColumn(xid_policy_header, self.RAWROW, "XIDs\nAbsent")

        self.SET_XIDS_MUST_EQUAL = policy_xidlist_2_set(self.XIDS_MUST_EQUAL)
        self.SET_XIDS_MUST_INTERSECT = policy_xidlist_2_set(self.XIDS_MUST_INTERSECT)
        self.SET_XIDS_MUST_NOT_INTERSECT = policy_xidlist_2_set(self.XIDS_MUST_NOT_INTERSECT)
        self.SET_XIDS_MAY_INTERSECT = policy_xidlist_2_set(self.XIDS_MAY_INTERSECT)

    def Matches(self, observed_burst: XIDBurst):
        applies = True

        if self.SET_XIDS_MUST_EQUAL and not self.SET_XIDS_MUST_EQUAL.issubset(observed_burst.xids):
            applies = False

        if self.SET_XIDS_MUST_INTERSECT and not self.SET_XIDS_MUST_INTERSECT.intersection(observed_burst.xids):
            applies = False

        if self.SET_XIDS_MUST_NOT_INTERSECT:
            if self.SET_XIDS_MUST_NOT_INTERSECT == {"*"}:
                if (
                    observed_burst.xids
                    - self.SET_XIDS_MUST_INTERSECT
                    - self.SET_XIDS_MUST_EQUAL
                    - self.SET_XIDS_MAY_INTERSECT
                ):
                    applies = False
            elif self.SET_XIDS_MUST_NOT_INTERSECT.intersection(observed_burst.xids - self.SET_XIDS_MAY_INTERSECT):
                applies = False

        return applies


@dataclass
class XidParsingPolicy:
    XIDPolicyEntries: list[XIDBurstPattern]

    def __init__(self):
        self.XIDPolicyEntries = []
        try:
            policy = load_xid_catalog_data()["xid_parsing_policy"]

            global xid_policy_header
            xid_policy_header = policy["header"]

            for row_list in policy["rows"]:
                self.XIDPolicyEntries.append(XIDBurstPattern(row_list))

            self.XIDPolicyEntries.sort(key=lambda x: int(x.PRIORITY) if x.PRIORITY.isdigit() else float("inf"))

            global XID_WATCHLIST

            XID_WATCHLIST = sorted({x.name: x for x in XID_WATCHLIST}.values(), key=lambda x: len(x.name), reverse=True)

        except Exception as e:
            logging.error(f"Error loading XID parsing policy: {e}")


def SplitListToBursts(xids_list):
    """Groups XIDs into bursts using the global STICKY_XIDS variable."""
    global STICKY_XIDS
    bursts = []
    xids_list.sort(key=lambda x: x.timestamp)

    past_ts = ensure_timezone_local(datetime.now() - timedelta(weeks=1000))
    future_ts = ensure_timezone_local(datetime.now() + timedelta(weeks=100))
    last_ts = future_ts
    last_driver = ""

    last_sticky_ts = {}
    for sx in STICKY_XIDS:
        last_sticky_ts[sx] = past_ts

    burst_to_add = []

    for xid in xids_list:
        new_burst = True

        if xid.name in STICKY_XIDS and xid.timestamp - last_sticky_ts[xid.name] < timedelta(hours=3):
            new_burst = False

        if xid.timestamp - last_ts < timedelta(minutes=3):
            new_burst = False

        if xid.driver != last_driver:
            new_burst = True

        if new_burst:
            if burst_to_add:
                xid_burst = XIDBurst.from_xid_list(burst_to_add)
                bursts.append(xid_burst)
                burst_to_add = []

        burst_to_add.append(xid)
        last_ts = xid.timestamp
        last_driver = xid.driver
        if xid.name in STICKY_XIDS:
            last_sticky_ts[xid.name] = last_ts

    if burst_to_add:
        xid_burst = XIDBurst.from_xid_list(burst_to_add)
        bursts.append(xid_burst)

    return bursts


def load_nvl5_xids_decoding_logic():
    """Load XID patterns from the 144-150 decode data."""
    try:
        rules = load_xid_catalog_data()["nvl5_xid_decoding_rules"]
        return {
            int(xid): [{key: _restore_missing(value) for key, value in pattern.items()} for pattern in patterns]
            for xid, patterns in rules.items()
        }
    except Exception as e:
        logging.error(f"Error loading XID 144-150 decode rules: {e}")
        return {}


def load_xid_sheet():
    """Load XID/SXID catalog information."""
    try:
        catalog = load_xid_catalog_data()["xid_catalog"]
        return {code: {key: _restore_missing(value) for key, value in entry.items()} for code, entry in catalog.items()}
    except Exception as e:
        logging.error(f"Error loading XID catalog: {e}")
        return {}


def does_xid_intrinfo_match_rule(intrinfo_in_rule, intrinfo_in_message):
    logging.debug(f"intrinfo_in_rule: {intrinfo_in_rule}")
    logging.debug(f"intrinfo_in_message: {intrinfo_in_message}")

    csv_intrinfo_mask_str = intrinfo_in_rule.replace("0", "1")
    csv_intrinfo_mask = int(csv_intrinfo_mask_str.replace("-", "0"), 2)
    csv_intrinfo_value = int(intrinfo_in_rule.replace("-", "1"), 2)

    lhs = intrinfo_in_message & csv_intrinfo_mask
    rhs = csv_intrinfo_value & csv_intrinfo_mask
    return lhs == rhs


def setup_rules():
    """Initialize XID resolution rules."""
    global XID_CATALOG, NVL5_XID_DECODING_RULES, STICKY_XIDS, XID_POLICY

    load_xid_catalog_data()

    XID_CATALOG = load_xid_sheet()
    NVL5_XID_DECODING_RULES = load_nvl5_xids_decoding_logic()

    STICKY_XIDS = []
    for xid, entry in XID_CATALOG.items():
        if entry.get("sticky") == "YES":
            STICKY_XIDS.append(xid)

    XID_POLICY = XidParsingPolicy()


def ensure_timezone_local(dt):
    """Ensure a datetime has timezone info using the system's local timezone."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        try:
            import tzlocal

            local_timezone = tzlocal.get_localzone()
            return dt.replace(tzinfo=local_timezone)
        except ImportError:
            try:
                return dt.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
            except Exception:
                import pytz

                return dt.replace(tzinfo=pytz.timezone("America/Los_Angeles"))
    return dt
