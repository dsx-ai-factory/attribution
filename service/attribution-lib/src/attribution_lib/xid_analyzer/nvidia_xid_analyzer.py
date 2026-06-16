#!/usr/bin/env python3

"""
Author: NVIDIA Corporation
Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

"""
NVIDIA XID Pattern Analyzer
---------------------------
Analyzes GPU XID (error) messages in two modes:
1. Single XID Analysis: Decode and categorize individual XID messages
2. Fleet Analysis: Analyze XID patterns across a fleet of GPUs
"""

import argparse
import pathlib
import sys
import logging
import csv
import re
import collections
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
import json
from zoneinfo import ZoneInfo
from typing import Optional, Union
from tabulate import tabulate  # pip install tabulate
import pandas as pd  # pip install pandas and pip install openpyxl
import shutil
import pytz
import tzlocal

# Global constants
INFINITE_MTBI = timedelta(days=3650)  # ~10 years as a representation of "infinity" for MTBI calculations


# Configure logging level based on command-line argument
logging.basicConfig(format='%(message)s', stream=sys.stdout)
# Get the folder containing this script
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

# The XID resolution data (catalog, NVL5 decode rules, and parsing policy) is
# embedded as JSON next to this module instead of an upstream vendor catalog
# source.
# Empty cells are stored as JSON null and restored to float('nan') on load so
# downstream math.isnan() checks keep working.
XID_CATALOG_DATA = SCRIPT_DIR / 'xid_catalog_data.json'

# Cache so the embedded data JSON is read at most once.
_XID_DATA_CACHE = None

# Global variables for XID resolution data
XID_CATALOG = None
NVL5_XID_DECODING_RULES = None
STICKY_XIDS = None
XID_POLICY = None
XID_WATCHLIST = []


def _restore_missing(value):
    # Empty cells are stored as JSON null; restore them to the float('nan')
    # the analyzer historically produced when it read the workbook directly.
    return float('nan') if value is None else value


def load_xid_catalog_data():
    """
    Load and cache the embedded XID catalog data JSON.

    Returns:
        dict: Parsed JSON with 'xid_catalog', 'nvl5_xid_decoding_rules', and
        'xid_parsing_policy' keys.
    """
    global _XID_DATA_CACHE
    if _XID_DATA_CACHE is None:
        with open(XID_CATALOG_DATA, encoding='utf-8') as data_file:
            _XID_DATA_CACHE = json.load(data_file)
    return _XID_DATA_CACHE


def normalized_string(s):
    # Handle None, NaN or non-string inputs
    if s is None or pd.isna(s) or not isinstance(s, str):
        return str(s)
    # Remove all whitespace characters
    return re.sub(r'\s+', '', s)

def ContentOfColumn(header, row, col_name):
    """
    Find and return the content of a column by matching normalized column names.
    Normalizes strings by removing all whitespace characters.
    
    Args:
        header: List of column headers
        row: List of values for a row
        col_name: Name of the column to find
        
    Returns:
        The value from the row at the matching column index or an empty string if column not found
    """
    for index, name in enumerate(header):
         # Normalize the column name we're searching for
        if normalized_string(col_name) in normalized_string(name):
            value = row[index]
            # Convert float/int to string if needed
            if isinstance(value, (float, int)):
                return str(value)
            return value
    return ""

def row_to_kv_pairs(header, row):
    kv_pairs = {}
    for index, name in enumerate(header):
        kv_pairs[name] = row[index]
    return kv_pairs

def policy_xidlist_2_set(xidlist):
    
    xidset = set()
    global XID_WATCHLIST

    # Handle float or None input - convert to empty string
    if not xidlist or pd.isna(xidlist):
        return xidset
    
    # Convert float or int to string if needed
    if isinstance(xidlist, (float, int)):
        xidlist = str(xidlist)
    
    # logging.info(f'Processing entry: {xidlist}')
    xidlist = xidlist.replace('\n', '').replace('\r', '')

    for xid in re.split(r',\s*(?=\d+\{|\d+\s*,|\d+$|\d+\s*$)', xidlist):

        logging.debug(f"\nProcessing Rule Fragment xid: {xid}")

        if not xid or xid == '':       # Skip empty rows
            continue

        if pd.isna(xid) or xid.lower() == 'nan':  # Skip NaN values
            continue

        if xid == '*':      #Skip universal catch-all rows
            xidset.add("*")
            continue

        xid = xid.strip()   #Remove newlines

        xid_ctx = ""
        xid_drv = ""
        xid_name = xid
        xid_subcode = ""
        xid_intrinfo = ""
        xid_errorstatus = ""
        xid_num = 0  # Default value to prevent reference before assignment

        entry_pattern = re.compile(r'(?P<number>\d+|\*)(?:\{(?P<properties>.*?)\})?')
        match = entry_pattern.match(xid)
        if match:
            xid_num = int(match.group('number')) if match.group('number') != '*' else 0
            if match.group('properties') is not None:
                try:
                    properties = '{' + match.group('properties') + '}'
                    properties_dict = json.loads(properties)
                    
                    if 'context' in properties_dict:
                        xid_ctx = properties_dict['context']
                    if 'driver' in properties_dict:
                        xid_drv = properties_dict['driver']
                    if 'errorstatus' in properties_dict:
                        xid_errorstatus = properties_dict['errorstatus']
                    if 'subcode' in properties_dict:
                        xid_subcode = properties_dict['subcode']
                    if 'intrinfo' in properties_dict:
                        xid_intrinfo = properties_dict['intrinfo']
                    if 'rate' in properties_dict:   #TODO: Handling of rates is TBD
                        del properties_dict['rate']
                        properties = json.dumps(properties_dict)
                    xid_name = str(xid_num) + properties
                except json.JSONDecodeError as e:
                    logging.warning(f"JSON parsing error in entry '{xid}': {e}")
                    # Fall back to using just the XID number
                    xid_name = str(xid_num)
        else:
            logging.warning(f"Failed to parse entry {xid}")
            # Skip entries that don't match the expected pattern
            continue

        xidset.add(xid_name)

        logging.debug(f"\nAdding XIDEntry to XID_WATCHLIST: name={xid_name}, number={xid_num}, context={xid_ctx}, driver={xid_drv}, subcode={xid_subcode}, intrinfo={xid_intrinfo}, errorstatus={xid_errorstatus}\n")
        XID_WATCHLIST.append(XIDEntry(name=xid_name, number=xid_num, context=xid_ctx, driver=xid_drv, subcode=xid_subcode, intrinfo=xid_intrinfo, errorstatus=xid_errorstatus))

    return xidset

def unique_values(data, attribute):
    return set(getattr(item, attribute) for item in data)


@dataclass
class XIDEntry:
    """Structured class for storing and extracting NVIDIA XID error information"""
    name: str                       # Unique Name for this xid (=number+context+driver etc.). This is used to create & compare sets
    number: int                     # The XID number itself
    context: str                    # Additional string that comes along with the XID number
    driver: str                     # Driver version that reported this XID
    timestamp: Optional[Union[datetime, str]] = None  # The time at which the XID happened
    machine: str = "na"             # Machine ID which reported the XID
    source: Optional[str] = None    # source of the line in the dmesg log, used by FACT

    pcie_bdf: Optional[str] = None  # PCIe bus:device.function identifier
    subcode: Optional[str] = None   # Specific error subcode (like NETIR_LINK_EVT/NETIR_LINK_DOWN)
    severity: Optional[str] = None  # Severity level of the error
    cross_contain: Optional[str] = None  # Cross-containment information
    injected: Optional[str] = None  # Injection flags
    link: Optional[str] = None      # Link identifier
    intrinfo: Optional[str] = None  # Interrupt information (hex)
    errorstatus: Optional[str] = None  # Error status code (hex)
    nvl5_port_down_reason: Optional[int] = None  # NVL5 port down reason (6 bits)
    nvl5_error_code: Optional[int] = None  # NVL5 error code (7 bits)
    vbios_version: Optional[str] = None  # GPU VBIOS version
    mnemonic: Optional[str] = None  # Mnemonic from the XID catalog
    
    # Resolution-related fields
    resolution: Optional[str] = None  # Resolution action (e.g., RESET_GPU)
    investigatory_action: Optional[str] = None  # Investigatory or deferred action
    comments: Optional[str] = None  # Additional comments or details
    decoded_xid_string: Optional[str] = None  # Decoded XID string (e.g. "31:ROBUST_CHANNEL_FIFO_ERROR_MMU_ERR_FLT","149:149.4", etc.)
    
    job_failure_severity: Optional[str] = None  # Job Failure Severity (Job-Fatal, Job-Non-Fatal, Possibly-Job-Fatal)
    hw_sw_classification: Optional[str] = None  # HW-SW Classification (HW, SW, FW, HW/SW, NA)

    def to_dict(self):
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def to_json(self, indent=2):
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), 
                          indent=indent, 
                          default=lambda obj: obj.isoformat() if isinstance(obj, datetime) else str(obj))
    
    @classmethod
    def to_table(cls, entries, fields=None, format="psql"):
        """
        Format XID entries as a table. Works with both single entry and multiple entries.
        
        Args:
            entries: Single XIDEntry object or list of XIDEntry objects
            fields: List of field names to include, or None for default fields
            format: Table format for tabulate (e.g., "psql", "grid", "plain")
            
        Returns:
            str: Formatted table as string
        """
        # Default fields if none specified
        if fields is None:
            fields = ['timestamp', 'number', 'mnemonic', 'driver', 'vbios_version', 'machine', 'context', 'resolution', 'job_failure_severity', 'hw_sw_classification']
        
        # Handle field name to display header mapping
        def get_display_name(attr):
            display_name = attr.replace('_', ' ').title()
            name_map = {
                'Timestamp': 'Timestamp',
                'Number': 'XID Number',
                'Mnemonic': 'Mnemonic',
                'Driver': 'SW Ver(Driver)',
                'Vbios Version': 'FW Ver(VBIOS)',
                'Machine': 'Machine Id', 
                'Context': 'XID Message',
                'Resolution': 'Resolution',
                'Pcie Bdf': 'PCIe BDF',
                'Intrinfo': 'Interrupt Info',
                'Errorstatus': 'Error Status',
                'Investigatory Action': 'Investigatory/Deferred Action',
                'Job Failure Severity': 'Job Failure Severity',
                'Hw Sw Classification': 'HW/SW Classification'
            }
            return name_map.get(display_name, display_name)
        
        # Format a value for display
        def format_value(value):
            if isinstance(value, datetime):
                return value.isoformat()
            elif value is None:
                return 'N/A'
            else:
                return str(value)
        
        # For empty data
        if entries is None:
            return "No entry to display"
            
        # Always use horizontal format (matching the README examples)
        if not isinstance(entries, list):
            # Convert single entry to a list with one element
            entries = [entries]
            
        if not entries:
            return "No entries to display"
            
        # Get headers
        headers = [get_display_name(attr) for attr in fields]
        
        # Build table data
        table_data = []
        for entry in entries:
            row = []
            for attr in fields:
                if hasattr(entry, attr):
                    value = getattr(entry, attr)
                    # Use empty string instead of N/A for None values in horizontal format
                    row.append(format_value(value) if value is not None else '')
                else:
                    row.append('')
            table_data.append(row)
        
        return tabulate(table_data, headers=headers, tablefmt=format)
    
    def parse_timestamp(self):
        """
        Parse timestamp string into a datetime object with proper timezone.
        This handles various timestamp formats and adds default timezone if needed.
        """
        if not self.timestamp:
            return
        
        # If timestamp is already a datetime object, no need to parse
        if isinstance(self.timestamp, datetime):
            # Just ensure it has a timezone
            if self.timestamp.tzinfo is None:
                self.timestamp = ensure_timezone_local(self.timestamp)
            return
            
        # Convert timestamp to string if it's not already
        timestamp_str = str(self.timestamp).replace('0000000', '000000')
        
        # Fix timezone format if needed
        if re.search(r'[+-]\d{2}$', timestamp_str):
            timestamp_str += "00"
        
        # Try parsing the date with different formats
        formats = [
            "%Y-%m-%d %H:%M:%S.%f%z",   # 2024-03-22 09:25:40.475505-07:00 or 2024-03-22 09:25:40.475505-0700
            "%Y-%m-%d %H:%M:%S.%f",     # 2024-07-02 20:11:58.0000000 (no timezone)
            "%Y-%m-%d %H:%M:%S",        # 2024-07-02 20:11:58.0000000 (no timezone, no milliseconds)
            "%Y-%m-%d %H:%M:%S%z",      # 2020-01-31 13:11:01+00:00
            "%a %b %d %H:%M:%S %Y",     # Tue Feb 11 04:11:05 2025
            "%a %b  %d %H:%M:%S %Y",    # Tue Feb  1 04:11:05 2025 (note double space for single-digit days)
            "%a %b %d %H:%M:%S %Z %Y",  # Mon May 25 21:07:12 UTC 1970
        ]
        
        parsed = False
        for fmt in formats:
            try:
                self.timestamp = datetime.strptime(timestamp_str, fmt)
                parsed = True
                break
            except ValueError:
                continue
        
        # Handle Unix timestamp (microseconds since epoch)
        if not parsed and timestamp_str.isdigit():
            try:
                # Convert microseconds to seconds
                timestamp_seconds = float(timestamp_str) / 1000000
                self.timestamp = datetime.fromtimestamp(timestamp_seconds)
                parsed = True
            except (ValueError, OverflowError):
                pass
        
        # If still not parsed, try a more flexible approach for common log formats
        if not parsed:
            try:
                # This is a more permissive approach using dateutil
                from dateutil import parser as dateutil_parser
                self.timestamp = dateutil_parser.parse(timestamp_str)
                parsed = True
            except (ImportError, ValueError):
                # If dateutil is not available or parsing fails
                pass
            
            # Try to normalize the format with single-digit days (fix the double space)
            if not parsed and re.search(r'[A-Za-z]{3}\s+\s+\d{1}\s', timestamp_str):
                try:
                    # Convert "Dec  7" to "Dec 7" (remove extra space)
                    fixed_ts = re.sub(r'([A-Za-z]{3})\s+\s+(\d{1}\s)', r'\1 \2', timestamp_str)
                    self.timestamp = datetime.strptime(fixed_ts, "%a %b %d %H:%M:%S %Y")
                    parsed = True
                except ValueError:
                    pass
        
        if not parsed and timestamp_str:
            # If we can't parse the timestamp, log a warning but don't fail
            logging.debug(f"Date string '{timestamp_str}' could not be parsed. Using current time instead.")
            self.timestamp = datetime.now()
        
        # Add timezone info if needed
        if self.timestamp and self.timestamp.tzinfo is None:
            self.timestamp = ensure_timezone_local(self.timestamp)
    
    @classmethod
    def from_message(cls, message=None, timestamp=None, driver="unknown", machine="na", vbios_version=None):
        """
        Extract XID information from a message and create an XIDEntry object.
        
        Args:
            message (str): The error message containing an XID.
            timestamp (str or datetime, optional): Timestamp when the error occurred
            driver (str): Driver version that reported this XID
            machine (str): Machine ID which reported the XID
            vbios_version (str, optional): VBIOS version of the GPU
            
        Returns:
            XIDEntry: Object with XID information or None if no XID found
        """
        if not message:
            return None
        
        # Try to extract timestamp from the message if not provided
        if not timestamp and message:
            # For dmesg format, try to extract timestamp
            if "NVRM: Xid" in message:
                # Try full timestamp format [Wed Dec 11 13:23:01 2024] with flexible spacing
                full_ts_match = re.search(r'^\[\s*([A-Za-z]{3}\s+[A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4})\s*\]', message)
                if full_ts_match:
                    timestamp = full_ts_match.group(1).strip()
                
                # Try kernel timestamp format [12.153079]
                if not timestamp:
                    timestamp_match = re.search(r'^\[\s*([0-9.]+)\s*\]', message)
                    if timestamp_match:
                        timestamp = timestamp_match.group(1)
                
                # Try syslog timestamp format: Jan 1 12:34:56
                if not timestamp:
                    syslog_match = re.search(r'^([A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})', message)
                    if syslog_match:
                        timestamp = syslog_match.group(1)
            
            # For redfish format, extract timestamp
            elif "Driver Event Message has detected errors of type" in message:
                redfish_match = re.search(r'\[\s*([^\]]+)\s*\](?:\[[^\]]*\]){0,2} XID', message)
                if redfish_match:
                    timestamp = redfish_match.group(1).strip()
            
        # Pattern 1: NVL5 XID message in redfish (longest pattern)
        # example: The resource property GPU_1 Driver Event Message has detected errors of type '[Tue Jan 28 22:37:24 2025][18][0] XID 149 NETIR_LINK_EVT  Fatal   XC0 i0 Link 15 (0x0045e5c6 0x00000000 0x00000000 0x00000000 0x00000000 0x00000000)'
        pattern_nvl5_redfish = re.compile(
            r'Driver Event Message has detected errors of type \'\[(?P<timestamp>[^\]]+)\](?:\[[^\]]*\]){0,2} XID (?P<xid_number>\d+)\s+(?P<subcode>\w+)\s+(?P<severity>\w+)\s+(?P<cross_contain>\w+)\s+(?P<injected>\w+)\s+Link\s+(?P<link>-?\d+)\s+\((?P<intrinfo>0x[0-9a-fA-F]+)\s+(?P<errorstatus>0x[0-9a-fA-F]+)'
        )
        match = pattern_nvl5_redfish.search(message)
        if match:
            logging.debug(f'Matched pattern_nvl5_redfish')
            xid_number = int(match.group("xid_number"))
            xid_entry = cls(
                name='',
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
                vbios_version=vbios_version
            )
            xid_entry.parse_timestamp()
            return xid_entry

        # Pattern 2: NVL5 XID message in dmesg (second longest)
        # example1: NVRM: Xid (PCI:0018:01:00): 149, NETIR_LINK_EVT  Fatal   XC0 i0 Link 17 (0x025625c6 0x00000000 0x00000000 0x00000000 0x00000000 0x00000000)
        # example2: Mar 30 19:16:18 mvat4 kernel: NVRM: Xid (PCI:0018:01:00): 145, pid='<unknown>', name=<unknown>, RLW_SRC_TRACK Nonfatal XC1 i0 Link 17 (0x040a28e2 0x00000008 0x00000000 0x00000000 0x00000000 0x00000000)
        # pattern_nvl5_dmesg = re.compile(
        #     r'NVRM: Xid \(PCI:(?P<pcie_bdf>[^)]+)\): (?P<xid_number>\d+)(?:, pid=\d+, name=[^,]*,)?, (?P<subcode>\w+)\s+(?P<severity>\w+)\s+(?P<cross_contain>\w+)\s+(?P<injected>\w+)\s+Link\s+(?P<link>-?\d+)\s+\((?P<intrinfo>0x[0-9a-fA-F]+)\s+(?P<errorstatus>0x[0-9a-fA-F]+)'
        # )
        pattern_nvl5_dmesg = re.compile(
            r'NVRM: Xid \(PCI:(?P<pcie_bdf>[^)]+)\): (?P<xid_number>\d+)(?:, pid=\d+, name=(?:[^,]*))?, (?P<subcode>\w+)\s+(?P<severity>\w+)\s+(?P<cross_contain>\w+)\s+(?P<injected>\w+)\s+Link\s+(?P<link>-?\d+)\s+\((?P<intrinfo>0x[0-9a-fA-F]+)\s+(?P<errorstatus>0x[0-9a-fA-F]+)'
        )
        match = pattern_nvl5_dmesg.search(message)
        if match:
            logging.debug(f'Matched pattern_nvl5_dmesg')
            xid_number = int(match.group("xid_number"))
            xid_entry = cls(
                name='',
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
                vbios_version=vbios_version
            )
            xid_entry.parse_timestamp()
            return xid_entry
        
        # Pattern 3: Simple XID message in redfish (third longest)
        # example: The resource property GPU_1 Driver Event Message has detected errors of type '[Wed Jan 29 09:58:21 2025][2][0] XID 45 Ch 00000001'.
        pattern_redfish = re.compile(
            # r'Driver Event Message has detected errors of type \'\[(?P<timestamp>[^\]]+)\](?:\[[^\]]*\]){0,2} XID (?P<xid_number>\d+)(?:\s+([^\']+))?\''
            r'Driver Event Message has detected errors of type \'\[(?P<timestamp>[^\]]+)\](?:\[[^\]]*\]){0,2} XID (?P<xid_number>\d+)(?:\s+[^\']*)?'
        )
        match = pattern_redfish.search(message)
        if match:
            logging.debug(f'Matched pattern_redfish')
            xid_number = int(match.group("xid_number"))
            xid_entry = cls(
                name='',
                number=xid_number,
                context=message,
                driver=driver,
                timestamp=timestamp if timestamp else match.group("timestamp"),
                machine=machine,
                # No other fields available in this pattern
            )
            xid_entry.parse_timestamp()
            return xid_entry

        # Pattern 4: Simple XID message in dmesg (fourth longest)
        # example: NVRM: Xid (PCI:0000:ae:00): 13, pid='<unknown>', name=<unknown>, Graphics SM Warp Exception on (GPC 2, TPC 0, SM 0): Illegal Instruction Encoding
        pattern_dmesg = re.compile(
            r'NVRM: Xid \(PCI:(?P<pcie_bdf>[^)]+)\): (?P<xid_number>\d+), pid=(?:\'[^\']*\'|[^,]*), name=[^,]*, (.*)'
        )
        match = pattern_dmesg.search(message)
        if match:
            logging.debug(f'Matched pattern_dmesg')
            xid_number = int(match.group("xid_number"))
            xid_entry = cls(
                name='',
                number=xid_number,
                context=message,
                driver=driver,
                timestamp=timestamp if timestamp else None,
                machine=machine,
                pcie_bdf=match.group("pcie_bdf")
                # No other fields available in this pattern
            )
            xid_entry.parse_timestamp()
            return xid_entry
        
        # Pattern 5: Fallback pattern for simple dmesg XID messages
        # This is a more generic pattern that should catch most dmesg XID messages
        pattern_dmesg_simple = re.compile(
            r'NVRM: Xid \(PCI:(?P<pcie_bdf>[^)]+)\): (?P<xid_number>\d+)'
        )
        match = pattern_dmesg_simple.search(message)
        if match:
            logging.debug(f'Matched pattern_dmesg_simple')
            xid_number = int(match.group("xid_number"))
            xid_entry = cls(
                name='',
                number=xid_number,
                context=message,
                driver=driver,
                timestamp=timestamp if timestamp else None,
                machine=machine,
                pcie_bdf=match.group("pcie_bdf"),
                vbios_version=vbios_version
                # No other fields available in this pattern
            )
            xid_entry.parse_timestamp()
            return xid_entry
                
        return None
        
    def lookup_resolution(self):
        """
        Lookup resolution information for this XID entry.
        
        This method enhances the XID entry with resolution information by:
        1. Looking up relevant information in the XID catalog
        2. Applying NVL5-specific decoding rules if applicable
        
        Returns:
            XIDEntry: Self with enhanced resolution information
        """
        global XID_CATALOG, NVL5_XID_DECODING_RULES
        
        try:
            logging.debug(f"Looking up resolution for XID: {self.number}")
            
            # Enhance with catalog information
            if XID_CATALOG and str(self.number) in XID_CATALOG:
                logging.debug(f"1st level lookup of xid in XID_CATALOG")
                catalog_entry = XID_CATALOG[str(self.number)]
                self.resolution = catalog_entry.get('resolution_datacenter', 'Unknown')
                
                # Add mnemonic from catalog
                self.mnemonic = catalog_entry.get('mnemonic', '')
                # self.decoded_xid_string = str(self.number) + ":" + self.mnemonic
                self.decoded_xid_string = str(self.number)
                
                # Add investigatory action if available
                if catalog_entry.get('resolution_investigatory'):
                    self.investigatory_action = catalog_entry.get('resolution_investigatory')
                
                # Special handling for XID_154
                if catalog_entry.get('resolution_datacenter') == "XID_154":
                    extracted_text = get_resolution_for_xid154(self.context)
                    logging.debug(f"2nd level lookup of xid {self.number} in XID_CATALOG")
                    if extracted_text:
                        self.resolution = extracted_text
                    else:
                        self.resolution = f"No resolution found for XID_154"
                 
                # Special handling for WORKFLOW_NVLINK5_ERR
                if catalog_entry.get('resolution_datacenter') == "WORKFLOW_NVLINK5_ERR" and NVL5_XID_DECODING_RULES:
                    logging.debug(f"2nd level lookup of xid {self.number} in NVL5_XID_DECODING_RULES")
                    
                    # Check if we have the necessary attributes for NVL5 matching
                    if hasattr(self, 'subcode') and hasattr(self, 'errorstatus') and hasattr(self, 'intrinfo'):
                        logging.debug(f"XID entry has required NVL5 attributes")
                        
                        # Extract port down reason and error code from intrinfo
                        self._extract_port_down_reason_and_error_code_from_intrinfo()

                        # Check if this XID exists in the NVL5 rules
                        if int(self.number) in NVL5_XID_DECODING_RULES:
                            logging.debug(f"Found XID {self.number} in NVL5 rules")
                            
                            for nvl5_decoding_rule in NVL5_XID_DECODING_RULES[int(self.number)]:
                                logging.debug(f"Checking NVL5 rule: {nvl5_decoding_rule}")
                                try:                                
                                    # Check if this rule matches
                                    if (self.subcode in nvl5_decoding_rule['sheet_subcode'] and 
                                        self.errorstatus in nvl5_decoding_rule['sheet_error_status_hex'] and 
                                        does_xid_intrinfo_match_rule(nvl5_decoding_rule['sheet_intrinfo_binary'], self.intrinfo)):
                                        
                                        logging.debug(f"Found matching NVL5 rule: {nvl5_decoding_rule}")
                                        self.resolution = nvl5_decoding_rule['sheet_resolution']
                                        
                                        # Add investigatory action from NVL5 rules if available
                                        if nvl5_decoding_rule.get('sheet_investigatory_action'):
                                            self.investigatory_action = nvl5_decoding_rule['sheet_investigatory_action']
                                        
                                        # Add comments if available to provide more details
                                        if nvl5_decoding_rule['sheet_comments']:
                                            self.comments = nvl5_decoding_rule['sheet_comments']
                                            
                                        # Determine Job Failure Severity based on NVL5 rule
                                        self._determine_job_failure_severity(nvl5_decoding_rule)
                                        
                                        # Determine HW/SW Classification based on NVL5 rule
                                        self._determine_hw_sw_classification(nvl5_decoding_rule)
                                        
                                        break
                                    # Check for message like "Mar 30 19:16:09 mvat4 kernel: NVRM: Xid (PCI:0019:01:00): 145, pid='unknown', name=<unknown>, Ch 00000002"
                                    else:
                                        # HACK: hard coded policy which should be coming from the XID catalog finally
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
                        logging.debug(f"XID entry missing required NVL5 attributes")
            else:
                logging.debug(f"XID {self.number} not found in catalog")
            
            # Special handling for XID_154_EVAL
            if self.resolution == "XID_154_EVAL":
                self.resolution = "Follow guidance from 154 if present otherwise RESTART_APP"
                
            # Only set default Job Failure Severity for XIDs between 144-150 if not already set
            if 144 <= self.number <= 150 and self.job_failure_severity is None:
                self.job_failure_severity = "Job-Non-Fatal"
                
            # Return self with enhanced resolution information
            return self
            
        except Exception as e:
            logging.error(f"Error looking up resolution for XID: {e}")
            return self
            
    def _determine_job_failure_severity(self, nvl5_decoding_rule):
        """
        Determine Job Failure Severity based on NVL5 XID decoding rule.
        
        Logic (Only applied for XIDs between 144-150):
        1. If column K contains "fatal" (case-insensitive) → Job-Fatal
        2. If column K contains "fatal**" → Possibly Job-Fatal
        3. If column K contains "non-fatal" or "link fatal":
           a. If column M contains "AppCrash" → Job-Fatal
           b. Else if Column M lists other XIDs → evaluate their severity using the "XIDs" sheet:
              If any listed XID is marked as Yes in XIDs sheet column Job-Fatal → Job-Fatal
        4. If column K contains "non-fatal*":
           a. If the XID string contains "fatal" → Job-Fatal
        5. Default → Job-Non-Fatal
        
        For XIDs outside the 144-150 range, job_failure_severity is not set.
        
        Args:
            nvl5_rule: Dictionary containing the matched NVL5 rule
        """
        global XID_CATALOG
        
        # Only apply job failure severity logic for XIDs between 144-150
        if not (144 <= self.number <= 150):
            return
            
        # Column K corresponds to Severity and Column M to Local/Remote in the rule
        severity = nvl5_decoding_rule.get('sheet_severity', '').lower()
        local_remote = nvl5_decoding_rule.get('sheet_local_remote', '').lower()
        
        # Case 1: Contains "fatal" but not special cases like "fatal**", "non-fatal", etc.
        fatal_exclusions = ["fatal**", "non-fatal", "link fatal", "link fatal?"]
        if "fatal" in severity and not any(exclusion in severity for exclusion in fatal_exclusions):
            self.job_failure_severity = "Job-Fatal"
            logging.debug(f"Setting Job-Fatal due to 'fatal' in Severity column")
            return
            
        # Case 2: "fatal**" in column K -> Possibly Job-Fatal
        if "fatal**" in severity:
            self.job_failure_severity = "Possibly-Job-Fatal"
            logging.debug(f"Setting Possibly-Job-Fatal due to 'fatal**' in Severity column. May or may not be Job-Fatal.")
            return
            
        # Case 3: Contains "non-fatal" or "link fatal"
        non_fatal_terms = ["non-fatal", "link fatal", "link fatal?"]
        if any(term in severity for term in non_fatal_terms):
            # Case 3a: If local_remote contains "AppCrash" → Job-Fatal
            if "appcrash" in local_remote:
                self.job_failure_severity = "Job-Fatal"
                logging.debug(f"Setting Job-Fatal due to 'AppCrash' in Local/Remote column")
                return
            
            # Case 3b: Check if local_remote references other XIDs
            if local_remote and XID_CATALOG:
                xid_numbers = re.findall(r'\b\d+\b', local_remote)
                
                if xid_numbers:
                    logging.debug(f"Found potential XID references: {xid_numbers}")
                    for xid_num in xid_numbers:
                        if xid_num in XID_CATALOG:
                            catalog_entry = XID_CATALOG[xid_num]
                            job_fatal_value = catalog_entry.get('job_fatal', '').lower()
                            
                            # If referenced XID is marked as Job-Fatal
                            if job_fatal_value == 'yes':
                                self.job_failure_severity = "Job-Fatal"
                                logging.debug(f"Setting Job-Fatal due to referenced XID {xid_num}")
                                return
                
        # Case 4: Contains "non-fatal*" and context contains "fatal"
        if "non-fatal*" in severity and self.context and "fatal" in self.context.lower():
            self.job_failure_severity = "Job-Fatal"
            logging.debug(f"Setting Job-Fatal due to 'fatal' in context with 'non-fatal*'")
            return
                
        # Default case: Job-Non-Fatal
        self.job_failure_severity = "Job-Non-Fatal"
        logging.debug(f"Default case: Setting Job-Non-Fatal")
        
    def _determine_hw_sw_classification(self, nvl5_decoding_rule):
        """
        Determine HW/SW Classification based on NVL5 XID decoding rule.
        
        This method extracts the HW/SW Classification from column L in the XID 144-150 sheet
        and sets it for XIDs between 144-150.
        
        Args:
            nvl5_rule: Dictionary containing the matched NVL5 rule
        """
        # Only apply HW/SW classification for XIDs between 144-150
        if not (144 <= self.number <= 150):
            return
            
        # Get HW/SW classification from the rule
        hw_sw_classification = nvl5_decoding_rule.get('sheet_hw_sw_classification', '')
        
        if hw_sw_classification:
            self.hw_sw_classification = hw_sw_classification
            logging.debug(f"Setting HW/SW Classification to '{hw_sw_classification}' for XID {self.number}")
        else:
            self.hw_sw_classification = ""
            logging.debug(f"No HW/SW Classification found for XID {self.number}, setting to empty string")

    def _extract_port_down_reason_and_error_code_from_intrinfo(self):
        """
        Extract port down reason and error code from intrinfo
        """
        if self.intrinfo is not None:
            # Extract bits 25:20 for port_down_reason (6 bits, 25 is msb, 20 is lsb)
            self.nvl5_port_down_reason = (self.intrinfo >> 20) & 0x3F  # 0x3F = 0b111111
            # Extract bits 6:0 for error_code (7 bits, 6 is msb, 0 is lsb)
            self.nvl5_error_code = self.intrinfo & 0x7F  # 0x7F = 0b1111111
            # self.decoded_xid_string = self.number + ":" + self.mnemonic + "." + str(self.nvl5_error_code)
            self.decoded_xid_string = f"{self.number}.{self.nvl5_port_down_reason:X}h"
        else:
            logging.debug(f"No intrinfo found for XID Message: {self.context}")
            self.nvl5_port_down_reason = 0
            self.nvl5_error_code = 0

@dataclass
class XIDBurst:
    """Class to represent a burst (group of XIDs within 3 minutes of each other)"""
    xids: set[XIDEntry]     # Set of XIDs in the burst
    xid_list: list[XIDEntry]  # List of XIDEntry objects in the burst (to help debug bursts categorization and attribution issues)
    decoded_xid_set: set[str] # Set of decoded XID strings in the burst
    count: int              # Number of actual XIDs in the burst (not the length of the set)
    sources: list[str]      # list of sources of the line in the dmesg log, used by FACT
    is_interruption: bool      # True if the burst is an interruption, False otherwise
    category: str           # Category of the interruption, if it is an interruption (Ref XID Catalog)
    sub_category: str       # Sub-category of the interruption, if it is an interruption (Ref XID Catalog)
    start_time: datetime    # Start time of the burst (from first xid in the interruption burst)
    end_time: datetime      # End time of the burst (from last xid in the interruption burst)
    vbios_version: str      # VBIOS version at the time of the burst (from first xid in the burst)
    driver_version: str     # Driver version at the time of the burst (from first xid in the burst)
    
    @classmethod
    def from_xid_list(cls, xid_list):
        """
        Create a XIDBurst object from a list of XIDEntry objects.
        
        Args:
            xid_list: List of XIDEntry objects
            
        Returns:
            XIDBurst: A new XIDBurst object, or None if the list is empty
        """
        if not xid_list:
            return None
        first_xid = xid_list[0]
        last_xid = xid_list[-1]
        return cls(
            xids=set(xid.name for xid in xid_list),
            xid_list=xid_list,
            decoded_xid_set=set(xid.decoded_xid_string for xid in xid_list if hasattr(xid, 'decoded_xid_string') and xid.decoded_xid_string),
            count=len(xid_list),
            sources=list(xid.source for xid in xid_list if xid.source is not None),
            is_interruption=False,
            category="",
            sub_category="",
            start_time=first_xid.timestamp,
            end_time=last_xid.timestamp,
            vbios_version=first_xid.vbios_version if first_xid.vbios_version else "",
            driver_version=first_xid.driver if first_xid.driver else ""
        )

@dataclass
class JobEntry:
    """Class to read in and manage job runtime data from CSV files"""
    name: str               # Name of the job
    node: str               # Node the job ran on
    job_type: str           # Type of job (e.g. "Megatron Single", "Megatron MultiNode", "HPC", etc.)
    start_time: datetime    # Start time of the job
    end_time: datetime      # End time of the job
    vbios_version: str      # VBIOS version of the node
    driver_version: str     # Driver version of the node

@dataclass
class InProductionTime:
    """Class to read in and manage nodes in production tenure data from CSV files"""
    node: str               # Node name
    joined_time: datetime   # Time the node joined the fleet
    left_time: datetime     # Time the node left the fleet

@dataclass
class PerDayNodeStat:
    """Class to hold per day stats for each node"""
    date: datetime   # Date of the stats
    part_of_fleet: bool = False  # True if the node was part of the fleet at the end of the day
    was_added_to_fleet_this_day: bool = False  # True if the node was added to the fleet on this day(any tenure with joined_time later than start of day)
    was_removed_from_fleet_this_day: bool = False  # True if the node was removed from the fleet on this day(any tenure with left_time before end of day)
    
    work_time: timedelta = field(default_factory=lambda: timedelta())  # Total work time of the node this day
    interruptions_count: int = 0                  # Number of interruptions seen in the node this day
    mtbi: timedelta = field(default_factory=lambda: timedelta())  # Mean Time Between Interruption of the node this day
    
    accumulated_work_time: timedelta = field(default_factory=lambda: timedelta())  # Accumulated work time of the node till this day
    accumulated_interruptions_count: int = 0  # Accumulated interruptions count of the node till this day
    accumulated_mtbi: timedelta = field(default_factory=lambda: timedelta())  # Accumulated MTBI of the node till this day

@dataclass
class PerDayFleetStat:
    """Class to hold per day stats for the fleet"""
    date: datetime   # Date of the stats
    nodes_added_to_fleet: int = 0  # Number of nodes added to the fleet on this day
    nodes_removed_from_fleet: int = 0  # Number of nodes removed from the fleet on this day
    nodes_in_production: int = 0  # Number of nodes in production on this day
    nodes_in_outcast: int = 0  # Number of nodes in outcast on this day
    fleet_mtbi: timedelta = field(default_factory=lambda: timedelta())  # Mean Time Between Interruption of the fleet this day
    
    accumulated_work_time: timedelta = field(default_factory=lambda: timedelta())  # Accumulated work time of the fleet till this day
    accumulated_interruptions_count: int = 0  # Accumulated interruptions count of the fleet till this day
    accumulated_mtbi: timedelta = field(default_factory=lambda: timedelta())  # Accumulated MTBI of the fleet till this day

@dataclass
class NodeEntry:
    """Class to read in and hold data for each node"""
    name: str                                     # Name of the node
    work_time: timedelta = field(default_factory=lambda: timedelta())  # Total work time of the node
    interruptions_count: int = 0                  # Number of interruptions seen in the node
    interruptions_in_production_count: int = 0           # Number of interruptions while node was in production
    mtbi: timedelta = field(default_factory=lambda: timedelta())  # Mean Time Between Interruption of the node
    in_production_times: list[InProductionTime] = field(default_factory=list)  # List of in production tenure entries for this node
    job_runtimes: list[JobEntry] = field(default_factory=list)    # List of Job entries for this node
    xid_entries: list[XIDEntry] = field(default_factory=list)     # List of XID entries for this node
    xid_bursts: list[XIDBurst] = field(default_factory=list)      # List of XID bursts for this node
    sw_upgrades_driver: list[tuple[datetime, str]] = field(default_factory=list)  # List of SW upgrades/downgrades (time and version upgraded to)
    fw_upgrades_vbios: list[tuple[datetime, str]] = field(default_factory=list)  # List of FW upgrades/downgrades (time and version upgraded to)
    vbios_hours: dict[str, float] = field(default_factory=dict)   # Hours spent on each VBIOS version
    daily_stats: dict[datetime, PerDayNodeStat] = field(default_factory=dict)  # Maps days to their stats
    
    def update_version_history(self, timestamp, vbios_version=None, driver_version=None):
        """Update version history when new information is detected"""
        if vbios_version and timestamp:
            # Always add the entry
            self.fw_upgrades_vbios.append((timestamp, vbios_version))
            
            # Sort by timestamp to maintain chronological order
            self.fw_upgrades_vbios.sort(key=lambda x: x[0])
            
            # Now consolidate any consecutive duplicates
            i = 0
            while i < len(self.fw_upgrades_vbios) - 1:
                if self.fw_upgrades_vbios[i][1] == self.fw_upgrades_vbios[i+1][1]:
                    # Keep the earlier timestamp
                    del self.fw_upgrades_vbios[i+1]
                else:
                    i += 1
        
        if driver_version and timestamp:
            # Always add the entry
            self.sw_upgrades_driver.append((timestamp, driver_version))
            
            # Sort by timestamp to maintain chronological order
            self.sw_upgrades_driver.sort(key=lambda x: x[0])
            
            # Now consolidate any consecutive duplicates
            i = 0
            while i < len(self.sw_upgrades_driver) - 1:
                if self.sw_upgrades_driver[i][1] == self.sw_upgrades_driver[i+1][1]:
                    # Keep the earlier timestamp
                    del self.sw_upgrades_driver[i+1]
                else:
                    i += 1
        
    def _in_production_at_time(self, timestamp):
        """
        Helper method to determine if the node was in production at a specific point in time
        
        Args:
            timestamp: The point in time to check
        
        Returns:
            bool: True if node was in production at the given time, False otherwise
        """
        if not self.in_production_times:
            # If no production times data available, assume always in production if we have XIDs or jobs
            return (len(self.xid_entries) > 0 or len(self.job_runtimes) > 0)
            
        for tenure in self.in_production_times:
            if not tenure.joined_time:
                continue
                
            tenure_end = tenure.left_time or ensure_timezone_local(datetime.now())
            
            # Check if timestamp falls within tenure period
            if tenure.joined_time <= timestamp <= tenure_end:
                return True
                
        return False
    
    def _count_interruptions_in_period(self, period_start=None, period_end=None):
        """
        Helper method to count interruptions within a specific time period
        
        Args:
            period_start: Start time of the period (default: None = all time)
            period_end: End time of the period (default: None = all time)
        
        Returns:
            int: Number of interruptions in the period
        """
        count = 0
        
        if not self.xid_bursts:
            return 0
            
        # Default period is all time
        if period_start is None:
            period_start = datetime.min.replace(tzinfo=timezone.utc)
        if period_end is None:
            period_end = ensure_timezone_local(datetime.now())
        
        # Count interruptions within the period
        for burst in self.xid_bursts:
            if not (burst and burst.is_interruption and burst.start_time):
                continue
                
            if period_start <= burst.start_time <= period_end:
                # Check if burst occurred during an active tenure (only if we have tenure data)
                if self.in_production_times:
                    # Check if burst occurred during an active period 
                    if self._in_production_at_time(burst.start_time):
                        count += 1
                else:
                    # No tenure data, count all interruptions in the period
                    count += 1
        
        return count
    
    def _calculate_work_time(self, period_start=None, period_end=None, days=0):
        """
        Helper method to calculate work time for a specific period or days
        
        Args:
            start_time: Start time (default: None = all time)
            end_time: End time (default: None = all time) 
            days: Number of days to use instead of time range (overrides start/end)
            
        Returns:
            timedelta: Total work time
        """
            
        # Default period is all time
        if period_start is None:
            period_start = datetime.min.replace(tzinfo=timezone.utc) # beginning of times as per Python
        if period_end is None:
            period_end = ensure_timezone_local(datetime.now()) # now
            
        work_time = timedelta() # initialize work_time
        
        # Only calculate work_time from days if explicitly provided and > 0
        if days > 0:
            work_time = timedelta(seconds=days * 24 * 3600)
        
        # Try to calculate from job runtimes first
        elif self.job_runtimes:
            for job in self.job_runtimes: # Add up all job runtimes that ran in the period
                # Check if job overlaps with the period
                if job.start_time <= period_end and job.end_time >= period_start:
                    # Calculate overlap
                    job_start = max(job.start_time, period_start)
                    job_end = min(job.end_time, period_end)
                    job_duration = (job_end - job_start).total_seconds()
                    if job_duration > 0:  # Avoid negative durations
                        work_time += timedelta(seconds=job_duration)
        
        # If no job data, fall back to tenure data
        elif self.in_production_times:
            for tenure in self.in_production_times: # Add up all time this node was in production in the period
                # Skip incomplete tenure data
                if not tenure.joined_time:
                    continue
                    
                # Get end time (either left time or current time)
                tenure_end = tenure.left_time or ensure_timezone_local(datetime.now())
                
                # Check if tenure overlaps with the period
                if tenure.joined_time <= period_end and tenure_end >= period_start:
                    # Calculate overlap
                    tenure_start = max(tenure.joined_time, period_start)
                    tenure_end = min(tenure_end, period_end)
                    tenure_duration = (tenure_end - tenure_start).total_seconds()
                    if tenure_duration > 0:  # Avoid negative durations
                        work_time += timedelta(seconds=tenure_duration)
                        
        return work_time
    
    def calculate_mtbi(self, days_used: int = 0):
        """
        Calculate Mean Time Between Interruptions for this node
        
        This calculates:
        1. The total work time based on job runtimes or membership tenures
        2. The MTBI based on work time and active period interruptions
        
        Args:
            days_used (int): Optional override for days used if no job/tenure data
            
        Returns:
            A tuple of (work_time, mtbi) both as timedelta objects
        """
        # Calculate total active work time
        self.work_time = self._calculate_work_time(days=days_used)
        
        # Count interruptions during active periods
        self.interruptions_in_production_count = self._count_interruptions_in_period()
        
        # If no interruptions were found in production periods, use total count
        if self.interruptions_in_production_count == 0:
            self.interruptions_in_production_count = self.interruptions_count
            
        # Calculate MTBI (using max to avoid division by zero)
        self.mtbi = timedelta(seconds=self.work_time.total_seconds() / max(self.interruptions_in_production_count, 1))
            
        return (self.work_time, self.mtbi)
    
    def calculate_vbios_hours(self, all_vbios_versions):
        """Calculate hours spent on each VBIOS version"""
        # Initialize hours for all versions including unknown
        self.vbios_hours = {vbios: 0.0 for vbios in list(all_vbios_versions) + ["unknown"]}
        
        # If we have no VBIOS data, default to unknown
        if not self.fw_upgrades_vbios:
            self.vbios_hours["unknown"] = self.work_time.total_seconds() / 3600 if self.work_time.total_seconds() > 0 else 24.0
            return self.vbios_hours
            
        # Add a final transition to "now" with the last known version to close the final interval
        now = ensure_timezone_local(datetime.now())
        last_known_version = self.fw_upgrades_vbios[-1][1]  # Last version from the last transition
        
        # Calculate hours for each transition segment
        for i in range(len(self.fw_upgrades_vbios)):
            current_ts, current_version = self.fw_upgrades_vbios[i]
            
            # Determine the next timestamp (next version transition or now)
            if i < len(self.fw_upgrades_vbios) - 1:
                next_ts = self.fw_upgrades_vbios[i + 1][0]
            else:
                next_ts = now
            
            # Calculate time between transitions
            segment_hours = (next_ts - current_ts).total_seconds() / 3600
            
            # Only add positive durations
            if segment_hours > 0:
                self.vbios_hours[current_version] += segment_hours
                
        return self.vbios_hours

    def derive_stats(self):
        """
        Derive daily statistics for this node and populate the daily_stats dictionary.
        
        This method:
        1. Identifies active days from job_runtimes or in_production_times
        2. Calculates work time per day
        3. Counts interruptions per day
        4. Calculates MTBI for each day
        5. Tracks when nodes are added to or removed from the fleet
        6. Calculates accumulated metrics across days
        
        Returns:
            Dictionary mapping datetime objects (days) to PerDayNodeStat objects
        """
        # Clear existing stats
        self.daily_stats = {}
        
        # Collect unique days from jobs or tenures
        all_days_this_node_was_active_on = set()
        if self.job_runtimes:
            for job in self.job_runtimes:
                current_day = job.start_time.replace(hour=0, minute=0, second=0, microsecond=0)
                end_day = job.end_time.replace(hour=0, minute=0, second=0, microsecond=0)
                while current_day <= end_day:
                    all_days_this_node_was_active_on.add(current_day)
                    current_day += timedelta(days=1)
        elif self.in_production_times:
            for tenure in self.in_production_times:
                if not tenure.joined_time:
                    continue
                current_day = tenure.joined_time.replace(hour=0, minute=0, second=0, microsecond=0)
                end_day = (tenure.left_time or ensure_timezone_local(datetime.now())).replace(hour=0, minute=0, second=0, microsecond=0)
                while current_day <= end_day:
                    all_days_this_node_was_active_on.add(current_day)
                    current_day += timedelta(days=1)
        
        # Prepare for accumulated stats
        accumulated_work_time = timedelta()
        accumulated_interruptions_count = 0

        # Process each day
        for day in sorted(all_days_this_node_was_active_on):
            stats = self.daily_stats[day] = PerDayNodeStat(date=day)
            day_end = min(day.replace(hour=23, minute=59, second=59, microsecond=999999), ensure_timezone_local(datetime.now())) # min used to avoid future dates
            
            stats.work_time = self._calculate_work_time(day, day_end)
            stats.interruptions_count = self._count_interruptions_in_period(day, day_end)
            stats.part_of_fleet = self._in_production_at_time(day_end)
            
            # Check fleet status changes
            for tenure in self.in_production_times:
                if not tenure.joined_time:
                    continue
                if tenure.joined_time.replace(hour=0, minute=0, second=0, microsecond=0) == day:
                    stats.was_added_to_fleet_this_day = True
                if tenure.left_time and tenure.left_time.replace(hour=0, minute=0, second=0, microsecond=0) == day:
                    stats.was_removed_from_fleet_this_day = True
            
            # Calculate daily MTBI
            stats.mtbi = timedelta(seconds=stats.work_time.total_seconds() / max(stats.interruptions_count, 1))
            
            # Update accumulated stats
            accumulated_work_time += stats.work_time
            accumulated_interruptions_count += stats.interruptions_count
            stats.accumulated_work_time = accumulated_work_time
            stats.accumulated_interruptions_count = accumulated_interruptions_count
            
            # Calculate accumulated MTBI (using max to avoid division by zero)
            stats.accumulated_mtbi = timedelta(seconds=accumulated_work_time.total_seconds() / max(accumulated_interruptions_count, 1))
        
        return

@dataclass
class FleetData:
    """Structured class for storing and processing XID Pattern Analysis results"""

    # Data read directly from the input files
    nodes: list[NodeEntry] = field(default_factory=list)                 # List of Node entries

    # Data extracted from the input data
    report_period_start: Optional[datetime] = None                      # Start of the report period (start of the first job)
    report_period_end: Optional[datetime] = None                        # End of the report period (end of the last job)
    driver_versions: set[str] = field(default_factory=set)              # Driver versions encountered in the report
    vbios_versions: set[str] = field(default_factory=set)               # VBIOS versions encountered in the report
    node_names: set[str] = field(default_factory=set)                   # Nodes encountered in the report
    job_types: set[str] = field(default_factory=set)                    # Job types encountered in the report
    interruption_categories: set[str] = field(default_factory=set)      # Interruption categories encountered in the report
    interruption_sub_categories: set[str] = field(default_factory=set)  # Interruption sub-categories encountered in the report
    
    # Data Calculated from above data
    total_work_time: timedelta = field(default=timedelta())             # Total work time of the fleet
    total_interruptions_count: int = 0                                  # Total interruptions count in the fleet
    total_interruptions_in_production_count: int = 0                           # Total active interruptions count in the fleet
    total_mtbi: timedelta = field(default=timedelta())                  # Total MTBI of the fleet
    total_bursts: int = 0                                               # Total number of bursts across all nodes 
    fleet_size: int = 0                                                 # Size of the fleet
    mtbi_years: float = 0.0                                             # MTBI in years
    daily_stats: dict[datetime, PerDayFleetStat] = field(default_factory=dict)  # Maps days to their stats
    
    # Internal data structures
    _unique_xid_entries: set = field(default_factory=set)               # For deduplication - tracks unique (machine, timestamp, number, context) tuples
    
    def get_category_counts(self):
        """Calculate category counts from xid_bursts"""
        counts = collections.defaultdict(int)
        for node in self.nodes:
            for burst in node.xid_bursts:
                if burst is not None and burst.is_interruption:
                    counts[(burst.category, burst.sub_category)] += 1
        return counts

    def _get_or_create_node(self, node_name):
        """Get an existing node or create a new one if it doesn't exist"""
        for node in self.nodes:
            if node.name == node_name:
                return node
        
        # Create new node if it doesn't exist
        new_node = NodeEntry(name=node_name)
        self.nodes.append(new_node)
        return new_node

    def export_to_csv(self, output_file='xid_analysis_results.csv'):
        """
        Export all XID entries to a CSV file.
        
        Args:
            output_file (str): Path to the output CSV file (default: xid_analysis_results.csv)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Define CSV headers based on XIDEntry fields
            headers = ['Timestamp', 'XID Number', 'Mnemonic', 'SW Ver(Driver)', 'FW Ver(VBIOS)', 
                      'Machine Id', 'XID Message', 'Resolution', 'Investigatory/Deferred Action', 
                      'Job Failure Severity', 'HW/SW Classification', 'Comments']
            
            # Collect all XID entries from all nodes
            all_xid_entries = []
            for node in self.nodes:
                all_xid_entries.extend(node.xid_entries)
            
            # Sort entries by timestamp
            all_xid_entries.sort(key=lambda x: x.timestamp if x.timestamp else datetime.min)
            
            logging.info(f"Exporting {len(all_xid_entries)} XID entries to {output_file}")
            
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                
                for entry in all_xid_entries:
                    # Format values for CSV
                    timestamp = entry.timestamp.isoformat() if entry.timestamp else ""
                    number = str(entry.number) if entry.number else ""
                    mnemonic = entry.mnemonic if entry.mnemonic else ""
                    driver = entry.driver if entry.driver else ""
                    vbios = entry.vbios_version if entry.vbios_version else ""
                    machine = entry.machine if entry.machine else ""
                    context = entry.context if entry.context else ""
                    resolution = entry.resolution if entry.resolution else ""
                    investigatory = entry.investigatory_action if entry.investigatory_action else ""
                    job_failure_severity = entry.job_failure_severity if entry.job_failure_severity else ""
                    hw_sw_classification = entry.hw_sw_classification if entry.hw_sw_classification else ""
                    comments = entry.comments if entry.comments else ""
                    
                    writer.writerow([timestamp, number, mnemonic, driver, vbios, machine, context, 
                                    resolution, investigatory, job_failure_severity, hw_sw_classification, comments])
            
            logging.info(f"Successfully exported XID data to {output_file}")
            return True
            
        except Exception as e:
            logging.error(f"Error exporting XID data to CSV: {e}")
            return False

    def read_xid_data(self, file_path, drop_duplicates=False):
        """
        Read XID data from a file (step 5 in expected flow)
        
        Args:
            file_path: Path to the file containing XID data
            drop_duplicates: Whether to eliminate duplicate XID entries
        """
        if not file_path:
            return False

        try:
            # Convert pathlib.Path to string if needed
            file_path_str = str(file_path)
            
            # Determine file type based on extension
            is_csv = file_path_str.lower().endswith('.csv')
            
            # If it's a CSV file, process it as CSV
            if is_csv:
                self._read_xids_from_csv_file(file_path_str, drop_duplicates)
            else:
                # For non-CSV files, process as a log file with XID entries
                self._read_xids_from_log_file(file_path_str, drop_duplicates)
            
            return True
                
        except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
            logging.error(f"Could not open observed XIDs data file: {file_path}")
            return False
        except Exception as e:
            return False

    def _read_xids_from_csv_file(self, csv_file, drop_duplicates):
        """Process a CSV file containing XID data."""
        try:
            with open(csv_file, 'r', newline='') as f:
                reader = csv.reader(f)

                global xid_tsdb_header
                # xid_tsdb_header: ['id', 'time', 'Timestamp', 'XID Number', 'XID Message', 'Machine Id', 'SW Ver(Driver)', 'FW Ver(VBIOS)', 'job_name_2']
                xid_tsdb_header = next(reader)

                for row in reader:
                    try:
                        row_as_kv = row_to_kv_pairs(xid_tsdb_header, row)
                        
                        logging.debug(f'row_as_kv: {row_as_kv}')
                        xid_seen = XIDEntry.from_message(timestamp=row_as_kv["Timestamp"],
                                                        message=row_as_kv["XID Message"],
                                                        driver=row_as_kv.get("SW Ver(Driver)", None),
                                                        machine=row_as_kv["Machine Id"],
                                                        vbios_version=row_as_kv.get("FW Ver(VBIOS)", None))
                        
                        self._add_xid_entry(xid_seen, drop_duplicates)
                    except KeyError as e:
                        missing_col = str(e).strip("'\"")
                        logging.error(f"Missing required column '{missing_col}' in CSV file '{csv_file}'.")
                        raise ValueError(f"Missing required column '{missing_col}' in CSV file")
                        
        except Exception as e:
            raise

    def _read_xids_from_log_file(self, log_file, drop_duplicates):
        """Process any log file (dmesg or redfish) containing XID messages."""
        logging.debug(f"Processing log file: {log_file}")
        driver_version = None
        vbios_version = None
        machine = "local"  # Default machine name for logs
        
        with open(log_file, 'r') as f:
            for line in f:
                # Look for NVIDIA driver version
                if "loading NVIDIA UNIX" in line:
                    # Try to match x86_64 format first
                    version_match = re.search(r'loading NVIDIA UNIX x86_64 Kernel Module\s+(\d+\.\d+\.\d+)', line)
                    if not version_match:
                        # Try to match aarch64 format
                        version_match = re.search(r'loading NVIDIA UNIX Open Kernel Module for aarch64\s+(\d+\.\d+)', line)
                    
                    if version_match:
                        driver_version = version_match.group(1)
                        self.driver_versions.add(driver_version)
                        logging.debug(f"Found driver version: {driver_version}")
                
                # Look for VBIOS version
                if "NVRM: NVIDIA VBIOS Image" in line:
                    vbios_match = re.search(r'NVRM: NVIDIA VBIOS Image Version:\s*(\d+\.\d+\.\d+\.\d+)', line)
                    if vbios_match:
                        vbios_version = vbios_match.group(1)
                        self.vbios_versions.add(vbios_version)
                        logging.debug(f"Found VBIOS version: {vbios_version}")
                
                # Try to extract XID from any line - let from_message handle pattern detection
                if "XID" in line.upper():
                    logging.debug(f"Found potential XID line: {line}")
                    try:
                        # For redfish format, use redfish as machine name
                        current_machine = "unknown" if "Driver Event Message" in line else machine
                        
                        # Create XID entry from the message
                        xid_seen = XIDEntry.from_message(
                            message=line,
                            driver=driver_version or "unknown",
                            machine=current_machine,
                            vbios_version=vbios_version
                        )
                        
                        if xid_seen:
                            self._add_xid_entry(xid_seen, drop_duplicates)
                    except Exception as e:
                        logging.debug(f"Error processing line: {e}")
                        continue

    def _add_xid_entry(self, xid_entry, drop_duplicates):
        """Add a XIDEntry to the fleet data, avoiding duplicates"""
        if not xid_entry:
            return

        # Check for duplicates if needed
        if drop_duplicates:
            # Include number and context in the duplicate detection key to make it more specific
            key = (xid_entry.machine, xid_entry.timestamp, xid_entry.number, xid_entry.context)
            if key in self._unique_xid_entries:
                return
            self._unique_xid_entries.add(key)
        
        # Add entry to the appropriate node
        node = self._get_or_create_node(xid_entry.machine)
        node.xid_entries.append(xid_entry)
        
        # If XID doesn't have VBIOS version, try to look it up from node's version history
        if not xid_entry.vbios_version and xid_entry.timestamp:
            xid_entry.vbios_version = self._lookup_version_for_timestamp(node, xid_entry.timestamp, 'vbios')
        
        # If XID doesn't have driver version, try to look it up from node's version history
        if (not xid_entry.driver or xid_entry.driver == "unknown" or (xid_entry.driver and xid_entry.driver.strip() == "")) and xid_entry.timestamp:
            xid_entry.driver = self._lookup_version_for_timestamp(node, xid_entry.timestamp, 'driver')
        
        # Update version history with XID data (after potential lookups)
        node.update_version_history(xid_entry.timestamp, xid_entry.vbios_version, xid_entry.driver)
        
        # Update metadata
        if xid_entry.driver:
            self.driver_versions.add(xid_entry.driver)
        if xid_entry.vbios_version:
            self.vbios_versions.add(xid_entry.vbios_version)
        if xid_entry.machine:
            self.node_names.add(xid_entry.machine)
    
    def _lookup_version_for_timestamp(self, node, timestamp, version_type):
        """
        Look up the version (VBIOS or driver) that was active at a specific timestamp for a node.
        
        Args:
            node: NodeEntry object
            timestamp: datetime object to look up
            version_type: str, either 'vbios' or 'driver'
            
        Returns:
            str: Version if found, None otherwise
        """
        if not timestamp:
            return None
            
        # Select the appropriate version history based on type
        if version_type == 'vbios':
            version_history = node.fw_upgrades_vbios
        elif version_type == 'driver':
            version_history = node.sw_upgrades_driver
        else:
            raise ValueError(f"Invalid version_type: {version_type}. Must be 'vbios' or 'driver'")
            
        if not version_history:
            return None
            
        # Find the most recent version before or at the given timestamp
        active_version = None
        for version_timestamp, version in version_history:
            if version_timestamp <= timestamp:
                active_version = version
            else:
                break  # Since the list is sorted, we can stop here
                
        return active_version
    
    def _lookup_vbios_version_for_timestamp(self, node, timestamp):
        """Convenience method for VBIOS version lookup."""
        return self._lookup_version_for_timestamp(node, timestamp, 'vbios')
    
    def _lookup_driver_version_for_timestamp(self, node, timestamp):
        """Convenience method for driver version lookup."""
        return self._lookup_version_for_timestamp(node, timestamp, 'driver')
    
    def tag_xids(self):
        """
        Tag XIDs based on the watchlist (step 7 in expected flow)
        """
        global XID_WATCHLIST
        # Process XIDs in all nodes
        for node in self.nodes:
            for xid_entry in node.xid_entries:
                if xid_entry.name != '':  # Skip already tagged XIDs
                    continue
                    
                for xid_wanted in XID_WATCHLIST:
                    #NOTE: uses guilty until proven innocent approach
                    if xid_wanted.context and xid_wanted.context not in xid_entry.context:
                        continue
                    if xid_wanted.driver and xid_entry.driver and xid_entry.driver not in xid_wanted.driver:
                        continue
                    if xid_wanted.subcode and xid_entry.subcode and xid_entry.subcode not in xid_wanted.subcode:
                        continue
                    if xid_wanted.errorstatus and xid_entry.errorstatus and str(xid_entry.errorstatus) not in xid_wanted.errorstatus:
                        continue
                    if xid_wanted.intrinfo and xid_entry.intrinfo:
                        match_found = False
                        for pattern in xid_wanted.intrinfo.split(';'):
                            if does_xid_intrinfo_match_rule(pattern.strip(), xid_entry.intrinfo):
                                match_found = True
                                break
                        if not match_found:
                            continue
                    if xid_entry.number == xid_wanted.number:
                        xid_entry.name = xid_wanted.name  # xid_entry has matched all characteristics so give it the same name
                        break

                if xid_entry.name == '':  # xid_entry has not matched any of the watchlist XIDs so use the number as the name
                    xid_entry.name = str(xid_entry.number)
    
    def find_resolutions(self):
        """
        Find resolutions for XIDs (step 6 in expected flow)
        """
        # Process XIDs in all nodes
        for node in self.nodes:
            for xid_entry in node.xid_entries:
                xid_entry.lookup_resolution()
    
    def get_xids_for_node(self, node_name):
        """Get all XIDs for a specific node"""
        for node in self.nodes:
            if node.name == node_name:
                return node.xid_entries
        return []
    
    def get_node_list(self):
        """Get list of all node names"""
        return list(self.node_names)
    
    def group_xids_into_bursts(self):
        """
        Group XIDs into bursts (step 8 in expected flow)
        
        Uses the global STICKY_XIDS variable directly
        """
        # Group XIDs by node
        for node in self.nodes:
            # Get bursts for this node
            node_bursts = SplitListToBursts(node.xid_entries)
            # Store bursts in the node
            node.xid_bursts = node_bursts
    
    def match_bursts_to_patterns(self):
        """
        Match bursts to patterns (step 9 in expected flow)

        Uses the global XID_POLICY variable directly
        """
        global XID_POLICY
        self.total_interruptions_count = 0
        
        for node in self.nodes:
            for burst in node.xid_bursts:
                if burst is None:
                    continue
                    
                # Check if the burst matches any pattern
                for pattern in XID_POLICY.XIDPolicyEntries:
                    if pattern.Matches(burst):
                        burst.category = pattern.CATEGORY
                        burst.sub_category = pattern.SUBCATEGORY
                        # Add to sets for reporting
                        self.interruption_categories.add(pattern.CATEGORY)
                        self.interruption_sub_categories.add(pattern.SUBCATEGORY)
                        if "Yes" in pattern.JOBDISRUPTION:
                            self.total_interruptions_count += 1
                            node.interruptions_count += 1  # Count at node level too
                            # Update the burst object with interruption info
                            burst.is_interruption = True
                        break
    
    def calculate_mtbi(self, fleet_size=0, days_used=0):
        """Calculate fleet-level MTBI metrics"""
        # Calculate node-level metrics
        for node in self.nodes:
            node.calculate_mtbi(days_used)
        
        # Calculate fleet-level work time
        if fleet_size > 0 and days_used > 0:
            # Use fleet size and days used if provided
            self.total_work_time = timedelta(seconds=fleet_size * days_used * 24 * 3600)
        else:
            # Otherwise sum up work time from all nodes
            total_seconds = sum(node.work_time.total_seconds() for node in self.nodes)
            self.total_work_time = timedelta(seconds=total_seconds)
        
        # Get in-production interruptions count
        self.total_interruptions_in_production_count = sum(node.interruptions_in_production_count for node in self.nodes)
        
        # Set fleet size based on nodes active at report period end
        if fleet_size <= 0:
            reference_day = (self.report_period_end or ensure_timezone_local(datetime.now())).replace(hour=0, minute=0, second=0, microsecond=0)
            active_nodes = sum(1 for node in self.nodes if reference_day in node.daily_stats and node.daily_stats[reference_day].part_of_fleet)
            self.fleet_size = active_nodes if active_nodes > 0 else 0
            
            # Log the calculated fleet size for debugging
            logging.info(f"Calculated fleet size: {self.fleet_size} active nodes as on {reference_day}")
        else:
            self.fleet_size = fleet_size
                    
        # Calculate total bursts
        self.total_bursts = sum(len(node.xid_bursts) for node in self.nodes)
        
        # Calculate fleet-level MTBI
        if self.total_work_time.total_seconds() > 0 and self.total_interruptions_in_production_count > 0:
            self.total_mtbi = timedelta(seconds=self.total_work_time.total_seconds() / self.total_interruptions_in_production_count)
            self.mtbi_years = self.total_mtbi.total_seconds() / (365 * 24 * 3600)
        else:
            self.mtbi_years = float('inf')
            self.total_mtbi = INFINITE_MTBI
    
    def calculate_vbios_hours(self):
        """
        Calculate hours spent on each VBIOS version for all nodes in the fleet
        """
        for node in self.nodes:
            node.calculate_vbios_hours(self.vbios_versions)

    def derive_stats(self):
        """
        Derive daily statistics for the fleet based on node-level daily stats.
        
        This method:
        1. Clears existing fleet-level daily stats
        2. Collects all unique dates from all nodes' daily stats
        3. For each date, counts nodes in production and outcast
        4. Tracks nodes added to or removed from the fleet
        5. Calculates fleet-level MTBI for each day
        6. Calculates accumulated metrics across days
        
        Returns:
            Dictionary mapping dates to PerDayFleetStat objects
        """
        # Clear existing stats and ensure all nodes have stats calculated
        self.daily_stats = {}
        for node in self.nodes:
            node.derive_stats()
        
        # Get all unique dates from nodes
        all_dates = set()
        for node in self.nodes:
            all_dates.update(node.daily_stats.keys())
        
        logging.info(f"FleetData.derive_stats: Found {len(all_dates)} unique dates for daily stats")
        
        # Process dates chronologically
        accumulated_work_time = timedelta()
        accumulated_interruptions_count = 0
        
        for date in sorted(all_dates):
            fleet_stat = PerDayFleetStat(date=date)
            total_work_seconds = total_interruptions = 0
            
            # Aggregate node stats for this date
            for node in self.nodes:
                if date in node.daily_stats:
                    node_stat = node.daily_stats[date]
                    
                    fleet_stat.nodes_in_production += node_stat.part_of_fleet
                    fleet_stat.nodes_in_outcast += not node_stat.part_of_fleet
                    fleet_stat.nodes_added_to_fleet += node_stat.was_added_to_fleet_this_day
                    fleet_stat.nodes_removed_from_fleet += node_stat.was_removed_from_fleet_this_day
                    
                    total_work_seconds += node_stat.work_time.total_seconds()
                    total_interruptions += node_stat.interruptions_count
            
            # Calculate daily and accumulated metrics
            fleet_stat.fleet_mtbi = timedelta(seconds=total_work_seconds / max(total_interruptions, 1))
            
            accumulated_work_time += timedelta(seconds=total_work_seconds)
            accumulated_interruptions_count += total_interruptions
            
            fleet_stat.accumulated_work_time = accumulated_work_time
            fleet_stat.accumulated_interruptions_count = accumulated_interruptions_count
            fleet_stat.accumulated_mtbi = timedelta(seconds=accumulated_work_time.total_seconds() / max(accumulated_interruptions_count, 1))
            
            self.daily_stats[date] = fleet_stat
            
        logging.info(f"FleetData.derive_stats: Generated {len(self.daily_stats)} daily fleet stat entries")

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
            
        #sanitize the row
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
        
        # Convert string XID lists to sets
        self.SET_XIDS_MUST_EQUAL = policy_xidlist_2_set(self.XIDS_MUST_EQUAL)
        self.SET_XIDS_MUST_INTERSECT = policy_xidlist_2_set(self.XIDS_MUST_INTERSECT)
        self.SET_XIDS_MUST_NOT_INTERSECT = policy_xidlist_2_set(self.XIDS_MUST_NOT_INTERSECT)
        self.SET_XIDS_MAY_INTERSECT = policy_xidlist_2_set(self.XIDS_MAY_INTERSECT)
        
    def Matches(self, observed_burst: XIDBurst):
        # Check if policy applies to the given set of XIDs
        applies = True
        
        # All of these XIDs must be present
        if self.SET_XIDS_MUST_EQUAL and not self.SET_XIDS_MUST_EQUAL.issubset(observed_burst.xids):
            applies = False

        # At least one of these XIDs must be present
        if self.SET_XIDS_MUST_INTERSECT and not self.SET_XIDS_MUST_INTERSECT.intersection(observed_burst.xids):
            applies = False
        
        # None of these XIDs should be present (except for the optionally present ones)
        if self.SET_XIDS_MUST_NOT_INTERSECT:
            if self.SET_XIDS_MUST_NOT_INTERSECT == {"*"}:
                # Special case: no unexpected XIDs allowed- nothing must remain except expected ones
                if observed_burst.xids - self.SET_XIDS_MUST_INTERSECT - self.SET_XIDS_MUST_EQUAL - self.SET_XIDS_MAY_INTERSECT:
                    applies = False
            elif self.SET_XIDS_MUST_NOT_INTERSECT.intersection(observed_burst.xids - self.SET_XIDS_MAY_INTERSECT):
                applies = False
        
        return applies

@dataclass
class XidParsingPolicy:
    XIDPolicyEntries: list[XIDBurstPattern]

    # Load XID parsing policy from the embedded catalog data.
    def __init__(self):
        self.XIDPolicyEntries = []
        try:
            policy = load_xid_catalog_data()['xid_parsing_policy']

            # Set the module-level header used by ContentOfColumn lookups.
            global xid_policy_header
            xid_policy_header = policy['header']

            # Rebuild a burst pattern from each stored policy row.
            for row_list in policy['rows']:
                self.XIDPolicyEntries.append(XIDBurstPattern(row_list))

            # Sort entries by priority (convert to int for proper numerical sorting)
            self.XIDPolicyEntries.sort(key=lambda x: int(x.PRIORITY) if x.PRIORITY.isdigit() else float('inf'))

            # Find the final list of XIDs we need to watch for
            global XID_WATCHLIST
            # Use dict comprehension to deduplicate by name while keeping last occurrence
            XID_WATCHLIST = sorted({x.name: x for x in XID_WATCHLIST}.values(),
                                 key=lambda x: len(x.name), reverse=True)

        except Exception as e:  # Catch any exception that might occur
            logging.error(f"Error loading XID parsing policy: {e}")

    def PrintPolicy(self):
        for policy in self.XIDPolicyEntries:
            logging.info(f'{policy.PRIORITY} {policy.CATEGORY} {policy.SUBCATEGORY} {policy.XIDS_MUST_EQUAL} {policy.XIDS_MUST_INTERSECT} {policy.XIDS_MUST_NOT_INTERSECT}')
            pass

def SplitListToBursts(xids_list):
    """
    Groups XIDs into bursts using the global STICKY_XIDS variable.
    Args:
        xids_list: List of XIDEntry objects to group into bursts
        
    Returns:
        List of XIDBurst objects representing bursts of XIDs
    """
    global STICKY_XIDS
    bursts = []
    xids_list.sort(key=lambda x: x.timestamp)

    past_ts = ensure_timezone_local(datetime.now() - timedelta(weeks=1000))
    future_ts = ensure_timezone_local(datetime.now() + timedelta(weeks=100))
    last_ts = future_ts
    last_driver = ""
    
    # sticky_xids = ["119", "109", "95", "95[ LTC TAG ]", "79", "74"]
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

        if xid.driver != last_driver: # override new_burst to True if driver version differs
            new_burst = True
        
        if new_burst:
            if burst_to_add:
                # Use XIDBurst class method to create a burst
                xid_burst = XIDBurst.from_xid_list(burst_to_add)
                bursts.append(xid_burst)
                burst_to_add = []

        burst_to_add.append(xid)
        last_ts = xid.timestamp
        last_driver = xid.driver
        if xid.name in STICKY_XIDS:
            last_sticky_ts[xid.name] = last_ts
    
    # Handle the last burst
    if burst_to_add:
        xid_burst = XIDBurst.from_xid_list(burst_to_add)
        bursts.append(xid_burst)

    return bursts

# Load XID 144-150 decoding rules from the embedded catalog data.
def load_nvl5_xids_decoding_logic():
    """
    Load XID patterns from the 144-150 decode data.
    Returns:
        dict: Dictionary with XID numbers as keys and lists of pattern dictionaries as values
    """
    try:
        rules = load_xid_catalog_data()['nvl5_xid_decoding_rules']
        return {
            int(xid): [{key: _restore_missing(value) for key, value in pattern.items()} for pattern in patterns]
            for xid, patterns in rules.items()
        }
    except Exception as e:
        logging.error(f"Error loading XID 144-150 decode rules: {e}")
        return {}

# Load XID and SXID listing from the embedded catalog data.
def load_xid_sheet():
    """
    Load XID/SXID information from the embedded catalog data.

    Returns:
        dict: Dictionary with XID/SXID codes as keys and attribute dictionaries
        as values. Missing cells are restored to float('nan') to preserve the
        historical workbook-backed behavior.
    """
    try:
        catalog = load_xid_catalog_data()['xid_catalog']
        return {
            code: {key: _restore_missing(value) for key, value in entry.items()}
            for code, entry in catalog.items()
        }
    except Exception as e:
        logging.error(f"Error loading XID catalog: {e}")
        return {}

def get_resolution_for_xid154(input_string):
    """
    Function to extract the string data from the last set of parentheses
    after finding the pattern "to 0x" followed by any characters and a space.
    
    Args:
        input_string (str): The input string to process
        
    Returns:
        str: The extracted text from the last set of parentheses, or None if not found
    """
    # First check if the pattern "to 0x" followed by characters and space exists
    pattern = r"to 0x[0-9a-fA-F]+ "
    if not re.search(pattern, input_string):
        return None
        
    # Find the last occurrence of opening and closing parentheses
    last_open = input_string.rfind("(")
    last_close = input_string.rfind(")")
    
    # Check if parentheses exist and are properly ordered
    if last_open != -1 and last_close != -1 and last_open < last_close:
        # Extract the text between the last set of parentheses
        extracted_text = input_string[last_open + 1:last_close]
        return extracted_text
    
    return None

def does_xid_intrinfo_match_rule(intrinfo_in_rule, intrinfo_in_message):

    logging.debug(f"intrinfo_in_rule: {intrinfo_in_rule}")
    logging.debug(f"intrinfo_in_message: {intrinfo_in_message}")

    # Convert binary strings to integers for bitwise comparison
    # hexadecimal to binary for xid
    csv_intrinfo_mask_str = intrinfo_in_rule.replace('0', '1')
    csv_intrinfo_mask = int(csv_intrinfo_mask_str.replace('-', '0'), 2)
    csv_intrinfo_value = int(intrinfo_in_rule.replace('-', '1'), 2)
    
    # Calculate LHS and RHS for the bitwise comparison
    lhs = intrinfo_in_message & csv_intrinfo_mask
    rhs = csv_intrinfo_value & csv_intrinfo_mask
    return lhs == rhs

def setup_argument_parser():
    """
    Set up command line argument parser with all available options.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='NVIDIA XID Pattern Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input source options
    parser.add_argument(
        '--calculate-mtbi', 
        type=pathlib.Path,
        help="CSV file with list of XIDs observed on the fleet"
    )
    
    parser.add_argument(
        '--find-resolutions', 
        type=pathlib.Path,
        help="Log file (e.g., dmesg.log or redfish events log)"
    )
    
    parser.add_argument(
        '--decode-xid-message', '--xid-message', 
        dest='xid_message',
        type=str, 
        help="Single XID message to decode"
    )
    
    # Add drop duplicates option
    parser.add_argument(
        '--drop-duplicates',
        action='store_true',
        help="Remove duplicate XID entries with the same timestamp, machine ID, XID number, and context. Might be needed if logging infra creates duplicate entries."
    )
    
    # Add chart generation option
    parser.add_argument(
        '--generate-charts',
        action='store_true',
        help="Generate charts and HTML report using visualizer.py functionality"
    )
    
    # Add CSV export option
    parser.add_argument(
        '--export-csvs',
        action='store_true',
        help="Export analysis results to CSV files (xid_bursts_analysis.csv, node_mtbi.csv, xid_list_analysis.csv) without generating charts"
    )
    
    parser.add_argument(
        '--output-dir', 
        type=pathlib.Path,
        default='.',
        help="Directory to save output files when using --generate-charts or --export-csvs (default: current directory)"
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default='xid_analysis_report.html',
        help="Name of the output HTML file when using --generate-charts (default: xid_analysis_report.html)"
    )
    
    parser.add_argument(
        '--job-runtimes',
        type=pathlib.Path,
        help="CSV file containing job runtime data for chart generation"
    )
    
    parser.add_argument(
        '--inventory-data',
        type=pathlib.Path,
        help="CSV file containing node inventory data (join/retire timestamps)"
    )
    
    # Fleet analysis options
    parser.add_argument(
        '--fleet-size', 
        type=int, 
        default=0,
        help="Size of the fleet as number of nodes (default: auto-detect from job data or use 1000)"
    )
    
    parser.add_argument(
        '--days-used', 
        type=int, 
        default=0,
        help="Number of days this data represents (default: auto-detect from job data)"
    )
    
    # Logging options
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help="Set the logging level (default: INFO)"
    )
    
    return parser

def setup_rules():
    """
    Set up the rules for the XID Pattern Analyzer.
    """
    global XID_CATALOG, NVL5_XID_DECODING_RULES, STICKY_XIDS, XID_POLICY
    
    # Preload the embedded catalog data
    load_xid_catalog_data()
    
    # Load the XIDs sheet from the Excel file
    XID_CATALOG = load_xid_sheet()
    NVL5_XID_DECODING_RULES = load_nvl5_xids_decoding_logic()

    STICKY_XIDS = []
    for xid, entry in XID_CATALOG.items():
        if entry['sticky'] == "YES":
            STICKY_XIDS.append(xid)

    XID_POLICY = XidParsingPolicy()

def ensure_timezone_local(dt):
    """
    Ensure a datetime has timezone info using the system's local timezone.
    If it doesn't have timezone info, add local timezone.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        # Add timezone info to naive datetimes
        try:
            import tzlocal
            local_timezone = tzlocal.get_localzone()
            return dt.replace(tzinfo=local_timezone)
        except ImportError:
            # Fallback if tzlocal is not available
            try:
                return dt.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
            except Exception:
                # If ZoneInfo fails, try a simpler approach
                import pytz
                return dt.replace(tzinfo=pytz.timezone("America/Los_Angeles"))
    return dt

def export_xid_bursts_analysis_to_csv(fleet_data, output_file='xid_bursts.csv'):
    """
    Export all XID bursts to a CSV file.
    
    Args:
        fleet_data (FleetData): The fleet data containing all nodes and their bursts
        output_file (str): Path to the output CSV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Define CSV headers
        headers = ['Timestamp', 'Machine Id', 'XID Set', 'Decoded XIDs Set', 'XIDs in Burst', 'Interruption Category', 'Interruption Subcategory', 'FW Ver(VBIOS)', 'SW Ver(Driver)']
        
        # Collect all bursts from all nodes
        all_bursts = []
        for node in fleet_data.nodes:
            for burst in node.xid_bursts:
                if burst is not None:  # Check if burst is not None
                    # Convert xids set to a comma-separated string
                    xids_str = ', '.join(sorted(burst.xids)) if burst.xids else ""
                    
                    # Convert decoded XIDs set to a comma-separated string
                    decoded_xids_str = ', '.join(sorted(burst.decoded_xid_set)) if burst.decoded_xid_set else ""
                    
                    all_bursts.append({
                        'Timestamp': burst.start_time.isoformat() if burst.start_time else "",
                        'Machine Id': node.name,
                        'XID Set': xids_str,
                        'Decoded XIDs Set': decoded_xids_str,
                        'XIDs in Burst': burst.count if hasattr(burst, 'count') and burst.count is not None else 0,
                        'Interruption Category': burst.category if burst.category else "Unknown",
                        'Interruption Subcategory': burst.sub_category if burst.sub_category else "Unknown",
                        'FW Ver(VBIOS)': burst.vbios_version if burst.vbios_version else "Unknown",
                        'SW Ver(Driver)': burst.driver_version if burst.driver_version else "Unknown"
                    })
        
        # Sort bursts by timestamp
        all_bursts.sort(key=lambda x: x['Timestamp'])
        
        logging.info(f"Exporting {len(all_bursts)} burst entries to {output_file}")
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_bursts)
        
        logging.info(f"Successfully exported burst data to {output_file}")
        return True
        
    except Exception as e:
        logging.error(f"Error exporting burst data to CSV: {e}")
        return False

def export_node_mtbi_to_csv(fleet_data, output_file='node_mtbi.csv'):
    """
    Export node-level MTBI data to a CSV file.
    
    Args:
        fleet_data (FleetData): The fleet data containing all nodes and their MTBI metrics
        output_file (str): Path to the output CSV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Define CSV headers
        headers = ['Machine Id', 'Work Time Hrs', 'Interruptions', 'Interruptions in Production', 'MTBI Hrs']
        
        # Collect MTBI data from all nodes
        node_mtbi_data = []
        for node in fleet_data.nodes:
            # Convert work time to hours for readability
            work_time_hours = node.work_time.total_seconds() / 3600 if node.work_time else 0
            mtbi_hours = node.mtbi.total_seconds() / 3600 if node.mtbi else 0
            
            node_mtbi_data.append({
                'Machine Id': node.name,
                'Work Time Hrs': f"{work_time_hours:.2f}",
                'Interruptions': node.interruptions_count,
                'Interruptions in Production': node.interruptions_in_production_count,
                'MTBI Hrs': f"{mtbi_hours:.2f}"
            })
        
        # Sort nodes by machine ID for consistent output
        node_mtbi_data.sort(key=lambda x: x['Machine Id'])
        
        logging.info(f"Exporting MTBI data for {len(node_mtbi_data)} nodes to {output_file}")
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(node_mtbi_data)
        
        logging.info(f"Successfully exported node MTBI data to {output_file}")
        return True
        
    except Exception as e:
        logging.error(f"Error exporting node MTBI data to CSV: {e}")
        return False

def export_xid_list_analysis_to_csv(fleet_data, output_file='xid_entries.csv'):
    """
    Export all individual XID entries to a CSV file.
    
    Args:
        fleet_data (FleetData): The fleet data containing all nodes and their XIDs
        output_file (str): Path to the output CSV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Define CSV headers - including the most important XID fields
        headers = [
            'Timestamp', 'Machine Id', 'XID Number', 'Mnemonic', 'XID Name', 'XID Message', 
            'SW Ver(Driver)', 'FW Ver(VBIOS)', 'PCIe BDF', 'Resolution', 'Decoded XID'
        ]
        
        # Collect all XIDs from all nodes
        all_xids = []
        for node in fleet_data.nodes:
            for xid in node.xid_entries:
                if xid is not None:  # Check if XID is not None
                    all_xids.append({
                        'Timestamp': xid.timestamp.isoformat() if xid.timestamp else "",
                        'Machine Id': xid.machine if xid.machine else node.name,
                        'XID Number': xid.number,
                        'Mnemonic': xid.mnemonic if xid.mnemonic else "",
                        'XID Name': xid.name if xid.name else "",
                        'XID Message': xid.context if xid.context else "",
                        'SW Ver(Driver)': xid.driver if xid.driver else "Unknown",
                        'FW Ver(VBIOS)': xid.vbios_version if xid.vbios_version else "Unknown",
                        'PCIe BDF': xid.pcie_bdf if xid.pcie_bdf else "",
                        'Resolution': xid.resolution if xid.resolution else "",
                        'Decoded XID': f"{xid.decoded_xid_string}" if xid.decoded_xid_string else str(xid.number)
                    })
        
        # Sort XIDs by timestamp
        all_xids.sort(key=lambda x: x['Timestamp'])
        
        logging.info(f"Exporting {len(all_xids)} XID entries to {output_file}")
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_xids)
        
        logging.info(f"Successfully exported XID data to {output_file}")
        return True
        
    except Exception as e:
        logging.error(f"Error exporting XID data to CSV: {e}")
        return False

def main():
    """
    Main entry point for the XID Pattern Analyzer.
    
    Expected Flow:
    1. Read in XID Catalog main sheet
    2. Read in XID Catalog parsing policy sheet
    3. Read in NVL5 XID decoding sheet
    4. Read in Job Runtime Data if provided
    5. Read in XID Data if provided
    6. Find resolution for each XID
    7. Tag XIDs
    8. Group XIDs into bursts
    9. Compare each burst to burst patterns
    10. Calculate MTBF
    11. Generate console report
    12. Generate HTML report
    """
    # Parse command line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Set up logging level based on command-line argument
    log_level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(log_level)
    
    # Steps 1-3: Read in XID Catalog and rules
    logging.info("Loading XID catalog and rules...")
    setup_rules()
    
    # Create the FleetData object for analysis
    fleet_data = FleetData()
    
    # Single XID message analysis has different flow
    if args.xid_message:
        
        logging.info("Processing single XID message...")
        xid_entry = XIDEntry.from_message(message=args.xid_message)
        if not xid_entry:
            logging.error("Error processing single XID message.")
            return 1
            
        xid_entry.lookup_resolution()
        
        # Define fields to display, adding optional ones if they exist
        fields = ['timestamp', 'number', 'mnemonic', 'context', 'resolution', 'investigatory_action', 'job_failure_severity', 'hw_sw_classification', 'comments']
        
        logging.info("\nXID Analysis Results:")
        # Use to_table with single entry
        logging.info(XIDEntry.to_table(xid_entry, fields))
        return
    
    
    
    # Step 4: Read in Job Runtime Data if provided
    if args.job_runtimes:
        logging.info(f"Loading job runtime data from {args.job_runtimes}...")
        try:
            with open(args.job_runtimes, 'r') as job_file:
                reader = csv.DictReader(job_file)
                for row in reader:
                    try:
                        # Parse timestamps and ensure they have timezone
                        start_time = ensure_timezone_local(datetime.strptime(row['Job Start'], '%Y-%m-%d %H:%M:%S'))
                        end_time = ensure_timezone_local(datetime.strptime(row['Job End'], '%Y-%m-%d %H:%M:%S'))
                        
                        # Create JobEntry object
                        job = JobEntry(
                            name=row['Job Type'],
                            node=row['Machine Id'], 
                            job_type=row['Job Type'],
                            start_time=start_time,
                            end_time=end_time,
                            vbios_version=row.get('FW Ver(VBIOS)', ''),
                            driver_version=row.get('SW Ver(Driver)', '')  # May not be in file
                        )
                    except KeyError as e:
                        missing_col = str(e).strip("'\"")
                        logging.error(f"Missing required column '{missing_col}' in job runtime CSV file '{args.job_runtimes}'.")
                        raise ValueError(f"Missing required column '{missing_col}' in job runtime CSV file")
                    
                    # Add to fleet data
                    node_entry = fleet_data._get_or_create_node(job.node)
                    node_entry.job_runtimes.append(job)
                    
                    # Update version history with job data
                    node_entry.update_version_history(job.start_time, job.vbios_version, job.driver_version)
                    
                    # Update fleet metadata
                    fleet_data.node_names.add(job.node)
                    fleet_data.job_types.add(job.job_type)
                    if job.vbios_version:
                        fleet_data.vbios_versions.add(job.vbios_version)
                    if job.driver_version:
                        fleet_data.driver_versions.add(job.driver_version)
                    
                    # Update report period
                    if not fleet_data.report_period_start or start_time < fleet_data.report_period_start:
                        fleet_data.report_period_start = start_time
                    if not fleet_data.report_period_end or end_time > fleet_data.report_period_end:
                        fleet_data.report_period_end = end_time
                        
            logging.info(f"Loaded job entries for {len(fleet_data.nodes)} nodes")
            
        except Exception as e:
            logging.error(f"Error reading job runtime data: {e}")
            return 1
    
    # Read in inventory data if provided
    if args.inventory_data:
        logging.info(f"Loading inventory data from {args.inventory_data}...")
        try:
            with open(args.inventory_data, 'r') as membership_file:
                reader = csv.DictReader(membership_file)
                for row in reader:
                    try:
                        # Parse timestamp and ensure it has timezone
                        timestamp_in_row = ensure_timezone_local(datetime.strptime(row['Timestamp'], '%Y-%m-%d %H:%M:%S'))
                        node_name_in_row = row['Machine Id']
                        state_in_row = row['State']
                    except KeyError as e:
                        missing_col = str(e).strip("'\"")
                        logging.error(f"Missing required column '{missing_col}' in inventory CSV file '{args.inventory_data}'.")
                        raise ValueError(f"Missing required column '{missing_col}' in inventory CSV file")
                    
                    if state_in_row not in ['Deployed', 'Retired']:
                        continue
                    
                    # Get VBIOS and driver versions if available in the CSV
                    vbios_version = row.get('FW Ver(VBIOS)') if 'FW Ver(VBIOS)' in row else None
                    driver_version = row.get('SW Ver(Driver)') if 'SW Ver(Driver)' in row else None
                    
                    # Add to fleet data structures
                    if vbios_version:
                        fleet_data.vbios_versions.add(vbios_version)
                    if driver_version:
                        fleet_data.driver_versions.add(driver_version)
                    
                    node_entry = fleet_data._get_or_create_node(node_name_in_row)
                    
                    # Update version history with VBIOS and driver information
                    node_entry.update_version_history(timestamp_in_row, vbios_version, driver_version)
                    
                    # Create new tenure if node is being deployed and doesn't already have an active tenure
                    # (active tenure = one where left_time is None)
                    if state_in_row == 'Deployed' and not any(t.left_time is None for t in node_entry.in_production_times):
                        node_entry.in_production_times.append(InProductionTime(
                            node=node_name_in_row,
                            joined_time=timestamp_in_row,
                            left_time=None
                        ))
                    elif state_in_row == 'Retired':
                        # Update most recent active tenure or create new one
                        for tenure in reversed(node_entry.in_production_times):
                            if tenure.left_time is None:
                                tenure.left_time = timestamp_in_row
                                break
                        else:
                            node_entry.in_production_times.append(InProductionTime(
                                node=node_name_in_row,
                                joined_time=None,
                                left_time=timestamp_in_row
                            ))
                    
                    fleet_data.node_names.add(node_name_in_row)
                    
                    # Update report period bounds
                    fleet_data.report_period_start = min(timestamp_in_row, fleet_data.report_period_start or timestamp_in_row)
                    fleet_data.report_period_end = max(timestamp_in_row, fleet_data.report_period_end or timestamp_in_row)
                
            total_entries = sum(len(node.in_production_times) for node in fleet_data.nodes)
            logging.info(f"Loaded {total_entries} inventory entries for {len(fleet_data.nodes)} nodes")
            
        except Exception as e:
            logging.error(f"Error reading inventory data: {e}")
            return 1
    
    
    # Step 5: Read in XID Data if provided
    if args.find_resolutions or args.calculate_mtbi:
        input_file = args.find_resolutions or args.calculate_mtbi
        logging.info(f"Reading XID data from {input_file}...")
        
            
        # Check if file reading was successful
        if not fleet_data.read_xid_data(input_file, drop_duplicates=args.drop_duplicates):
            logging.error(f"Failed to read XID data from {input_file}. Exiting.")
            return 1
        
        # Check if any XIDs were loaded
        has_xids = any(len(node.xid_entries) > 0 for node in fleet_data.nodes)
        if not has_xids:
            logging.info("No XIDs found in the file.")
            return 1
        
        # Step 6: Find resolution for each XID
        logging.info("Finding resolutions for XIDs...")
        fleet_data.find_resolutions()
        
    
    
    
    # If calculating MTBI (fleet analysis)
    if args.calculate_mtbi:
        # Step 7: Tag XIDs
        logging.info("Tagging XIDs based on watchlist...")
        fleet_data.tag_xids()

        # Step 8: Group XIDs into bursts
        logging.info("Grouping XIDs into bursts...")
        fleet_data.group_xids_into_bursts()
        
        # Step 9: Compare each burst to burst patterns
        logging.info("Matching bursts to known failure patterns...")
        fleet_data.match_bursts_to_patterns()
        
        # Generate daily statistics
        logging.info("Generating daily statistics...")
        fleet_data.derive_stats()
        
        # Calculate VBIOS hours for charting
        logging.info("Calculating VBIOS usage hours...")
        fleet_data.calculate_vbios_hours()

        # Step 10: Calculate MTBI
        logging.info("Calculating MTBI...")
        # Pass CLI arguments only if provided, otherwise calculate_mtbi will infer from job data
        fleet_size = args.fleet_size if args.fleet_size > 0 else 0
        days_used = args.days_used if args.days_used > 0 else 0
        fleet_data.calculate_mtbi(fleet_size, days_used)        
        
        # Step 11: Generate console report
        # Display metrics summary
        metrics_table = []
        metrics_table.append(["fleet_size", f"{fleet_data.fleet_size}"])
        metrics_table.append(["work_time (days)", f"{fleet_data.total_work_time.total_seconds() / (3600 * 24)}"])
        metrics_table.append(["total_bursts", f"{fleet_data.total_bursts}"])
        metrics_table.append(["total_interruptions", f"{fleet_data.total_interruptions_count}"])
        metrics_table.append(["total_interruptions_in_production", f"{fleet_data.total_interruptions_in_production_count}"])
        
        # Add MTBI metrics
        metrics_table.append(["mtbi_years", "∞" if fleet_data.mtbi_years == float('inf') else f"{fleet_data.mtbi_years:.2f}"])
        metrics_table.append(["mtbi_hours", "∞" if fleet_data.mtbi_years == float('inf') else f"{fleet_data.total_mtbi.total_seconds() / 3600:.2f}"])
        
        logging.info("\nXID Analysis Summary:")
        logging.info(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="psql", colalign=("left", "right")))
        
        # Display the category data if present
        category_counts = fleet_data.get_category_counts()
        if category_counts:
            cat_data = []
            for (category, subcategory), count in sorted(category_counts.items(), 
                                                         key=lambda x: x[1], 
                                                         reverse=True):
                if category:  # Skip empty categories
                    cat_data.append([category, subcategory, count])
            
            if cat_data:
                logging.info("\nMTBI by Category:")
                logging.info(tabulate(cat_data, headers=["Category", "Subcategory", "Count"], tablefmt="psql"))
        
        # Display node-level MTBI data
        nodes_with_interruptions = [node for node in fleet_data.nodes if node.interruptions_count > 0]
        if nodes_with_interruptions:
            # Sort nodes by interruption count (descending)
            sorted_nodes = sorted(nodes_with_interruptions, 
                                key=lambda n: n.interruptions_count, 
                                reverse=True)
            
            # Prepare table data
            node_data = []
            for node in sorted_nodes[:20]:  # Show top 20 nodes by default
                work_hours = node.work_time.total_seconds() / 3600
                mtbi_hours = node.mtbi.total_seconds() / 3600
                node_data.append([
                    node.name,
                    f"{work_hours:.2f}",
                    node.interruptions_count,
                    f"{mtbi_hours:.2f}"
                ])
            
            logging.info("\nTop 20 Nodes by Interruption Count:")
            logging.info(tabulate(node_data, 
                                headers=["Node", "Work Hours", "Interruptions", "MTBI (hours)"], 
                                tablefmt="psql"))


    # If finding resolutions in a file (not fleet analysis)
    elif args.find_resolutions:
        # Step 11: Generate console report for resolution finding
        # Get all XID entries from all nodes
        all_xid_entries = []
        for node in fleet_data.nodes:
            all_xid_entries.extend(node.xid_entries)
            
        if all_xid_entries:
            # Define fields to display
            fields = ['timestamp', 'number', 'mnemonic', 'driver', 'vbios_version', 'machine', 'context', 'resolution', 'job_failure_severity', 'hw_sw_classification']
            
            # Add optional fields if present
            if any(entry.investigatory_action for entry in all_xid_entries):
                fields.append('investigatory_action')
            if any(entry.comments for entry in all_xid_entries):
                fields.append('comments')
            
            # Use the class method to generate the table
            logging.info("\nXID Analysis Results:")
            logging.info(XIDEntry.to_table(all_xid_entries, fields))
            
            # Get unique counts
            total_xids = len(all_xid_entries)
            
            # Initialize variables to avoid UnboundLocalError
            unique_nodes = 0
            unique_jobs = None
            unique_drivers = None
            unique_vbios_versions = 0
            
            # Count unique jobs across all nodes
            all_jobs = []
            for node in fleet_data.nodes:
                all_jobs.extend(node.job_runtimes)
                
            if all_jobs:
                unique_nodes = len(fleet_data.nodes)
                unique_jobs = len(set(entry.name for entry in all_jobs))
                unique_vbios_versions = len(set(entry.vbios_version for entry in all_jobs if entry.vbios_version))
                # Calculate unique drivers from fleet_data if available
                unique_drivers = len(fleet_data.driver_versions)
            else:
                unique_nodes = len(fleet_data.nodes)
                unique_drivers = len(set(entry.driver if entry.driver else "unknown" for entry in all_xid_entries))
                unique_vbios_versions = len(set(entry.vbios_version for entry in all_xid_entries))
            
            # Log summary statistics
            logging.info("\nSummary Statistics:")
            logging.info(f"Unique nodes found: {unique_nodes}")
            logging.info(f"Unique jobs found: {unique_jobs if unique_jobs else 'N/A'}")
            logging.info(f"Unique driver versions found: {unique_drivers if unique_drivers else 'N/A'}")
            logging.info(f"Unique vbios versions found: {unique_vbios_versions if unique_vbios_versions else 'N/A'}")
            logging.info(f"Total XID entries: {total_xids}")
    
    # Step 12: Export CSV files and/or generate HTML report if requested
    if (args.generate_charts or args.export_csvs) and fleet_data.nodes:
        # Create output directory if it doesn't exist
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.export_csvs or args.generate_charts:
            logging.info("\nExporting analysis results to CSV files...")
            
            # Export XID data to CSV first
            csv_output_file = str(args.output_dir / "xid_analysis_results.csv")
            fleet_data.export_to_csv(csv_output_file)
            
            # Copy XID data file if provided
            if args.find_resolutions or args.calculate_mtbi:
                xid_file = args.find_resolutions or args.calculate_mtbi
                logging.info(f"Copying input xid file: {xid_file}")
                try:
                    shutil.copy2(xid_file, args.output_dir / f"{xid_file.name}")
                except Exception as e:
                    logging.error(f"Failed to copy XID data file: {e}")

            # Copy job runtimes file if provided  
            if args.job_runtimes:
                logging.info(f"Copying input job runtimes file: {args.job_runtimes}")
                try:
                    shutil.copy2(args.job_runtimes, args.output_dir / f"{args.job_runtimes.name}")
                except Exception as e:
                    logging.error(f"Failed to copy job runtimes file: {e}")
            
            # Copy inventory data file if provided
            if args.inventory_data:
                logging.info(f"Copying input inventory data file: {args.inventory_data}")
                try:
                    shutil.copy2(args.inventory_data, args.output_dir / f"{args.inventory_data.name}")
                except Exception as e: 
                    logging.error(f"Failed to copy inventory data file: {e}")

            # Export bursts to CSV
            export_xid_bursts_analysis_to_csv(fleet_data, str(args.output_dir / "xid_bursts_analysis.csv"))
            
            # Export node MTBI data to CSV
            export_node_mtbi_to_csv(fleet_data, str(args.output_dir / "node_mtbi.csv"))
            
            # Export individual XIDs to CSV
            export_xid_list_analysis_to_csv(fleet_data, str(args.output_dir / "xid_list_analysis.csv"))
            
            logging.info(f"CSV files exported to: {args.output_dir}")
        
        if args.generate_charts:
            logging.info("\nGenerating charts and HTML report...")
            try:
                # Generate HTML report
                from visualizer import generate_charts_from_fleet_data
                generate_charts_from_fleet_data(fleet_data, args.output_dir, args.output_file, timeline_node_limit=20)
            except ImportError:
                logging.error("Could not import visualizer.py module. Make sure it's in the same directory.")
    elif args.generate_charts or args.export_csvs:
        logging.info("Please provide a file to analyze with --calculate-mtbi along with --generate-charts or --export-csvs")
        logging.info("For example: ./nvidia_xid_analyzer.py --calculate-mtbi xid_data.csv --export-csvs")
        
    
    
    # Display help if no valid input source provided
    if not any([args.xid_message, args.find_resolutions, args.calculate_mtbi, args.generate_charts, args.export_csvs]):
        parser.print_help()
    
    # Return 0 on successful execution
    return 0

if __name__ == "__main__":
    sys.exit(main())
    
