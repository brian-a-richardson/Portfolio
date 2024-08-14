#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json 
import pprint
import pytz
import sys
import csv
from dateutil import tz
from datetime import datetime

def parse_json(filepath):
    """Given the path to a json file, parse the content in a more readable manner
       Convert a UNIX epoch milli sec UTC timestamp to local timestamp
    Args:
        value (str): json file path
        structure: 
        {“<random_key>”:
        {“cubeMac”: “<cube_mac_address>“, “status”: “2022”, “message”: “”, “timestamp”: “163037…”}
        }
    Returns:
        value (dict): JST time stamp
        structure: 
        {“<cube_mac_address>”: {“status”: “2022”, “message”: “”, “timestamp”: “2021-12-01 12:00:00”}}
    """
    with open(filepath) as f:
        original = json.load(f)
    parsed = {}
    for random_key, cubeMac_value in original.items():
        cube_mac_address = cubeMac_value["cubeMac"].replace(':', '')
        status = cubeMac_value["status"]
        message = cubeMac_value["message"]
        timestamp = int(cubeMac_value["timestamp"])
        # The last timestamp is saved
        if not parsed.get(cube_mac_address, "") or timestamp > parsed.get(cube_mac_address, {}).get("timestamp", 0):
            parsed[cube_mac_address] = {"status": status, "message": message, "timestamp": timestamp}

    for cube_mac_address, value in parsed.items():
        # UNIX epoch milli sec (UTC) -> datetime.datetime object (local JST)
        dt = datetime.fromtimestamp(value["timestamp"] / 1000).replace(tzinfo=tz.tzutc()).astimezone(tz.tzlocal())
        value["timestamp"] = dt.strftime('%Y-%m-%d %H:%M:%S')

    pprint.pprint(parsed)
    return parsed

def replace_cube_mac_with_sn(parsed_json, filepath):
    """
    Find corresponding mac addresses and replace them with corresponding serial numbers.  
    If no corresponding serial number is available leave the mac address.

    Args:
    value (dict) : parsed JSON
    value (str) : json file path

    Returns:
    value (dict) : replaced mac address with serial number
    """
    with open(filepath) as f:
        mac_to_sn = json.load(f)
    
    parsed = parsed_json

    for cube_mac_address, cube_sn in mac_to_sn.items():
        if cube_mac_address in parsed:
            serial = cube_sn
            parsed[serial] = parsed.pop(cube_mac_address) 

    pprint.pprint(parsed)
    return parsed

def write_to_csv(parsed_json, filepath):
    """
    Takes a parsed json dictionary and writes it to a .csv file
    Args:
    value (dict) : parsed JSON
    value (str) : json file path
    """
    with open(filepath) as f:
        unsorted_dict = json.load(f)

    serial_numbers = []

    for serial in unsorted_dict:
        serial_numbers.append(unsorted_dict[serial])

    serial_numbers.sort()

    with open('plencubes.csv', 'w') as csvfile:
        header_key = ['cube_sn', 'status', 'message', 'time']
        writer = csv.DictWriter(csvfile, fieldnames = header_key)
        writer.writeheader()
        for serial in serial_numbers:
            if serial in parsed_json:
                writer.writerow({'cube_sn' : serial, 'status' : parsed_json[serial]['status'], 
                    'message' : parsed_json[serial]['message'], 'time' : parsed_json[serial]['timestamp']})
            else:
                writer.writerow({'cube_sn' : serial, 'status' : '', 
                    'message' : '', 'time' : ''})

def main():
    if len(sys.argv) == 2:
        parse_json(sys.argv[1])
    elif len(sys.argv) == 3:
        write_to_csv(replace_cube_mac_with_sn(parse_json(sys.argv[1]), sys.argv[2]), sys.argv[2])
    elif len(sys.argv) < 2:
        print("Error. Insufficient arguments supplied.  Please provide filepath to JSON data file.")
    else:
        print("Error. Too many arguments supplied.")



if __name__ == "__main__":
    main()