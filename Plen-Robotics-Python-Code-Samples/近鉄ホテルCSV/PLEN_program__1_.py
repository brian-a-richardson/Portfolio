#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import csv
import pprint
import chardet
import json


def read_csv(filepath):
    data = []
    # get encoding data
    with open(filepath, 'rb') as rawdata:
    	encoding = chardet.detect(rawdata.read(10000))
    # open file and return data
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            try: data.append([r.decode(encoding.get('encoding')) for r in row])
            except: data.append([row[0]]+[None for r in row[1:]])
    return header, data

def sort_csv_data():
	'''
	Take data from two CSV files, then create a dictionary that combines data from both CSV files

	'''
	# load data from csv files
	csv_header, csv_data = read_csv(u'今野SL.csv')
	cube_csv_header, cube_csv_data = read_csv('./CubeData/attendance_20220225171628_utf8.csv')

	# create a dictionary from data
	data = csv_to_dict(csv_data)

	# add cube data
	for line in cube_csv_data:
		id_num = id_string_to_int(line[3])
		if id_num in data:
			date_time = line[0].split(" ")
			for d in data[id_num]:
				if date_time[0] == d["date"]:	
					if line[4] == u"入室":
						d["in"]["cube"] = format_time(date_time[1])
					elif line[4] == u"滞在中":
						d["out"]["cube"] = format_time(date_time[1])
					else:
						print("Else")

	# write data to JSON
	json_object = json.dumps(data, indent=4)
	with open("comparison.json", "w") as outfile:
		outfile.write(json_object)

	pprint.pprint(data)


def csv_to_dict(input_data):
	'''
	Takes a list of list containg 'log in' and 'log out' data.  Organizes that list into a dictionary.
	Dictionary Format:
	{
	  'ID001': [
	    {
	      'date': '2022-01-01',
	      'in': {
	        'actual': '2022-01-01 00:00:05',
	        'plan': '2022-01-01 00:00:00',
	        'cube': ['2022-01-01 00:48:59']
	      },
	      "out": {
	        "actual": '2022-01-02 06:03:00',
	        "plan": '2022-01-02 06:00:00',
	        'cube': []
	      }
	    }
	  ]
	}
	'''
	data = {}
	id_counter = 1
	for line in input_data:
		id_num = id_string_to_int(line[0])
		if id_num not in data:
			# create a new user id if it doesn't exist
			# remove the commas from user id and turn it into an integer
			data[id_num] = []
		# add data by date
		time_in = {"actual": format_time(line[4]),
					"plan": format_time(line[3]),
					"cube": []}
		time_out = {"actual": format_time(line[6]),
					"plan": format_time(line[5]),
					"cube": []}
		data[id_num] += [{"date": format_date(line[2]),
							"in": time_in,
							"out": time_out}] 


	return data

def format_date(date):
	'''
	Takes the date string in the 2022/1/2 format, and converts it to the 2022-01-02 format 
	'''
	date_divided = date.split("/")
	if len(date_divided[1]) < 2:
		date_divided[1] = "0" + date_divided[1]
	if len(date_divided[2]) < 2:
		date_divided[2] = "0" + date_divided[2]
	return date_divided[0] + "-" + date_divided[1] + "-" + date_divided[2]

def format_time(time):
	'''
	Rounds the time to the nearest second for times that include smaller units of time
	For example xx:xx:xx => xx:xx
	'''
	if len(time) > 5:
		return time[0:5]
	else:
		return time

def id_string_to_int(input_string):
	'''
	The ID number comes as two strings of different formats from the csv files
	The first:
		A ten digit number with leading 0s if the id number < 10 digits
	The second:
		A n digit number with commas every three values

	Commas (if any) are removed and the string is converted to an integer
	'''
	if len(input_string) > 0:
		no_coma = input_string.replace(",","")
		id_num = int(no_coma)
	else:
		id_num = None
	return id_num
			

def main():
	sort_csv_data()    

if __name__ == "__main__":
    main()
