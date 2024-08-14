#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'Brian Richardson'
__copyright__ = 'PLEN Robotics Inc, and all authors.'
__license__ = 'All rights reserved'

import sys
import csv
import pprint
from calendar import monthrange
import datetime

def read_csv(filepath, encoding='utf-8'):
    # Return the data and take the first line of csv as header
    # Since name is required to be not null, 
    # rows that cannot be imported will only have the first column and 
    # None for the rest of the values
    data = []
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = reader.next()
        for row in reader:
            try: data.append([r.decode(encoding) for r in row])
            except: data.append([row[0]]+[None for r in row[1:]])
    return header, data

def summarize_csv(filepath, date_constraint):
	header, data = read_csv(filepath)
	date_constraint_ymd = date_constraint.split("-",2)
	print(date_constraint_ymd)

	date_totals = initialize_dates(date_constraint_ymd)
	dates = []
	males = []
	females = []

	for line in data:
		date_time = line[0].split(" ", 2)
		year_month_date = date_time[0].split("/", 2)

		if len(date_constraint_ymd) == 0 or len(date_constraint_ymd) == 1:
			if year_month_date[0] == date_constraint_ymd[0]: 
				date_totals = summarize_gender_data(date_totals, date_time, line)
		elif len(date_constraint_ymd) >= 2:
			if year_month_date[0] == date_constraint_ymd[0] and year_month_date[1] == date_constraint_ymd[1]:
				date_totals = summarize_gender_data(date_totals, date_time, line) 

	for key, val in sorted(date_totals.items()):
		dates.append(key)
		males.append(val['m'])
		females.append(val['f'])
	pprint.pprint(date_totals)
	return dates, males, females

def summarize_gender_data(dictionary, date_time, line):
	date_totals = dictionary
	if date_time[0] in date_totals:
		if line[2] == u"女性":
			date_totals[date_time[0]]["f"] += 1
		if line[2] == u"男性":
			date_totals[date_time[0]]["m"] += 1
	return date_totals

def initialize_dates(date_constraint_ymd):
	# initialize current date for default or invalid date constraints
	dt = datetime.datetime.today()
	month = dt.month
	year = dt.year
	dates = {}

	if len(date_constraint_ymd) >= 2:
		month = date_constraint_ymd[1]
		year = date_constraint_ymd[0]

	days = monthrange(int(year), int(month))

	for day in range(days[1]):
		if month > 9:
			if day > 8:
				date_string = u"{year}/{month}/{day}".format(month = month, year = year, day = str(day +1))
			else:
				date_string = u"{year}/{month}/0{day}".format(month = month, year = year, day = str(day +1))
		else:
			if day > 8:
				date_string = u"{year}/0{month}/{day}".format(month = month, year = year, day = str(day +1))
			else:
				date_string = u"{year}/0{month}/0{day}".format(month = month, year = year, day = str(day +1))
		
		dates[date_string] = {"m": 0, "f": 0}

	return dates



if __name__ == "__main__":
	filepath = sys.argv[1]
	summarize_csv(filepath)





