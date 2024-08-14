#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'Brian Richardson'
__copyright__ = 'PLEN Robotics Inc, and all authors.'
__license__ = 'All rights reserved'

import flask
from flask import render_template, redirect, url_for
import summarize_csv as sv
import sys

app = flask.Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')
app.config["DEBUG"] = True

@app.route("/chart/")
@app.route("/chart/<date_constraint>", methods=['GET', 'POST'])
def chart(date_constraint = "2021"):
	# get and format data for chart.js
	filepath = sys.argv[1]
	dates, males, females = sv.summarize_csv(filepath, date_constraint)
	# dynamically adjust the chart of the chart based on numbers in a list
	height = str(len(dates) * 40) + "px" 
	return render_template('chart.html', dates = dates, males = males, females = females, height=height)

if __name__ == '__main__':
	app.run(host="localhost", port=8000, debug=True)
