#!/usr/bin/env python3

import logging

# a terrible solution to a simple problem; a shame I can't come
#  up with anything better

# basically, it uses process-level state to tack on an
#  extra level to the logging heirarchy, so that output
#  from each process can be identified in the log

job_suffix = None

def set_job_suffix(s):
	global job_suffix
	job_suffix = str(s)

def getLogger(name=None):
	dot = '.' if name and job_suffix else ''

	name = name or ''
	name = str(name) + dot + job_suffix

	if name:
		return logging.getLogger(name)
	else:
		return logging.getLogger()
	assert False
