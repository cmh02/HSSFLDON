'''
	# HSSFLDON - Client Application

	This module will provide the client application for HSSFLDON.
	
	The client is responsible for:
	- Managing localized labeled dataset
	- Training local model
	- Receiving active labeling requests
	- Uploading model updates to server
'''

# Library Imports
import os

# Project Imports
from hssfldon.common.hssfldon_logger import HSSFLDON_Logger


class HSSFLDON_ClientApplication:
	"""
	The main client application for HSSFLDON.
	"""
	def __init__(self):

		# Get logger
		self.logger = HSSFLDON_Logger(name=f"Client_{os.getpid()}")
		self.logger.info(f"Initialized HSSFLDON Client Application with PID: {os.getpid()}!")

		pass

	