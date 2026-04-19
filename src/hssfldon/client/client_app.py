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
from dotenv import load_dotenv

# Project Imports
from hssfldon.common.hssfldon_logger import HSSFLDON_Logger


class HSSFLDON_ClientApplication:
	"""
	The main client application for HSSFLDON.
	"""
	def __init__(self):

		# Load env file
		load_dotenv(dotenv_path='../.env')

		# Get client ID from env
		self.client_id: str = os.getenv("HSSFLDON_CLIENT_ID", f"{os.getpid()}")

		# Get logger
		self.logger = HSSFLDON_Logger(name=f"Client {self.client_id}")
		self.logger.info(f"Initialized HSSFLDON Client Application with PID: {os.getpid()}!")

		pass

	