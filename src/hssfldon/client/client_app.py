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
import requests
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
		self.client_id: int = int(os.getenv("HSSFLDON_CLIENT_ID", f"{os.getpid()}"))

		# Get logger
		self.logger = HSSFLDON_Logger(name=f"Client {self.client_id}")
		self.logger.info(f"Initialized HSSFLDON Client Application with PID: {os.getpid()}!")

		# Load server API details from env
		self.server_host = os.getenv("HSSFLDON_SERVER_HOST", "127.0.0.1")
		self.server_port = int(os.getenv("HSSFLDON_SERVER_PORT", 8000))
		self.server_api_url = f"http://{self.server_host}:{self.server_port}"

		# Register with server
		self.register()

	def register(self):
		"""
		Register the client with the server.
		"""
		self.logger.info(f"Making request to register with server!")

		# Build payload
		payload = {"client_id": self.client_id}
		headers = {"Content-Type": "application/json"}

		# Make request to server
		try:
			response = requests.post(f"{self.server_api_url}/register", json=payload, headers=headers)
			response.raise_for_status()
			data = response.json()
			self.logger.info(f"Successfully registered with server!")
			return data
		except Exception as e:
			self.logger.error(f"An exception occured when trying to register with server!")
			self.logger.debug(f"-> Exception details: {e}")
		return None