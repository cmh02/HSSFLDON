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
import time
import requests
from dotenv import load_dotenv

# Project Imports
from hssfldon.common.hssfldon_logger import HSSFLDON_Logger
from hssfldon.common.hssfldon_enum import HSSFLDON_ClientState, HSSFLDON_ClientTask


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
		self.logger: HSSFLDON_Logger = HSSFLDON_Logger(name=f"Client {self.client_id}")
		self.logger.info(f"Initialized HSSFLDON Client Application with PID: {os.getpid()}!")

		# Load server API details from env
		self.server_host: str = os.getenv("HSSFLDON_SERVER_HOST", "127.0.0.1")
		self.server_port: int = int(os.getenv("HSSFLDON_SERVER_PORT", 8000))
		self.server_api_url: str = f"http://{self.server_host}:{self.server_port}"

		# Load client standby delay from env
		self.standbyDelay: int = int(os.getenv("HSSFLDON_CLIENT_STANDBY_DELAY", 5))

		# Register with server
		self.registered: bool = self.register()

	def doClientLoop(self):
		"""
		Main loop for the client application.
		"""
		self.logger.info(f"Starting main client loop!")
		while True:

			# Get next task from the server
			task: HSSFLDON_ClientTask = self.getNextTask()

			# Task: Standby
			if (task == HSSFLDON_ClientTask.STANDBY):
				self.logger.debug(f"Client received STANDBY task. Waiting for {self.standbyDelay} seconds before checking for new tasks!")
				time.sleep(self.standbyDelay)
				continue

			# Task: Passive Learning


	def checkServerHealth(self) -> bool:
		"""
		Check the health of the server by making a request to the /health endpoint.
		"""

		# Get number of attempts and delay between attempts from env
		attempts: int = int(os.getenv("HSSFLDON_SERVER_HEALTH_CHECK_ATTEMPTS", 5))
		delay: int = int(os.getenv("HSSFLDON_SERVER_HEALTH_CHECK_DELAY", 5))

		# Try to check server health with retries
		for attempt in range(1, attempts + 1):
			self.logger.debug(f"Checking server health (Attempt {attempt}/{attempts})...")
			try:
				response: requests.Response = requests.get(f"{self.server_api_url}/health")
				response.raise_for_status()
				data: dict = response.json()
				if data.get("status") == "ok":
					self.logger.debug(f"Server is online and responsive!")
					return True
				else:
					self.logger.warning(f"Server health check failed: {data.get('message')}")
			except Exception as e:
				self.logger.error(f"An exception occurred during server health check: {e}")

			# Wait before next attempt
			if attempt < attempts:
				self.logger.info(f"Waiting for {delay} seconds before next health check attempt!")
				time.sleep(delay)

		self.logger.error(f"Server health check failed after {attempts} attempts. Server may be offline or unresponsive.")
		return False

	def register(self) -> bool:
		"""
		Register the client with the server.
		"""
		self.logger.debug(f"Making request to register with server!")

		# Check server health before attempting to register
		if not self.checkServerHealth():
			self.logger.error(f"Cannot register with server because server is offline or unresponsive!")
			return False

		# Build payload
		payload = {"client_id": self.client_id}
		headers = {"Content-Type": "application/json"}

		# Make request to server
		try:
			response: requests.Response = requests.post(f"{self.server_api_url}/register", json=payload, headers=headers)
			response.raise_for_status()
			data: dict = response.json()
			self.logger.info(f"Successfully registered with server!")
			return True
		except Exception as e:
			self.logger.error(f"An exception occured when trying to register with server!")
			self.logger.debug(f"-> Exception details: {e}")
		return False
	
	def getNextTask(self) -> HSSFLDON_ClientTask:
		"""
		Retrieve the next task for the client from the server.
		"""
		self.logger.debug(f"Making request to fetch next task!")
		try:
			response: requests.Response = requests.get(f"{self.server_api_url}/task?client_id={self.client_id}")
			response.raise_for_status()
			data: dict = response.json()
			task: HSSFLDON_ClientTask = HSSFLDON_ClientTask(data.get("task"))
			self.logger.info(f"Received task from server: {task.name}")
			return task
		except Exception as e:
			self.logger.error(f"An exception occurred while fetching next task: {e}")
		return HSSFLDON_ClientTask.STANDBY