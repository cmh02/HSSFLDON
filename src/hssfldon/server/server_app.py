'''
	# HSSFLDON - Server Application

	This module will provide the server application for HSSFLDON.
	
	The server is responsible for:
	- Managing unlabeled dataset
	- Distributing unknown datapoints to clients for labeling
	- Aggregating client updates in global model



'''

# Library Imports
import os
import time
from dotenv import load_dotenv

# Project Imports
from hssfldon.common.hssfldon_logger import HSSFLDON_Logger
from hssfldon.common.hssfldon_enum import HSSFLDON_ServerState, HSSFLDON_ClientTask


class HSSFLDON_ServerApplication:
	"""
	The main server application for HSSFLDON.
	"""
	def __init__(self):

		# Parse dotenv for env variables
		load_dotenv()

		# Get logger
		self.logger = HSSFLDON_Logger(name=f"Server")
		self.logger.info(f"Initialized HSSFLDON Server Application with PID: {os.getpid()}!")

		# Initialize client tracking
		self.clients: list[int] = []
		self.clientTasks: dict[int, HSSFLDON_ClientTask] = {}

		# Startup API and idle until API online
		self.launchApi()
		self.enterState(HSSFLDON_ServerState.IDLE)

		# Wait for client registration for configured window in seconds
		self.enterState(HSSFLDON_ServerState.WAITING_CLIENT_REGISTRATION)
		registrationWindow = int(os.getenv("HSSFLDON_CLIENT_REGISTRATION_WINDOW", 30))
		self.logger.info(f"Waiting for client registration for {registrationWindow} seconds!")


	def launchApi(self):
		"""
		Launch the server API to listen for client requests.
		"""
		self.logger.info(f"Launching Server API!")

	def closeApi(self):
		"""
		Close the server API and clean up resources.
		"""
		self.logger.info(f"Closing Server API!")

	def enterState(self, state: HSSFLDON_ServerState):
		"""
		Enter a specific state and perform actions for that state.
		"""
		self.logger.debug(f"Entering Server State: {state.name}!")
		self.state = state

	def registerClient(self, clientId: int):
		"""
		Register a new client with the server.
		"""
		self.logger.info(f"Registering Client with ID: {clientId}!")
		self.clients.append(clientId)
		self.clientTasks[clientId] = HSSFLDON_ClientTask.STANDBY