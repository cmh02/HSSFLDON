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
from peft import PeftModel
import torch
import uvicorn
import requests
import threading
from fastapi import FastAPI
from dotenv import load_dotenv



# Project Imports
from hssfldon.common.hssfldon_logger import HSSFLDON_Logger
from hssfldon.common.hssfldon_enum import HSSFLDON_ServerState, HSSFLDON_ClientTask
from hssfldon.server.server_api import HSSFLDON_ServerAPIRouter
from hssfldon.common.hssfldon_model import HSSFLDON_ModelManager

class HSSFLDON_ServerApplication:
	"""
	The main server application for HSSFLDON.
	"""
	def __init__(self):

		# Parse dotenv for env variables
		envStatus: bool = load_dotenv()
		if envStatus is False:
			print(f"Warning: .env file not found or failed to load. Make sure to create a .env file with the necessary configuration variables!")

		# Get logger
		self.logger = HSSFLDON_Logger(name=f"Server")
		self.logger.info(f"Initialized HSSFLDON Server Application with PID: {os.getpid()}!")

		# Initialize client tracking
		self.clients: list[int] = []
		self.clientTasks: dict[int, HSSFLDON_ClientTask] = {}

		# Setup API
		self.api_host = os.getenv("HSSFLDON_SERVER_HOST", "127.0.0.1")
		self.api_port = int(os.getenv("HSSFLDON_SERVER_PORT", 8000))
		self.api_app = FastAPI(title="HSSFLDON Server API")
		self.api_app.include_router(HSSFLDON_ServerAPIRouter)
		self.api_app.state.server_app = self
		self.api_config = uvicorn.Config(app=self.api_app, host=self.api_host, port=self.api_port, log_level="info")
		self.api_server = uvicorn.Server(config=self.api_config)
		self.api_thread = None

		# Launch API and idle
		self.launchApi()
		self.enterState(HSSFLDON_ServerState.IDLE)

		# Wait for client registration for configured window in seconds
		self.enterState(HSSFLDON_ServerState.WAITING_CLIENT_REGISTRATION)
		registrationWindow = int(os.getenv("HSSFLDON_CLIENT_REGISTRATION_WINDOW", 30))
		self.logger.info(f"Waiting for client registration for {registrationWindow} seconds!")
		time.sleep(registrationWindow)
		self.enterState(HSSFLDON_ServerState.IDLE)

		# Setup model
		self.adaptersDirectory = os.getenv("HSSFLDON_MODEL_ADAPTERS_DIRECTORY", "model_adapters")
		self.adaptersGlobalName = os.getenv("HSSFLDON_MODEL_ADAPTERS_GLOBAL", "global")
		self.adaptersGlobalFullPath = os.path.join(self.adaptersDirectory, self.adaptersGlobalName)
		self.modelName = os.getenv("HSSFLDON_HF_MODEL", "meta-llama/Llama-3.2-1B")
		self.initializeModel()

		# # Close API and shutdown everything (for now)
		# self.closeApi()

	def launchApi(self) -> bool:
		"""
		Launch the server API to listen for client requests.
		"""
		if self.api_thread is not None and self.api_thread.is_alive():
			self.logger.warning("Call was made to launch API but API server is already running!")
			return False

		self.logger.info(f"Starting API server on `http://{self.api_host}:{self.api_port}`!")
		self.api_thread = threading.Thread(target=self.api_server.run, daemon=True)
		self.api_thread.start()
		return True

	def closeApi(self) -> bool:
		"""
		Close the server API and clean up resources.
		"""
		if self.api_thread is None or not self.api_thread.is_alive():
			self.logger.warning("Call was made to close API but API server is not running!")
			return False

		self.logger.info(f"Stopping API server on `http://{self.api_host}:{self.api_port}`!")
		self.api_server.should_exit = True
		self.api_thread.join()
		return True
	
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

	def initializeModel(self):
		"""
		Initialize the model manager and load the base model.
		"""
		self.logger.info(f"Initializing model manager and loading base model!")
		self.modelManager: HSSFLDON_ModelManager = HSSFLDON_ModelManager(modelId=self.modelName)
		self.globalAdapter: PeftModel = self.modelManager.getFreshModel()
		self.modelManager.saveAdapterToFile(self.globalAdapter, self.adaptersGlobalFullPath)

	def shutdownModel(self):
		"""
		Clean up model resources.
		"""
		self.logger.info(f"Shutting down model and cleaning up resources!")
		del self.globalAdapter
		del self.modelManager
		torch.cuda.empty_cache()