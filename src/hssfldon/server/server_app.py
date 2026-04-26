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
from datasets import load_dataset, Dataset
from fastapi import FastAPI
from dotenv import load_dotenv
from safetensors.torch import save_file, load_file



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
		self.clientUpdateStatus: dict[int, bool] = {}
		self.clientHeadPathCache: dict[int, str | None] = {}

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
		self.modelName: str = os.getenv("HSSFLDON_MODEL_NAME", "bert-base-uncased")
		modelManager: HSSFLDON_ModelManager = HSSFLDON_ModelManager(customHeadIdentifier=f"global")
		modelManager.saveBaseModel(model=modelManager.component_base, name = "pytorch_model.bin")
		modelManager.saveTokenizer(tokenizer=modelManager.tokenizer, name = "tokenizer.pt")
		modelManager.saveClassificationHead(head=modelManager.component_head, name = "classification_head_global.pt")

		# Setup data directory for server unlabeled data
		self.dataDirectory: str = os.getenv("HSSFLDON_CLIENT_DATA_DIRECTORY", "data")
		self.dataFilePath: str = os.path.join(self.dataDirectory, f"server/server.parquet")
		if not os.path.exists(self.dataFilePath):
			self.logger.error(f"Data file missing: {self.dataFilePath}")
			raise FileNotFoundError(f"Cannot train without data: {self.dataFilePath}")		

		# Begin server loop for configured iterations
		self.learningIterations: int = int(os.getenv("HSSFLDON_LEARNING_ITERATIONS", 10))
		self.doLearningLoop()

		# # Close API and shutdown everything (for now)
		# self.closeApi()

	def doLearningLoop(self):
		"""
		Main loop for the server application to perform learning iterations.
		"""
		self.logger.info(f"Starting main server learning loop for {self.learningIterations} iterations!")
		for iteration in range(self.learningIterations):
			self.logger.info(f"Starting learning iteration {iteration+1}/{self.learningIterations}!")

			# Passive Learning: Tell clients to perform passive learning
			self.enterState(HSSFLDON_ServerState.PASSIVE_LEARNING)
			self.logger.info(f"Performing passive learning for iteration {iteration+1}/{self.learningIterations}!")
			for clientId in self.clients:

				# Assign passive learning task to client
				self.clientTasks[clientId] = HSSFLDON_ClientTask.DO_PASSIVE_LEARNING
				self.clientUpdateStatus[clientId] = False

				# Wait for client update
				self.logger.debug(f"Waiting for update from client {clientId}!")
				while not self.clientUpdateStatus[clientId]:
					time.sleep(1)

				# Tell client to standby until next iteration
				self.clientTasks[clientId] = HSSFLDON_ClientTask.STANDBY
				self.logger.debug(f"Received update from client {clientId} and set to standby!")

			# Passive Aggregation: Aggregate client updates into global model for passive learning
			self.enterState(HSSFLDON_ServerState.AGGREGATING)
			self.logger.info(f"Aggregating client updates for iteration {iteration+1}/{self.learningIterations}!")
			modelManager: HSSFLDON_ModelManager = HSSFLDON_ModelManager(customHeadIdentifier=f"global")
			avgStateDict = self._fedAverageClientUpdates(modelManager=modelManager, clientHeadPaths=self.clientHeadPathCache)
			if avgStateDict is not None:
				modelManager.component_head.load_state_dict(avgStateDict)
				modelManager.saveClassificationHead(head=modelManager.component_head, name=f"classification_head_global.pt")
				self.logger.info(f"Successfully aggregated client updates for iteration {iteration+1}/{self.learningIterations}!")
			else:
				self.logger.warning(f"Failed to aggregate client updates for iteration {iteration+1}/{self.learningIterations}!")

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
		self.clientUpdateStatus[clientId] = False
		self.clientHeadPathCache[clientId] = None

	def _fedAverageClientUpdates(self, modelManager: HSSFLDON_ModelManager, clientHeadPaths: dict[int, str | None]) -> dict | None:
		"""
		Perform federated averaging on the client updates to create a new global model.

		Args:
			modelManager (HSSFLDON_ModelManager): The model manager for handling classification heads (should be global at server).
			clientHeadPaths (dict[int, str | None]): A dictionary mapping client IDs to their classification head file paths.
		"""
		self.logger.debug(f"Performing federated averaging on client updates: {clientHeadPaths}!")

		# Load each client head to get state dict
		clientStateDicts = {}
		for clientId, headPath in clientHeadPaths.items():
			if headPath is None:
				self.logger.warning(f"Client {clientId} does not have a valid head path for aggregation. Skipping!")
				continue

			try:
				clientHead = modelManager.loadClassificationHead(name=headPath)
				clientStateDicts[clientId] = clientHead.state_dict()
			except Exception as e:
				self.logger.error(f"Error loading classification head from client {clientId} at path `{headPath}`: {e}")
				continue

		# Make sure we got at least one valid client update before trying to aggregate
		if not clientStateDicts:
			self.logger.warning(f"No valid client updates available for aggregation!")
			return None
		
		# Perform federated averaging on the state dicts
		avgStateDict = {}
		for key in modelManager.component_head.state_dict().keys():
			clientValues = [stateDict[key] for stateDict in clientStateDicts if key in stateDict]
			if clientValues:
				avgStateDict[key] = torch.mean(torch.stack(clientValues), dim=0)
			else:
				self.logger.warning(f"No client values found for key `{key}` during aggregation!")
				avgStateDict[key] = modelManager.component_head.state_dict()[key]

		# Load the averaged state dict into the global model
		return avgStateDict