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
import gc
import time
import torch
import requests
from dotenv import load_dotenv
from transformers import TrainingArguments
from datasets import load_dataset, Dataset

# Project Imports
from hssfldon.common.hssfldon_data import HSSFLDON_DataLoader
from hssfldon.common.hssfldon_logger import HSSFLDON_Logger, HSSFLDON_TrainerCallbackLogger
from hssfldon.common.hssfldon_model import HSSFLDON_ModelManager
from hssfldon.common.hssfldon_enum import HSSFLDON_ClientState, HSSFLDON_ClientTask


class HSSFLDON_ClientApplication:
	"""
	The main client application for HSSFLDON.
	"""
	def __init__(self):

		# Load env file
		envStatus: bool = load_dotenv()
		if envStatus is False:
			print(f"Warning: .env file not found or failed to load. Make sure to create a .env file with the necessary configuration variables!")
 
		# Get client ID from env
		self.client_id: int = int(os.getenv("HSSFLDON_CLIENT_ID", f"{os.getpid()}"))

		# Get data directory from env and make path for client
		self.dataDirectory: str = os.getenv("HSSFLDON_CLIENT_DATA_DIRECTORY", "data")
		self.dataPath: str = os.path.join(self.dataDirectory, f"clients/Client_{self.client_id}.parquet")
		if not os.path.exists(self.dataPath):
			self.logger.error(f"Data file missing: {self.dataPath}")
			raise FileNotFoundError(f"Cannot train without data: {self.dataPath}")

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

		# Start main client loop
		if self.registered:
			self.doClientLoop()

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
			elif (task == HSSFLDON_ClientTask.DO_PASSIVE_LEARNING):
				self.logger.debug(f"Client received DO_PASSIVE_LEARNING task. Starting passive learning process!")
				self.doPassiveLearning()
				time.sleep(60)
				continue

	def doPassiveLearning(self):
		"""
		Perform the passive learning process for the client.
		"""
		self.logger.info(f"Starting passive learning process!")

		# Create new model manager (model) for this round
		modelManager: HSSFLDON_ModelManager = HSSFLDON_ModelManager(customHeadIdentifier=f"client_{self.client_id}")

		# Get dataset for this client
		dataset: Dataset | None = HSSFLDON_DataLoader().loadDataset(path=self.dataPath, split="train")
		if dataset is None:
			self.logger.error(f"Failed to load dataset for passive learning. Aborting this round of passive learning!")
			return

		# Tokenize dataset and prepare dataloader
		dataloader: torch.utils.data.DataLoader = modelManager.tokenize_and_create_dataloader(
			texts = dataset["text"],
			labels = dataset["labels"]
		)

		# Train model on dataset
		epochTrainingHistory = modelManager.train(
			trainingDataLoader = dataloader,
			validationDataLoader = None,
			epochs = 1,
			learningRate = 0.0001,
			weightDecay = 0.01,
			maxGradientNorm = 1 
		)

		# Log training history
		self.logger.debug(f"Passive learning training history for this round: {epochTrainingHistory}")

		# Save classifier head and submit update to server
		modelManager.saveClassificationHead(name=f"client_{self.client_id}_head.pt")
		self.submitUpdateToServer(headPath=f"client_{self.client_id}_head.pt")

		# Final info log and cleanup
		self.logger.info(f"Completed passive learning process for this round!")
		del modelManager
		gc.collect()
		torch.cuda.empty_cache()

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
			task: HSSFLDON_ClientTask = HSSFLDON_ClientTask[f"{data.get('task')}"]
			self.logger.info(f"Received task from server: {task.name}")
			return task
		except Exception as e:
			self.logger.error(f"An exception occurred while fetching next task: {e}")
		return HSSFLDON_ClientTask.STANDBY
	
	def submitUpdateToServer(self, headPath: str):
		"""
		Submit the updated local head to the server.

		Args:
			headPath: The file path to the updated local head.
		"""
		self.logger.debug(f"Submitting updated head to server from path: {headPath}")
		
		# Build payload
		payload = {
			"client_id": self.client_id,
			"head_path": headPath
		}
		headers = {"Content-Type": "application/json"}

		# Send it off
		try:
			response: requests.Response = requests.post(
				f"{self.server_api_url}/submit_update",
				json=payload,
				headers=headers
			)
			response.raise_for_status()
			data: dict = response.json()
			self.logger.info(f"Successfully submitted update to server! Server response: {data.get('message')}")
		except Exception as e:
			self.logger.error(f"An exception occurred while submitting update to server: {e}")