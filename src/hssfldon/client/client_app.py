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
import ast
import json
import time
import unsloth
import torch
import requests
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv
from transformers import TrainingArguments
from datasets import load_dataset, Dataset

# Project Imports
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

		# Get model ID from env
		self.modelName: str  |  None = os.getenv("HSSFLDON_HF_MODEL", None)

		# Get data directory from env
		self.dataDirectory: str = os.getenv("HSSFLDON_CLIENT_DATA_DIRECTORY", "data")

		# Build adapters directory path
		self.adaptersDirectory: str = os.getenv("HSSFLDON_CLIENT_ADAPTERS_DIRECTORY", "adapters")
		self.adaptersDirectoryClientPrefix: str = os.getenv("HSSFLDON_MODEL_ADAPTERS_CLIENT", "clients")
		self.adaptersDirectoryClientPath: str = os.path.join(self.adaptersDirectory, self.adaptersDirectoryClientPrefix, f"Client_{self.client_id}")

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
				modelManager = HSSFLDON_ModelManager(modelId=self.modelName)
				self.doPassiveLearning(modelManager=modelManager)
				del modelManager
				torch.cuda.empty_cache()
				gc.collect()
				time.sleep(60)
				continue


	def doPassiveLearning(self, modelManager: HSSFLDON_ModelManager):
		"""
		Perform the passive learning process for the client.
		"""
		self.logger.info(f"Starting passive learning process!")

		# Get max token length from env
		maxTokenLength: int = int(os.getenv("HSSFLDON_MAX_TOKEN_LENGTH", 512))

		# Get the global adapter path and load it
		globalAdapterPath: str = self.getGlobalAdapterPath()
		if not globalAdapterPath:
			self.logger.error(f"Failed to retrieve global adapter path from server. Cannot perform passive learning without global model!")
			return
		clientModel = modelManager.loadAdapterFromFile(globalAdapterPath)
		numTrainableParams = sum(p.numel() for p in clientModel.parameters() if p.requires_grad)
		self.logger.debug(f"Successfully loaded global adapter from `{globalAdapterPath}` with `{numTrainableParams}` trainable parameters!")

		# Load data
		try:
			trainDataset: Dataset = self.getData()
		except Exception as e:
			self.logger.error(f"Error loading training data: {e}")
			return

		# tokenizer = modelManager.tokenizer
		# print("Chat template:", tokenizer.chat_template)
		# print("EOS token:", tokenizer.eos_token)
		# print("PAD token:", tokenizer.pad_token)

		# # Try manually
		# sample = trainDataset[49]
		# result = tokenizer.apply_chat_template(
		# 	sample["messages"],
		# 	tokenize=False,  # Get the string first
		# 	add_generation_prompt=False
		# )
		# print("Rendered string:", result[:200])
		# print("Rendered length:", len(result))

		# Configure training arguments and train
		# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
		try:
			trainingArgs = SFTConfig(

				# Specify temporary output (if we had checkpoints)
				output_dir=f"./temp_outputs/client_{self.client_id}",

				# Batching
				per_device_train_batch_size=1,
				gradient_accumulation_steps=1,
				num_train_epochs=1,

				# Datatype
				# fp16=False,                         
				bf16=True,

				# Specify LR and other optimizer params
				learning_rate=5e-07,
				weight_decay=0.01,
				max_grad_norm=1.0,
				warmup_steps=0,
				lr_scheduler_type="constant",

				# Disable checkpoints and reduce logging
				save_strategy="no",
				logging_steps=10,

				# Max sequence length for qwen with our dataset
				max_length=maxTokenLength,

				# Specify eos token for qwen
				eos_token="<|im_end|>",

				# Disable progress bars to reduce clutter
				disable_tqdm=True,

				# Specify column for text
				dataset_text_field="text"

			)
			trainer = SFTTrainer(
				model=clientModel,
				processing_class=modelManager.tokenizer,
				train_dataset=trainDataset,
				args=trainingArgs,
				callbacks=[HSSFLDON_TrainerCallbackLogger(logger=self.logger)]
			)
			trainer.train()
			
		except Exception as e:
			self.logger.error(f"Error during training: {e}")
			return

		# Save adapter
		try:
			localAdapterPath = os.path.join(self.adaptersDirectoryClientPath)
			modelManager.saveAdapterToFile(clientModel, localAdapterPath)
			self.logger.debug(f"Saved client adapter to `{localAdapterPath}`!")
		except Exception as e:
			self.logger.error(f"Error saving local adapter: {e}")
			return

		# Cleanup
		try:
			del clientModel
			del trainer
		except Exception as e:
			self.logger.error(f"Error during cleanup after training: {e}")

		# Send update to server
		try:
			self.logger.info("Passive learning complete. Notifying server.")
			self.submitUpdateToServer(localAdapterPath)
		except Exception as e:
			self.logger.error(f"Error submitting update to server: {e}")
			return

	def getData(self) -> Dataset:
		"""
		Loads pre-processed training data for this specific client from a Parquet file.
		Returns a Hugging Face Dataset object formatted for the SFTTrainer.
		"""

		# Make path
		dataPath: str = os.path.join(self.dataDirectory, f"clients/Client_{self.client_id}.parquet")
		if not os.path.exists(dataPath):
			self.logger.error(f"Data file missing: {dataPath}")
			raise FileNotFoundError(f"Cannot train without data: {dataPath}")

		# Load dataset from Parquet file
		hf_dataset = load_dataset("parquet", data_files=dataPath, split="train")
		if len(hf_dataset) == 0:
			self.logger.error(f"Loaded dataset is empty from path: {dataPath}")
			raise ValueError(f"Cannot train on empty dataset: {dataPath}")
		
		# Remove uneeded columns for client training
		columns_to_remove = [col for col in hf_dataset.column_names if col != "slm_prompt_labeled"]
		hf_dataset = hf_dataset.remove_columns(columns_to_remove)

		# Rename the target column to "text"
		hf_dataset = hf_dataset.rename_column("slm_prompt_labeled", "text")

		# Remove all columns except for 'messages'
		# columns_to_remove = [col for col in hf_dataset.column_names if col != "messages"]
		# hf_dataset = hf_dataset.remove_columns(columns_to_remove)

		# Log and return
		self.logger.info(f"Loaded dataset with {len(hf_dataset)} examples from `{dataPath}`!")
		return hf_dataset

	def getGlobalAdapterPath(self):
		"""
		Get the global model adapter path from the server.
		"""
		self.logger.debug(f"Making request to fetch global model adapter path from server!")
		try:
			response: requests.Response = requests.get(f"{self.server_api_url}/global_model?client_id={self.client_id}")
			response.raise_for_status()
			self.logger.debug(f"Successfully fetched global model adapter path from server!")
			return response.json().get("adapter_path")
		except Exception as e:
			self.logger.error(f"An exception occurred while fetching global model adapter path: {e}")
		return None

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
			task: HSSFLDON_ClientTask = HSSFLDON_ClientTask[data.get("task")]
			self.logger.info(f"Received task from server: {task.name}")
			return task
		except Exception as e:
			self.logger.error(f"An exception occurred while fetching next task: {e}")
		return HSSFLDON_ClientTask.STANDBY
	
	def submitUpdateToServer(self, adapterPath: str):
		"""
		Submit the updated local adapter to the server.

		Args:
			adapterPath: The file path to the updated local adapter.
		"""
		self.logger.debug(f"Submitting updated adapter to server from path: {adapterPath}")
		
		# Build payload
		payload = {
			"client_id": self.client_id,
			"adapter_path": adapterPath
		}
		headers = {"Content-Type": "application/json"}


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