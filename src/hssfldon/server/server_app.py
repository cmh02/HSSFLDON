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
import torch.nn.functional as F
from sklearn.cluster import KMeans
from datasets import load_dataset, Dataset
from fastapi import FastAPI
from dotenv import load_dotenv
from safetensors.torch import save_file, load_file
from torch.utils.data import DataLoader


# Project Imports
from hssfldon.common.hssfldon_logger import HSSFLDON_Logger
from hssfldon.common.hssfldon_enum import HSSFLDON_ServerState, HSSFLDON_ClientTask, HSSFLDON_PredictionOutputType
from hssfldon.server.server_api import HSSFLDON_ServerAPIRouter
from hssfldon.common.hssfldon_model import HSSFLDON_ModelManager
from hssfldon.common.hssfldon_data import HSSFLDON_DataLoader

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

		# Intialize index-category mapping
		self.indexCategoryMapping: dict[int, str] = {}
		self.indexCategoryMapping[0] = "sentiment"
		self.indexCategoryMapping[1] = "respect"
		self.indexCategoryMapping[2] = "insult"
		self.indexCategoryMapping[3] = "humiliate"
		self.indexCategoryMapping[4] = "status"
		self.indexCategoryMapping[5] = "dehumanize"
		self.indexCategoryMapping[6] = "violence"
		self.indexCategoryMapping[7] = "genocide"
		self.indexCategoryMapping[9] = "attack_defend"
		self.indexCategoryMapping[10] = "hatespeech"

		# Initialize client oracle mapping
		self.clientOracleMapping: dict[int, list[int]] = {}
		self.clientOracleMapping[0] = [2, 5, 6]
		self.clientOracleMapping[1] = [7, 9, 1]
		self.clientOracleMapping[2] = [0, 3, 10]

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
		self.unlabeledDataset: Dataset | None = HSSFLDON_DataLoader().loadDataset(path=self.dataFilePath, split="train")
		if self.unlabeledDataset is None:
			self.logger.error(f"Failed to load dataset from path: {self.dataFilePath}")
			raise ValueError(f"Dataset loading failed for path: {self.dataFilePath}")
		self.unlabeledDataloader: torch.utils.data.DataLoader = modelManager.tokenize_and_create_dataloader(
			texts = self.unlabeledDataset["text"],
			labels=None
		)

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
			modelManager: HSSFLDON_ModelManager = HSSFLDON_ModelManager(customHeadIdentifier=f"global")

			# # Passive Learning: Tell clients to perform passive learning
			# self.enterState(HSSFLDON_ServerState.PASSIVE_LEARNING)
			# self.logger.info(f"Performing passive learning for iteration {iteration+1}/{self.learningIterations}!")
			# for clientId in self.clients:

			# 	# Assign passive learning task to client
			# 	self.clientTasks[clientId] = HSSFLDON_ClientTask.DO_PASSIVE_LEARNING
			# 	self.clientUpdateStatus[clientId] = False

			# 	# Wait for client update
			# 	self.logger.debug(f"Waiting for update from client {clientId}!")
			# 	while not self.clientUpdateStatus[clientId]:
			# 		time.sleep(1)

			# 	# Tell client to standby until next iteration
			# 	self.clientTasks[clientId] = HSSFLDON_ClientTask.STANDBY
			# 	self.logger.debug(f"Received update from client {clientId} and set to standby!")

			# # Passive Aggregation: Aggregate client updates into global model for passive learning
			# self.enterState(HSSFLDON_ServerState.AGGREGATING)
			# self.logger.info(f"Aggregating client updates for iteration {iteration+1}/{self.learningIterations}!")
			# avgStateDict = self._fedAverageClientUpdates(modelManager=modelManager, clientHeadPaths=self.clientHeadPathCache)
			# if avgStateDict is not None:
			# 	modelManager.component_head.load_state_dict(avgStateDict)
			# 	modelManager.saveClassificationHead(head=modelManager.component_head, name=f"classification_head_global.pt")
			# 	self.logger.info(f"Successfully aggregated client updates for iteration {iteration+1}/{self.learningIterations}!")
			# else:
			# 	self.logger.warning(f"Failed to aggregate client updates for iteration {iteration+1}/{self.learningIterations}!")

			# Active Preparation: Determine finalist datapoints to send to clients
			self.enterState(HSSFLDON_ServerState.ACTIVE_LEARNING)
			self.logger.info(f"Preparing for active learning for iteration {iteration+1}/{self.learningIterations}!")
			probabilities, _ = modelManager.predict(
				dataLoader=self.unlabeledDataloader,
				outputType=HSSFLDON_PredictionOutputType.PROBABILITY_PREDICTION
			)
			self.logger.info(f"Calculated probabilities for unlabeled dataset for iteration {iteration+1}/{self.learningIterations}!")
			activeDataloader = self.unlabeledDataloader.dataset.add_column("probabilities", probabilities)
			finalistDataloader = self._getFinalistDatapointsForActiveLearning(
				modelManager=modelManager,
				dataLoader=activeDataloader,
				confidenceThreshold = float(os.getenv("HSSFLDON_ACTIVE_LEARNING_CONFIDENCE", 0.5)),
				numFinalists=int(os.getenv("HSSFLDON_ACTIVE_LEARNING_NUM_FINALISTS", 10)),
				numCentroids=int(os.getenv("HSSFLDON_ACTIVE_LEARNING_NUM_CENTROIDS", 5))
			)
			self.logger.info(f"Selected finalist datapoints for active learning for iteration {iteration+1}/{self.learningIterations}!")

			# For now, print off finalist datapoitns
			self.logger.info(f"Finalist datapoints for active learning in iteration {iteration+1}/{self.learningIterations}:")
			for datapoint in finalistDataloader:
				self.logger.info(f"Datapoint: {datapoint}")

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
			clientValues = [stateDict[key] for stateDict in clientStateDicts if key in stateDict.keys()]
			if clientValues:
				avgStateDict[key] = torch.mean(torch.stack(clientValues), dim=0)
			else:
				self.logger.warning(f"No client values found for key `{key}` during aggregation!")
				avgStateDict[key] = modelManager.component_head.state_dict()[key]

		# Load the averaged state dict into the global model
		return avgStateDict
	
	def _getFinalistDatapointsForActiveLearning(self, modelManager: HSSFLDON_ModelManager, dataLoader: DataLoader, confidenceThreshold: float, numFinalists: int, numCentroids: int) -> DataLoader:
		"""
		Get the finalist datapoints to send to clients for active learning based on confidence threshold.

		To calculate this, we seek to find which datapoints both best represent the unlabeled dataset and
		are the most uncertain for the model. To do this, we calculate the which datapoints among the
		unconfident datapoints are closest to the centroids of the unlabeled dataset while farthest from
		the centroids of the labeled dataset.
		"""
		self.logger.debug(f"Getting finalist datapoints for active learning with confidence threshold {confidenceThreshold}, num finalists {numFinalists}, and num centroids {numCentroids}!")

		# Get confident vs unconfident datapoints based on confidence threshold
		confidentDatapoints = []
		unconfidentDatapoints = []
		for datapoint in dataLoader:
			if max(datapoint["probabilities"]) >= confidenceThreshold:
				confidentDatapoints.append(datapoint)
			else:
				unconfidentDatapoints.append(datapoint)

		# If we don't have any unconfident datapoints, just return empty dataloader
		if not unconfidentDatapoints:
			self.logger.warning(f"No unconfident datapoints found with confidence threshold {confidenceThreshold}! Returning empty dataloader for active learning finalists!")
			return DataLoader([])

		# If we have less unconfident datapoints than the number of finalists we want to send, just return all unconfident datapoints
		if len(unconfidentDatapoints) <= numFinalists:
			self.logger.warning(f"Number of unconfident datapoints ({len(unconfidentDatapoints)}) is less than or equal to the number of finalists requested ({numFinalists})! Returning all unconfident datapoints for active learning finalists!")
			return DataLoader(unconfidentDatapoints)

		# Turn unconfident and confident datapoints into dataloaders for processing
		unconfidentDataloader = DataLoader(unconfidentDatapoints, batch_size=32)
		confidentDataloader = DataLoader(confidentDatapoints, batch_size=32)

		# Get centroids for unconfident datapoints and confident datapoints
		unconfidentEmbeddings, unconfidentCentroids = self._getEmbeddingsAndCentroids(modelManager=modelManager, dataLoader=unconfidentDataloader, numCentroids=numCentroids)
		confidentEmbeddings, confidentCentroids = self._getEmbeddingsAndCentroids(modelManager=modelManager, dataLoader=confidentDataloader, numCentroids=numCentroids)

		# Add embeddings to dataloaders
		for i in range(len(unconfidentDatapoints)):
			unconfidentDatapoints[i]["embeddings"] = unconfidentEmbeddings[i].cpu()
			
		for i in range(len(confidentDatapoints)):
			confidentDatapoints[i]["embeddings"] = confidentEmbeddings[i].cpu()

		# Calculate C-score for each unconfident datapoint
		unconfidentWithCScores = self._calculateCScores(
			modelManager=modelManager,
			unconfidentDataLoader=unconfidentDataloader,
			unconfidentCentroids=unconfidentCentroids,
			confidentCentroids=confidentCentroids
		)

		# Get top datapoints with highest C-scores
		finalistDataLoader = self._getTopCScores(
			dataLoader=unconfidentWithCScores,
			numFinalists=numFinalists
		)
		return finalistDataLoader


	def _getEmbeddingsAndCentroids(self, modelManager: HSSFLDON_ModelManager, dataLoader: DataLoader, numCentroids: int) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Gets the embeddings and centroids for a given dataloader using KMeans clustering on model embeddings.
		"""
		self.logger.debug(f"Calculating {numCentroids} centroids for dataloader with {len(dataLoader)} datapoints!")

		# Get embeddings for each datapoint using model
		embeddings, _ = modelManager.predict(
			dataLoader=dataLoader,
			outputType=HSSFLDON_PredictionOutputType.EMBEDDING_PREDICTION
		)

		# Perform KMeans clustering on embeddings to get centroids
		kmeans = KMeans(n_clusters=numCentroids, random_state=0)
		kmeans.fit(embeddings.cpu().numpy())
		centroids = kmeans.cluster_centers_

		# Return centroids
		return embeddings, torch.from_numpy(centroids)
	
	def _calculateCScores(self, modelManager: HSSFLDON_ModelManager, unconfidentDataLoader: DataLoader, unconfidentCentroids: torch.Tensor, confidentCentroids: torch.Tensor) -> DataLoader:
		"""
		Calculate C-score as described in the paper.
		"""
		self.logger.debug(f"Calculating C-scores for unconfident candidates!")

		# Batch embeddings and send to device
		allEmbeddings = []
		with torch.no_grad():
			for batch in unconfidentDataLoader:
				embeddings = batch["embeddings"].to(modelManager.device)
				allEmbeddings.append(embeddings)

		# Concat embeddings into tensor
		allEmbeddings = torch.cat(allEmbeddings, dim=0)

		# Calculate cosine similarities to each centroid
		unconfidentCentroidSimilarities = F.cosine_similarity(
			x1=allEmbeddings.unsqueeze(1), 
			x2=unconfidentCentroids.unsqueeze(0),
			dim=2
		)
		confidentCentroidSimilarities = F.cosine_similarity(
			x1=allEmbeddings.unsqueeze(1), 
			x2=confidentCentroids.unsqueeze(0),
			dim=2
		)

		# Calculate average similarities for each datapoint
		averageUnconfidentSimilarity = unconfidentCentroidSimilarities.mean(dim=1)
		averageConfidentSimilarity = confidentCentroidSimilarities.mean(dim=1)

		# Compute final C-scores as avg(unconfident) - avg(confident) for each datapoint
		cScores = averageUnconfidentSimilarity - averageConfidentSimilarity
		return unconfidentDataLoader.dataset.add_column("c_scores", cScores.cpu())
	
	def _getTopCScores(self, dataLoader: DataLoader, numFinalists: int) -> DataLoader:
		"""
		Get the top N datapoints with the highest C-scores from the dataloader.
		"""
		self.logger.debug(f"Getting top {numFinalists} datapoints with highest C-scores!")

		# Sort dataloader by C-score and get top N
		sortedData = sorted(dataLoader.dataset, key=lambda x: x["c_scores"], reverse=True)
		topData = sortedData[:numFinalists]
		return DataLoader(topData)