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
import json
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
		self.clientActiveLearningDatapointCache: dict[int, dict] = {}

		# Initialize model evaluation tracking {learning_iteration: {passive_or_active: {metric_name: metric_value}}}}
		self.modelEvaluationHistory: dict[int, dict[str, dict[str, float]]] = {}

		# Intialize index-category mapping
		self.indexToCategoryMapping: dict[int, str] = {}
		self.indexToCategoryMapping[0] = "sentiment"
		self.indexToCategoryMapping[1] = "respect"
		self.indexToCategoryMapping[2] = "insult"
		self.indexToCategoryMapping[3] = "humiliate"
		self.indexToCategoryMapping[4] = "status"
		self.indexToCategoryMapping[5] = "dehumanize"
		self.indexToCategoryMapping[6] = "violence"
		self.indexToCategoryMapping[7] = "genocide"
		self.indexToCategoryMapping[9] = "attack_defend"
		self.indexToCategoryMapping[10] = "hatespeech"
		self.categoryToIndexMapping: dict[str, int] = {v: k for k, v in self.indexToCategoryMapping.items()}

		# Initialize category oracle mapping
		self.categoryToOracleMapping: dict[int, list[int]] = {}
		self.categoryToOracleMapping[0] = [2, 5, 6]
		self.categoryToOracleMapping[1] = [7, 9, 1]
		self.categoryToOracleMapping[2] = [0, 3, 10]
		self.oracleToCategoryMapping: dict[int, int] = {}
		for clientId, oracleCategories in self.categoryToOracleMapping.items():
			for categoryIndex in oracleCategories:
				self.oracleToCategoryMapping[categoryIndex] = clientId

		# Initialize 

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

		# Setup test data directory for gloabl testing data
		self.testDataFilePath: str = os.path.join(self.dataDirectory, f"test/test.parquet")
		if not os.path.exists(self.testDataFilePath):
			self.logger.error(f"Test data file missing: {self.testDataFilePath}")
			raise FileNotFoundError(f"Cannot evaluate without test data: {self.testDataFilePath}")
		self.testDataset: Dataset | None = HSSFLDON_DataLoader().loadDataset(path=self.testDataFilePath, split="train")
		if self.testDataset is None:
			self.logger.error(f"Failed to load test dataset from path: {self.testDataFilePath}")
			raise ValueError(f"Test dataset loading failed for path: {self.testDataFilePath}")
		self.testDataloader: torch.utils.data.DataLoader = modelManager.tokenize_and_create_dataloader(
			texts = self.testDataset["text"],
			labels = self.testDataset["labels"]
		)

		# Setup evaluation results directory
		self.evaluationResultsDirectory: str = os.getenv("HSSFLDON_EVALUATION_RESULTS_DIRECTORY", "results")
		os.makedirs(self.evaluationResultsDirectory, exist_ok=True)

		# Begin server loop for configured iterations
		self.learningIterations: int = int(os.getenv("HSSFLDON_LEARNING_ITERATIONS", 10))
		self.doLearningLoop()

		# Close API and shutdown everything (for now)
		self.closeApi()

	def doLearningLoop(self) -> bool:
		"""
		Main loop for the server application to perform learning iterations.
		"""
		self.logger.info(f"Starting main server learning loop for {self.learningIterations} iterations!")
		for iteration in range(self.learningIterations):
			self.logger.info(f"Starting learning iteration {iteration+1}/{self.learningIterations}!")
			modelManager: HSSFLDON_ModelManager = HSSFLDON_ModelManager(customHeadIdentifier=f"global")
			self.modelEvaluationHistory[iteration+1] = {}

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
			avgStateDict = self._fedAverageClientUpdates(modelManager=modelManager, clientHeadPaths=self.clientHeadPathCache)
			if avgStateDict is not None:
				modelManager.component_head.load_state_dict(avgStateDict)
				modelManager.saveClassificationHead(head=modelManager.component_head, name=f"classification_head_global.pt")
				self.logger.info(f"Successfully aggregated client updates for iteration {iteration+1}/{self.learningIterations}!")
			else:
				self.logger.warning(f"Failed to aggregate client updates for iteration {iteration+1}/{self.learningIterations}!")

			# Evaluate global model on test dataset
			self.enterState(HSSFLDON_ServerState.EVALUATING)
			self.logger.info(f"Evaluating global model on test dataset for iteration {iteration+1}/{self.learningIterations}!")
			testLoss, testAccuracy = modelManager.evaluate(
				dataLoader=self.testDataloader
			)
			self.logger.debug(f"Global evaluation completed after passive learning; Test Loss: {testLoss}, Test Accuracy: {testAccuracy}")
			self.modelEvaluationHistory[iteration+1]["passive"] = {
				"loss": testLoss,
				"accuracy": testAccuracy
			}

			# Active Preparation: Determine finalist datapoints to send to clients
			self.enterState(HSSFLDON_ServerState.ACTIVE_LEARNING)
			self.logger.info(f"Preparing for active learning for iteration {iteration+1}/{self.learningIterations}!")
			allUnlabeledCandidatesDataloader = modelManager.tokenize_and_create_dataloader(
				texts = self.unlabeledDataset["text"],
				labels=None
			)
			output, _ = modelManager.predict(
				dataLoader=allUnlabeledCandidatesDataloader
			)
			self.logger.info(f"Calculated probabilities for unlabeled dataset for iteration {iteration+1}/{self.learningIterations}!")
			activeDataSet = allUnlabeledCandidatesDataloader.dataset
			activeDataSet = activeDataSet.add_column("probabilities", output[HSSFLDON_PredictionOutputType.PROBABILITY_PREDICTION])
			activeDataSet = activeDataSet.add_column("embeddings", output[HSSFLDON_PredictionOutputType.EMBEDDING_PREDICTION])
			activeDataloader = DataLoader(activeDataSet, batch_size=32)}")
			finalistDataloader = self._getFinalistDatapointsForActiveLearning(
				modelManager=modelManager,
				dataloader=activeDataloader,
				confidenceThreshold = float(os.getenv("HSSFLDON_ACTIVE_LEARNING_CONFIDENCE", 0.5)),
				numFinalists=int(os.getenv("HSSFLDON_ACTIVE_LEARNING_NUM_FINALISTS", 10)),
				numCentroids=int(os.getenv("HSSFLDON_ACTIVE_LEARNING_NUM_CENTROIDS", 5))
			)
			self.logger.info(f"Selected finalist datapoints for active learning for iteration {iteration+1}/{self.learningIterations}!")

			# Verify we have finalist datapoints
			if len(finalistDataloader.dataset) == 0:
				self.logger.warning(f"Attempted to perform Active Learning but no valid final candidates were produced! Skipping active learning for this iteration!")
				continue

			# Active Assignment: Determine which clients are best for each data point based on oracle mapping
			for datapoint in finalistDataloader.dataset:
				maxCategoryIndex = torch.argmax(datapoint["probabilities"]).item()
				oracleId = self.oracleToCategoryMapping.get(maxCategoryIndex)
				if self.clientActiveLearningDatapointCache.get(oracleId) is None:
					self.clientActiveLearningDatapointCache[oracleId] = datapoint["text"]
					self.logger.info(f"Assigned active learning datapoint to client {oracleId} based on oracle mapping for category index {maxCategoryIndex}!")
				else:
					self.logger.info(f"Client {oracleId} already has an active learning datapoint assigned, skipping additional datapoint assignment for category index {maxCategoryIndex}!")

			# Active Learning: Tell clients to perform active learning with assigned datapoints
			for clientId in self.clients:

				# Make sure client will have a datapoint
				if self.clientActiveLearningDatapointCache.get(clientId) is None:
					self.logger.info(f"No active learning datapoint assigned to client {clientId} for iteration {iteration+1}/{self.learningIterations}, skipping active learning task assignment for this client!")
					continue

				# Assign active learning task to client
				self.clientTasks[clientId] = HSSFLDON_ClientTask.DO_ACTIVE_LEARNING
				self.clientUpdateStatus[clientId] = False

				# Wait for client update
				self.logger.debug(f"Waiting for update from client {clientId}!")
				while not self.clientUpdateStatus[clientId]:
					time.sleep(1)

				# Tell client to standby until next iteration
				self.clientTasks[clientId] = HSSFLDON_ClientTask.STANDBY
				self.logger.debug(f"Received update from client {clientId} and set to standby!")

			# Clear active learning datapoint cache for next iteration
			for clientId in self.clientActiveLearningDatapointCache.keys():
				self.clientActiveLearningDatapointCache[clientId] = None

			# Active Aggregation: Aggregate client updates into global model for active learning
			self.enterState(HSSFLDON_ServerState.AGGREGATING)
			self.logger.info(f"Aggregating client updates for iteration {iteration+1}/{self.learningIterations}!")
			avgStateDict = self._fedAverageClientUpdates(modelManager=modelManager, clientHeadPaths=self.clientHeadPathCache)
			if avgStateDict is not None:
				modelManager.component_head.load_state_dict(avgStateDict)
				modelManager.saveClassificationHead(head=modelManager.component_head, name=f"classification_head_global.pt")
				self.logger.info(f"Successfully aggregated client updates for iteration {iteration+1}/{self.learningIterations}!")
			else:
				self.logger.warning(f"Failed to aggregate client updates for iteration {iteration+1}/{self.learningIterations}!")	

			# Evaluate global model on test dataset
			self.enterState(HSSFLDON_ServerState.EVALUATING)
			self.logger.info(f"Evaluating global model on test dataset for iteration {iteration+1}/{self.learningIterations}!")
			testLoss, testAccuracy = modelManager.evaluate(
				dataLoader=self.testDataloader
			)
			self.logger.debug(f"Global evaluation completed after active learning; Test Loss: {testLoss}, Test Accuracy: {testAccuracy}")
			self.modelEvaluationHistory[iteration+1]["active"] = {
				"loss": testLoss,
				"accuracy": testAccuracy
			}
		
		# Save model evaluation history to file after all iterations are complete
		evaluationHistoryPath = os.path.join(self.evaluationResultsDirectory, f"model_evaluation_history.json")
		with open(evaluationHistoryPath, "w") as f:
			json.dump(self.modelEvaluationHistory, f)
		self.logger.info(f"Saved model evaluation history to {evaluationHistoryPath} after completing all learning iterations!")

		# Return to idle state after learning loop is complete
		self.enterState(HSSFLDON_ServerState.IDLE)
		return True

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
			clientValues = [stateDict[key] for clientId, stateDict in clientStateDicts.items() if key in stateDict.keys()]
			if clientValues:
				avgStateDict[key] = torch.mean(torch.stack(clientValues), dim=0)
			else:
				self.logger.warning(f"No client values found for key `{key}` during aggregation!")
				avgStateDict[key] = modelManager.component_head.state_dict()[key]

		# Load the averaged state dict into the global model
		return avgStateDict
	
	def _getFinalistDatapointsForActiveLearning(self, modelManager: HSSFLDON_ModelManager, dataloader: DataLoader, confidenceThreshold: float, numFinalists: int, numCentroids: int) -> DataLoader:
		"""
		Get the finalist datapoints to send to clients for active learning based on confidence threshold.

		To calculate this, we seek to find which datapoints both best represent the unlabeled dataset and
		are the most uncertain for the model. To do this, we calculate the which datapoints among the
		unconfident datapoints are closest to the centroids of the unlabeled dataset while farthest from
		the centroids of the labeled dataset.
		"""
		self.logger.debug(f"Getting finalist datapoints for active learning with confidence threshold {confidenceThreshold}, num finalists {numFinalists}, and num centroids {numCentroids}!")

		# Calculate confidence based on average deviation from rounded probabilities for each class
		probabilities = torch.stack([dp["probabilities"] for dp in dataloader.dataset])
		deviations = torch.mean(torch.abs(probabilities - torch.round(probabilities)), dim=1)
		mask_isConfident = deviations <= confidenceThreshold

		# Separate confident and unconfident datapoints
		confidentDatapoints = [dp for dp, conf in zip(dataloader.dataset, mask_isConfident) if conf]
		unconfidentDatapoints = [dp for dp, conf in zip(dataloader.dataset, mask_isConfident) if not conf]

		# If we don't have any unconfident datapoints, just return empty dataloader
		if len(unconfidentDatapoints) == 0:
			self.logger.warning(f"No unconfident datapoints found with confidence threshold {confidenceThreshold}! Returning empty dataloader for active learning finalists!")
			self.logger.debug(f"Num confident datapoints: {len(confidentDatapoints)}, Num unconfident datapoints: {len(unconfidentDatapoints)}")
			return DataLoader([])

		# If we have less unconfident datapoints than the number of finalists we want to send, just return all unconfident datapoints
		if len(unconfidentDatapoints) <= numFinalists:
			self.logger.warning(f"Number of unconfident datapoints ({len(unconfidentDatapoints)}) is less than or equal to the number of finalists requested ({numFinalists})! Returning all unconfident datapoints for active learning finalists!")
			return DataLoader(unconfidentDatapoints)

		# Turn unconfident and confident datapoints into dataloaders for processing
		unconfidentDataloader = DataLoader(unconfidentDatapoints, batch_size=32)
		confidentDataloader = DataLoader(confidentDatapoints, batch_size=32)

		# Get centroids for unconfident datapoints and confident datapoints
		unconfidentCentroids = self._getCentroids(modelManager=modelManager, dataLoader=unconfidentDataloader, numCentroids=numCentroids)
		
		if len(confidentDatapoints) > 0:
			confidentCentroids = self._getCentroids(modelManager=modelManager, dataLoader=confidentDataloader, numCentroids=numCentroids)

		# Calculate C-score for each unconfident datapoint
		unconfidentWithCScores = self._calculateCScores(
			modelManager=modelManager,
			unconfidentDataLoader=unconfidentDataloader,
			unconfidentCentroids=unconfidentCentroids,
			confidentCentroids=confidentCentroids if len(confidentDatapoints) > 0 else torch.empty((0, unconfidentCentroids.shape[1]))
		)

		# Get top datapoints with highest C-scores
		finalistDataLoader = self._getTopCScores(
			dataLoader=unconfidentWithCScores,
			numFinalists=numFinalists
		)
		return finalistDataLoader

	def _getCentroids(self, modelManager: HSSFLDON_ModelManager, dataLoader: DataLoader, numCentroids: int) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Gets the centroids for a given dataloader using KMeans clustering on model embeddings.
		"""
		self.logger.debug(f"Calculating {numCentroids} centroids for dataloader with {len(dataLoader)} datapoints!")

		# Get all embeddings from dataloader and send to device
		embeddings = []
		with torch.no_grad():
			for batch in dataLoader:
				batchEmbeddings = batch["embeddings"].to(modelManager.device)
				embeddings.append(batchEmbeddings)
		embeddings = torch.cat(embeddings, dim=0)

		# Verify that we have enough datapoints to calculate the requested number of centroids
		if embeddings.shape[0] < numCentroids:
			self.logger.warning(f"Number of datapoints ({embeddings.shape[0]}) is less than the number of centroids requested ({numCentroids})! Reducing number of centroids to {embeddings.shape[0]}!")
			numCentroids = embeddings.shape[0]

		# Perform KMeans clustering on embeddings to get centroids
		kmeans = KMeans(n_clusters=numCentroids, random_state=0)
		kmeans.fit(embeddings.cpu().numpy())
		centroids = kmeans.cluster_centers_

		# Return centroids
		return torch.from_numpy(centroids)
	
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

		# Ensure centroids are on same device as embeddings
		unconfidentCentroids = unconfidentCentroids.to(modelManager.device)
		if len(confidentCentroids) > 0:
			confidentCentroids = confidentCentroids.to(modelManager.device)

		# Calculate cosine similarities to each centroid
		unconfidentCentroidSimilarities = F.cosine_similarity(
			x1=allEmbeddings.unsqueeze(1), 
			x2=unconfidentCentroids.unsqueeze(0),
			dim=2
		)
		if len(confidentCentroids) > 0:
			confidentCentroidSimilarities = F.cosine_similarity(
				x1=allEmbeddings.unsqueeze(1), 
				x2=confidentCentroids.unsqueeze(0),
				dim=2
			)

		# Calculate average similarities for each datapoint
		averageUnconfidentSimilarity = unconfidentCentroidSimilarities.mean(dim=1)
		if len(confidentCentroids) > 0:
			averageConfidentSimilarity = confidentCentroidSimilarities.mean(dim=1)

		# Compute final C-scores as avg(unconfident) - avg(confident) for each datapoint
		if len(confidentCentroids) > 0:
			cScores = averageUnconfidentSimilarity - averageConfidentSimilarity
		else:
			cScores = averageUnconfidentSimilarity
		for i in range(len(cScores)):
			unconfidentDataLoader.dataset[i]["c_scores"] = cScores[i].cpu()
		return unconfidentDataLoader

	def _getTopCScores(self, dataLoader: DataLoader, numFinalists: int) -> DataLoader:
		"""
		Get the top N datapoints with the highest C-scores from the dataloader.
		"""
		self.logger.debug(f"Getting top {numFinalists} datapoints with highest C-scores!")

		# Sort dataloader by C-score and get top N
		sortedData = sorted(dataLoader.dataset, key=lambda x: x["c_scores"], reverse=True)
		topData = sortedData[:numFinalists]
		return DataLoader(topData)