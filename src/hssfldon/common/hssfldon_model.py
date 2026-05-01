'''
	# HSSFLDON - Model

	This common module will provide the model for use by the server and clients in HSSFLDON.
'''

### Library Imports
import os
import copy
import torch
import torch.nn as nn
from typing import Tuple, Any, Dict
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, hamming_loss
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

# Project Imports
from hssfldon.common.hssfldon_logger import HSSFLDON_Logger
from hssfldon.common.hssfldon_enum import HSSFLDON_PredictionOutputType

class HSSFLDON_ClassifierHead(torch.nn.Module):
	"""
	A simple classification head for the base model.
	"""
	def __init__(self, base, head):
		super(HSSFLDON_ClassifierHead, self).__init__()
		self.base = base
		self.head = head
	
	def forward(self, **kwargs):
		out = self.base(**kwargs)
		pooled = out.last_hidden_state[:,0]
		headOut = self.head(pooled)

		# If we wanted hidden states, also output
		if kwargs.get("output_hidden_states", False):
			return headOut, out.hidden_states
		return headOut

class HSSFLDON_ModelManager:
	"""
	The main model class for HSSFLDON.
	"""
	def __init__(self, customHeadIdentifier: str | None = None):

		# Parse dotenv for env variables
		envStatus: bool = load_dotenv()
		if envStatus is False:
			print(f"Warning: .env file not found or failed to load. Make sure to create a .env file with the necessary configuration variables!")
			
		# Get logger
		self.logger: HSSFLDON_Logger = HSSFLDON_Logger(name=f"ModelManager {os.getpid()}")
		self.logger.debug(f"Initialized Model Manager with PID {os.getpid()}!")

		# Determine device
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# Get needed values from env
		self.modelDirectory: str = os.getenv("HSSFLDON_MODEL_DIRECTORY", "models")
		self.modelId: str = os.getenv("HSSFLDON_MODEL_ID", "microsoft/deberta-v3-small")
		self.modelNClasses: int = int(os.getenv("HSSFLDON_MODEL_N_CLASSES", 9))
		self.huggingFaceAccessToken: str | None = os.getenv("HSSFLDON_HF_ACCESS_TOKEN", None)
		self.confidenceThreshold: float = float(os.getenv("HSSFLDON_CLIENT_CONFIDENCE_THRESHOLD", 0.75))

		# Build class position weights based on distribution
		# self.classPositionWeights = [
		# 	float(os.getenv(f"HSSFLDON_CLASS_POSWEIGHT_{i}", 1.0)) for i in range(self.modelNClasses)
		# ]
		# self.classPositionWeights = torch.tensor(self.classPositionWeights, device=self.device)
		# self.classPositionWeights = torch.clamp(self.classPositionWeights, min=1.0, max=5.0)

		# Create static paths for model components
		self.modelPath_base: str = os.path.join(self.modelDirectory, f"model_base")
		self.modelPath_head: str = os.path.join(self.modelDirectory, f"model_head")
		self.modelPath_tokenizer: str = os.path.join(self.modelDirectory, f"model_tokenizer")

		# Login to HF if needed
		if self.huggingFaceAccessToken:
			login(token=self.huggingFaceAccessToken)
			self.logger.info(f"Logged in to Hugging Face Hub successfully!")
		else:
			self.logger.warning(f"No Hugging Face access token provided. If the model `{self.modelId}` is private, model loading will fail!")

		# Initialize model, tokenizer, and classification head
		self.component_base = self.loadBaseModel(name=f"pytorch_model.bin")
		if customHeadIdentifier is None:
			customHeadIdentifier = "default"
			self.logger.info(f"No custom head identifier provided; using default: `{customHeadIdentifier}`")
		self.component_head = self.loadClassificationHead(name=f"classification_head_{customHeadIdentifier}.pt")
		self.model = HSSFLDON_ClassifierHead(base=self.component_base, head=self.component_head)
		self.tokenizer = self.loadTokenizer(name=f"tokenizer.pt")

		# Save for reuse
		self.saveClassificationHead(head=self.component_head, name=f"classification_head_{customHeadIdentifier}.pt")
		self.saveBaseModel(model=self.component_base, name=f"pytorch_model.bin")
		self.saveTokenizer(tokenizer=self.tokenizer, name=f"tokenizer.pt")

	def loadBaseModel(self, name: str = "pytorch_model.bin") -> Any:
		"""
		Load the (frozen) base model from file or HF if needed.
		"""

		# Make full file name for file
		modelFile_base: str = os.path.join(self.modelPath_base, name)
		
		# Check if the model has been saved locally
		if os.path.isfile(modelFile_base):
			self.logger.info(f"Loading base model from local path: {self.modelPath_base}")
			base_model = AutoModel.from_pretrained(modelFile_base)
		else:
			self.logger.info(f"Base model not found locally. Loading from Hugging Face Hub: {self.modelId}")
			base_model = AutoModel.from_pretrained(
				pretrained_model_name_or_path=self.modelId, 
			)

		# Freeze parameters of the base model
		for param in base_model.parameters():
			param.requires_grad = False
		self.logger.info(f"Base model parameters frozen. Only the classification head will be trainable.")

		# Determine available device and move model if needed
		base_model.to(self.device)
		self.logger.info(f"Base model loaded and moved to device: {self.device}")

		# Return
		return base_model

	def saveBaseModel(self, model: Any, name: str = "pytorch_model.bin"):
		"""
		Save the (frozen) base model to file.
		"""

		# Make sure model is given
		if model is None:
			self.logger.error(f"No model provided to saveBaseModel()!")
			return

		# Make full file name for file
		modelFile_base: str = os.path.join(self.modelPath_base, name)

		# Make sure directory exists for model components
		os.makedirs(self.modelPath_base, exist_ok=True)

		# Save the base model using transformers' save_pretrained when available
		try:
			try:
				model.save_pretrained(self.modelPath_base)
			except Exception as e:
				torch.save(model.state_dict(), modelFile_base)
			self.logger.info(f"Base model saved to {modelFile_base}")
		except Exception as e:
			self.logger.error(f"Failed to save base model to {modelFile_base}: {e}")

	def loadTokenizer(self, name: str = "tokenizer.pt") -> Any:
		"""
		Load the tokenizer from file or HF if needed.
		"""

		# Make full file name for file
		modelFile_tokenizer: str = os.path.join(self.modelPath_tokenizer, name)
		
		# Make sure directory exists for model components
		os.makedirs(self.modelPath_tokenizer, exist_ok=True)

		# Check if the tokenizer has been saved locally
		if os.path.exists(modelFile_tokenizer):
			self.logger.info(f"Loading tokenizer from local path: {modelFile_tokenizer}")
			tokenizer = AutoTokenizer.from_pretrained(modelFile_tokenizer)
		else:
			self.logger.info(f"Tokenizer not found locally. Loading from Hugging Face Hub: {self.modelId}")
			tokenizer = AutoTokenizer.from_pretrained(self.modelId)

		# Return
		return tokenizer

	def saveTokenizer(self, tokenizer: Any, name: str = "tokenizer.pt"):
		"""
		Save the tokenizer to file.
		"""

		# Make sure tokenizer is given
		if tokenizer is None:
			self.logger.error(f"No tokenizer provided to saveTokenizer()!")
			return

		# Make full file name for file
		modelFile_tokenizer: str = os.path.join(self.modelPath_tokenizer, name)

		# Make sure directory exists for model components
		os.makedirs(self.modelPath_tokenizer, exist_ok=True)

		# Save the tokenizer using transformers' save_pretrained when available
		try:
			if hasattr(tokenizer, "save_pretrained"):
				tokenizer.save_pretrained(self.modelPath_tokenizer)
			else:
				torch.save(tokenizer, modelFile_tokenizer)
			self.logger.info(f"Tokenizer saved to {modelFile_tokenizer}")
		except Exception as e:
			self.logger.error(f"Failed to save tokenizer to {modelFile_tokenizer}: {e}")

	def loadClassificationHead(self, name: str = "classification_head.pt") -> nn.Module:
		"""
		Load the (trainable) classification head from file or create a new one if needed.
		"""

		# Make full file name for file
		modelFile_head: str = os.path.join(self.modelPath_head, name)
		
		# Create classification head
		head = nn.Sequential(
			nn.LayerNorm(self.component_base.config.hidden_size),
			nn.Dropout(0.2),
			nn.Linear(self.component_base.config.hidden_size, self.component_base.config.hidden_size // 2),
			nn.GELU(),
			nn.Dropout(0.2),
			nn.Linear(self.component_base.config.hidden_size // 2, self.modelNClasses)
		)

		# Check if saved locally, and if so, load state dict
		if os.path.exists(modelFile_head):
			self.logger.info(f"Loading classification head from local path: {modelFile_head}")
			try:
				head.load_state_dict(torch.load(modelFile_head, map_location=self.device))
				self.logger.info(f"Classification head loaded successfully from {modelFile_head}")
			except Exception as e:
				self.logger.error(f"Failed to load classification head from {modelFile_head}: {e}")
				self.logger.info(f"Proceeding with newly initialized classification head.")
		else:
			self.logger.info(f"Classification head not found locally, using new head!")

		# Determine data type of base model and convert to same dtype if needed
		try:
			base_dtype = next(self.component_base.parameters()).dtype
		except StopIteration:
			base_dtype = torch.float16

		# Send to device
		head.to(self.device, dtype=base_dtype)

		# Make sure head is trainable
		for param in head.parameters():
			param.requires_grad = True
		return head

	def saveClassificationHead(self, head: nn.Module, name: str = "classification_head.pt"):
		"""
		Save the (trainable) classification head to file.
		"""

		# Make sure head is given
		if head is None:
			self.logger.error(f"No classification head provided to saveClassificationHead()!")
			return

		# Make full file name for file
		modelFile_head: str = os.path.join(self.modelPath_head, name)

		# Make sure directory exists for model components
		os.makedirs(self.modelPath_head, exist_ok=True)

		# Save the classification head
		try:
			torch.save(head.state_dict(), modelFile_head)
			self.logger.info(f"Classification head saved to {modelFile_head}")
		except Exception as e:
			self.logger.error(f"Failed to save classification head to {modelFile_head}: {e}")

	def getStateDict(self) -> dict:
		"""
		Get the state dict of the model (only the classification head, since the base is frozen).
		"""
		return self.component_head.state_dict()

	def getTrainableParameters(self):
		"""
		Get the trainable parameters of the model (i.e. the classification head).
		"""
		return self.component_head.parameters()
	
	def buildOptimizer(self, learningRate: float = 1e-4, weightDecay: float = 0.01):
		"""
		Build an optimizer for the trainable parameters of the model.
		"""
		return torch.optim.AdamW(self.getTrainableParameters(), lr=learningRate, weight_decay=weightDecay, eps=1e-5)
	
	def buildScheduler(self, optimizer, numWarmupSteps: int = 0, numTrainingSteps: int = 1000):
		"""
		Build a learning rate scheduler for the optimizer.
		"""
		return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=numWarmupSteps, num_training_steps=numTrainingSteps)
	
	def _forwardPass(self, batch: dict) -> Tuple:
		"""
		Perform a forward pass through the model and return the logits and labels.
		"""

		# Get encodings for input
		encodings: dict = {k: v.to(self.device) for k, v in batch.items() if (k != "labels" and k != "text")}
		
		# Keep labels separate and move to device
		labels = batch.get("labels", None)
		if labels is not None:
			labels = labels.to(self.device)
		
		# Forward pass through model
		outputs = self.model(
			**encodings, 
			output_hidden_states=True
		)

		# Separate logits and embeddings
		if hasattr(outputs, "logits") and hasattr(outputs, "hidden_states"):
			logits = outputs.logits
			embeddings = outputs.hidden_states[-1][:, 0, :]
			
		elif isinstance(outputs, tuple):
			logits = outputs[0]
			embeddings = outputs[1][-1][:, 0, :] 
		
		else:
			self.logger.error(f"Unexpected model output format `type: {type(outputs)}` during predict()!")
			self.logger.debug(f"---> Model output: {outputs}")
			raise ValueError("Unexpected model output format.")

		# Return logits and labels
		return logits, labels, embeddings

	def train(self, dataLoader: DataLoader, globalStateDict: dict, epochs: int = 1, learningRate: float = 1e-4, weightDecay: float = 0.00, maxGradientNorm: float = 1.0, schedulerWarmupSteps: int = 0):
		"""
		Train the model on the given data loader.

		Parameters:
		- dataLoader (DataLoader): The DataLoader containing the training data.
		- globalStateDict (dict): The global state dict to load into the model before training.
		- epochs (int): The number of epochs to train for.
		- learningRate (float): The learning rate for the optimizer.
		- weightDecay (float): The weight decay for the optimizer.
		- maxGradientNorm (float): The maximum gradient norm for gradient clipping.
		- schedulerWarmupSteps (int): The number of warmup steps for the learning rate scheduler.
		"""
		self.logger.info(f"Starting training for {epochs} epochs with learning rate {learningRate} and weight decay {weightDecay}!")

		# Build optimizer and scheduler
		optimizer = self.buildOptimizer(learningRate=learningRate, weightDecay=weightDecay)
		scheduler = self.buildScheduler(optimizer, numWarmupSteps=schedulerWarmupSteps, numTrainingSteps=epochs * len(dataLoader))

		# Training loop
		epochHistory = {}
		for epoch in range(1, epochs + 1):

			# Set model into training mode and prepare trackers
			self.model.train()
			epochStats_loss = 0.0
			epochStats_lossAverage = 0.0
			epochStats_accuracy = 0.0
			epochStats_correct = 0
			epochStats_total = 0

			# Iterate over training data
			lossFunction = torch.nn.BCEWithLogitsLoss()
			for i, batch in enumerate(dataLoader, 1):

				# Forward pass
				logits, labels, embeddings = self._forwardPass(batch)

				# Ensure labels have same dtype and device as logits for loss calculation
				if isinstance(labels, torch.Tensor):
					labels = labels.to(dtype=logits.dtype, device=logits.device)
				else:
					labels = torch.tensor(labels, dtype=logits.dtype, device=logits.device)

				# Validate for any potential errors
				if torch.isnan(labels).any():
					self.logger.error(f"DATA POISONING: batch['labels'] contains NaN values at batch {i}!")
				
				if (labels < 0).any() or (labels > 1).any():
					self.logger.error(f"DATA OUT OF BOUNDS: labels must be strictly 0 or 1. Found invalid values at batch {i}!")

				if torch.isnan(logits).any():
					self.logger.error(f"WEIGHT CORRUPTION: Logits are NaN at batch {i}. The weights exploded during the previous backward pass!")

				# Calculate loss
				loss = lossFunction(logits, labels)

				# Backward pass
				optimizer.zero_grad()
				loss.backward()

				# Gradient clipping
				torch.nn.utils.clip_grad_norm_(self.getTrainableParameters(), max_norm=maxGradientNorm)

				# Step optimizer and scheduler
				optimizer.step()
				scheduler.step()

				# Calculate probabilities and predictions
				probabilities = torch.sigmoid(logits)
				predictions = (probabilities > self.confidenceThreshold)

				# Track batch stats
				batchStats_total = labels.numel()
				batchStats_loss = loss.item() * batchStats_total
				batchStats_correct = (predictions == labels).sum().item()
				batchStats_lossAverage = batchStats_loss / batchStats_total
				batchStats_accuracy = batchStats_correct / batchStats_total

				# Track epoch stats
				epochStats_loss += batchStats_loss
				epochStats_correct += batchStats_correct
				epochStats_total += batchStats_total
				epochStats_lossAverage = epochStats_loss / epochStats_total
				epochStats_accuracy = epochStats_correct / epochStats_total

				# Log
				self.logger.debug(f"Processed batch {i} / {len(dataLoader)} for epoch {epoch}/{epochs}! Batch loss: {batchStats_lossAverage:.4f}, Batch accuracy: {batchStats_accuracy:.4f}, Epoch loss: {epochStats_lossAverage:.4f}, Epoch accuracy: {epochStats_accuracy:.4f}")

			# Save epoch stats to history
			epochHistory[epoch] = {"loss": epochStats_lossAverage, "accuracy": epochStats_accuracy}
			self.logger.debug(f"Epoch {epoch}/{epochs} completed! Train Loss: {epochStats_lossAverage:.4f}, Train Accuracy: {epochStats_accuracy:.4f}")
		
		self.logger.info(f"Training completed for {epochs} epochs!")
		return epochHistory

	def evaluate(self, dataLoader: DataLoader) -> dict[str, float]:
		"""
		Evaluate model on given data loader.

		Parameters:
		- dataLoader (DataLoader): The DataLoader containing the evaluation data.

		Returns:
		- dict: A dictionary containing metrics:
			{
				"loss": float,  		# Average loss over the evaluation dataset
				"accuracy": float,  	# Accuracy over the evaluation dataset
				"hamming_loss": float,  # Hamming loss over the evaluation dataset
				"precision": float,  	# Precision over the evaluation dataset
				"recall": float,  		# Recall over the evaluation dataset
				"f1_score": float  		# F1 score over the evaluation dataset
			}
		"""
		self.logger.debug(f"Starting model evaluation!")

		# Set model into training mode and prepare trackers
		self.model.eval()
		valStats_loss = 0.0
		valStats_total = 0
		allLabels = []
		allPredictions = []

		# Iterate over validation data
		lossFunction = torch.nn.BCEWithLogitsLoss()
		with torch.no_grad():
			for batch in dataLoader:

				# Forward pass
				logits, labels, embeddings = self._forwardPass(batch)
				loss = lossFunction(logits, labels.to(logits.dtype))

				# Calculate probabilities and predictions
				probabilities = torch.sigmoid(logits)
				predictions = (probabilities > self.confidenceThreshold)

				# Track validation stats
				total = labels.numel()
				loss = loss.item() * total
				valStats_total += total
				valStats_loss += loss

				# Append to list
				allPredictions.extend(predictions.cpu().view(-1).tolist())
				allLabels.extend(labels.cpu().view(-1).tolist())

		# Calculate metrics with sklearn
		valStats_lossAverage = valStats_loss / max(1, valStats_total)
		valStats_accuracy = accuracy_score(allLabels, allPredictions)
		valStats_hammingLoss = hamming_loss(allLabels, allPredictions)
		valStats_precision, valStats_recall, valStats_f1, _ = precision_recall_fscore_support(allLabels, allPredictions, average='macro', zero_division=0)

		# Log and return
		self.logger.info(f"Model evaluation completed! Validation Loss: {valStats_lossAverage:.4f}, Validation Accuracy: {valStats_accuracy:.4f}")
		return {
			"loss": valStats_lossAverage,
			"accuracy": valStats_accuracy,
			"hamming_loss": valStats_hammingLoss,
			"precision": valStats_precision,
			"recall": valStats_recall,
			"f1_score": valStats_f1
		}

	def predict(self, dataLoader: DataLoader) -> Tuple[Dict[HSSFLDON_PredictionOutputType, torch.Tensor], torch.Tensor]:
		
		# Holder for all logits
		logitsList = []
		labelsList = []
		embeddingsList = []

		# Iterate over validation data
		lossFunction = torch.nn.BCEWithLogitsLoss()
		with torch.no_grad():
			for i, batch in enumerate(dataLoader):
				self.logger.debug(f"Making predictions on batch {i} of {len(dataLoader)}!")

				# Forward pass
				logits, labels, embeddings = self._forwardPass(batch)
				logitsList.append(logits.cpu())
				if labels is not None:
					labelsList.append(labels.cpu())
				embeddingsList.append(embeddings.cpu())

		# Concatenate all logits
		logits = torch.cat(logitsList, dim=0) if len(logitsList) > 0 else torch.empty((0, self.modelNClasses))
		labels = torch.cat(labelsList, dim=0) if len(labelsList) > 0 else torch.empty((0, self.modelNClasses))
		embeddings = torch.cat(embeddingsList, dim=0) if len(embeddingsList) > 0 else torch.empty((0, self.component_base.config.hidden_size))

		# Process output type
		result = {}
		result[HSSFLDON_PredictionOutputType.LOGIT_PREDICTION] = logits.cpu()
		result[HSSFLDON_PredictionOutputType.EMBEDDING_PREDICTION] = embeddings.cpu()
		result[HSSFLDON_PredictionOutputType.PROBABILITY_PREDICTION] = torch.sigmoid(logits).cpu()
		result[HSSFLDON_PredictionOutputType.BINARY_PREDICTION] = (torch.sigmoid(logits) > self.confidenceThreshold).long().cpu()
		return result, labels
	
	def tokenize_and_create_dataloader(self, texts, labels, batch_size: int = 128, max_length: int = 256, shuffle: bool = True):
		class _SimpleDS(Dataset):

			def __init__(self, tokenizer, texts, labels, max_length):
				self.tokenizer = tokenizer
				self.texts = texts
				self.labels = labels
				self.max_length = max_length
				self.others = {}

			def __len__(self): 
				return len(self.texts)

			def __getitem__(self, idx):
				enc = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
				item = {k: v.squeeze(0) for k, v in enc.items()}
				item["text"] = self.texts[idx]
				if self.labels is not None:
					item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
				for col_name, col_values in self.others.items():
					val = col_values[idx]
					if not isinstance(val, torch.Tensor):
						val = torch.tensor(val)
					item[col_name] = val
				return item
			
			def add_column(self, name: str, values: Any):
				"""
				Adds a new column of data to the dataset.

				Args:
					name (str): The name of the new column.
					values (list): A list of values for the new column, which must have the same length as the dataset.
				"""
				if len(values) != len(self.texts):
					raise ValueError(f"Length mismatch: '{name}' has {len(values)} items, but dataset has {len(self.texts)} items.")
				self.others[name] = values
				return self
			
			def remove_column(self, name: str):
				"""
				Removes a column of data from the dataset.

				Args:
					name (str): The name of the column to remove.
				"""
				if name in self.others:
					self.others.pop(name)
				else:
					raise ValueError(f"Column '{name}' not found in dataset.")
				return self

		ds = _SimpleDS(self.tokenizer, texts, labels, max_length)
		return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=None)