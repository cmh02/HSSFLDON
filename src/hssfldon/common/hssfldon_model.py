'''
	# HSSFLDON - Model

	This common module will provide the model for use by the server and clients in HSSFLDON.
'''

### Library Imports
import os
import copy
import torch
import torch.nn as nn
from typing import Tuple, Any
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

# Project Imports
from hssfldon.common.hssfldon_logger import HSSFLDON_Logger

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
		return self.head(pooled)

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
		self.logger: HSSFLDON_Logger = HSSFLDON_Logger(name=f"ModelManager")

		# Determine device
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# Get needed values from env
		self.modelDirectory: str = os.getenv("HSSFLDON_MODEL_DIRECTORY", "models")
		self.modelId: str = os.getenv("HSSFLDON_MODEL_ID", "microsoft/deberta-v3-small")
		self.modelNClasses: int = int(os.getenv("HSSFLDON_MODEL_N_CLASSES", 9))
		self.huggingFaceAccessToken: str | None = os.getenv("HSSFLDON_HF_ACCESS_TOKEN", None)

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
		encodings: dict = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
		
		# Keep labels separate and move to device
		labels = batch.get("labels", None)
		if labels is not None:
			labels = labels.to(self.device)
			self.logger.debug(f"Labels found in batch; moved to device: {self.device}")
		
		# Forward pass through model
		logits = self.model(**encodings)
		if isinstance(logits, tuple):
			logits = logits[0]
			self.logger.warning(f"Model output is a tuple; using the first element as logits!")
		
		# Return logits and labels
		return logits, labels

	def train(self, trainingDataLoader: DataLoader, validationDataLoader: Any = None, epochs: int = 1, learningRate: float = 1e-4, weightDecay: float = 0.00, maxGradientNorm: float = 1.0, schedulerWarmupSteps: int = 0):
		"""
		Train the model on the given data loader.
		"""
		self.logger.info(f"Starting training for {epochs} epochs with learning rate {learningRate} and weight decay {weightDecay}!")

		# Build optimizer and scheduler
		optimizer = self.buildOptimizer(learningRate=learningRate, weightDecay=weightDecay)
		scheduler = self.buildScheduler(optimizer, numWarmupSteps=schedulerWarmupSteps, numTrainingSteps=epochs * len(trainingDataLoader))

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
			for i, batch in enumerate(trainingDataLoader, 1):

				# Forward pass
				logits, labels = self._forwardPass(batch)

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
				predictions = (probabilities > 0.5)

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
				self.logger.debug(f"Processed batch {i} / {len(trainingDataLoader)} for epoch {epoch}/{epochs}! Batch loss: {batchStats_lossAverage:.4f}, Batch accuracy: {batchStats_accuracy:.4f}, Epoch loss: {epochStats_lossAverage:.4f}, Epoch accuracy: {epochStats_accuracy:.4f}")

			# Save epoch stats to history
			epochHistory[epoch] = {"train": {}, "validation": {}}
			epochHistory[epoch]["train"]["loss"] = epochStats_lossAverage
			epochHistory[epoch]["train"]["accuracy"] = epochStats_accuracy

			# Perform validation if given after each epoch
			if validationDataLoader is None:
				self.logger.debug(f"Epoch {epoch}/{epochs} completed! Train Loss: {epochStats_lossAverage:.4f}, Train Accuracy: {epochStats_accuracy:.4f}")
			else:
				validationLoss, validationAccuracy = self.evaluate(validationDataLoader)
				epochHistory[epoch]["validation"]["loss"] = validationLoss
				epochHistory[epoch]["validation"]["accuracy"] = validationAccuracy
				self.logger.debug(f"Epoch {epoch}/{epochs} completed! Train Loss: {epochStats_lossAverage:.4f}, Train Accuracy: {epochStats_accuracy:.4f}, Validation Loss: {validationLoss:.4f}, Validation Accuracy: {validationAccuracy:.4f}")

		self.logger.info(f"Training completed for {epochs} epochs!")
		return epochHistory

	def evaluate(self, validationDataLoader: DataLoader) -> tuple[float, float]:
		"""
		Evaluate model on given data loader.
		"""
		self.logger.debug(f"Starting model evaluation!")

		# Set model into training mode and prepare trackers
		self.model.eval()
		valStats_loss = 0.0
		valStats_lossAverage = 0.0
		valStats_accuracy = 0.0
		valStats_correct = 0
		valStats_total = 0

		# Iterate over validation data
		lossFunction = torch.nn.BCEWithLogitsLoss()
		with torch.no_grad():
			for batch in validationDataLoader:

				# Forward pass
				logits, labels = self._forwardPass(batch)
				loss = lossFunction(logits, labels)

				# Calculate probabilities and predictions
				probabilities = torch.sigmoid(logits)
				predictions = (probabilities > 0.5)

				# Track validation stats
				total = labels.numel()
				loss = loss.item() * total
				correct = (predictions == labels).sum().item()
				valStats_total += total
				valStats_loss += loss
				valStats_correct += correct

		# Calculate average loss and accuracy
		valStats_lossAverage = valStats_loss / max(1, valStats_total)
		valStats_accuracy = valStats_correct / max(1, valStats_total)
		self.logger.info(f"Model evaluation completed! Validation Loss: {valStats_lossAverage:.4f}, Validation Accuracy: {valStats_accuracy:.4f}")
		return valStats_lossAverage, valStats_accuracy
	
	def predict(self, texts: list[str], batchSize: int = 16, maxLength: int = 256):

		# Create holder for output
		outputs = []

		# Pocess in batches
		with torch.no_grad():
			for i in range(0, len(texts), batchSize):
				batchTexts = texts[i:i+batchSize]
				encodings = self.tokenizer(batchTexts, truncation=True, padding=True, max_length=maxLength, return_tensors="pt")
				logits, _ = self._forwardPass(encodings)
				probabilities = torch.sigmoid(logits)
				predictions = (probabilities > 0.5)
				outputs.extend(predictions.cpu().tolist())
		return outputs
	
	def tokenize_and_create_dataloader(self, texts, labels, batch_size: int = 16, max_length: int = 256, shuffle: bool = True):
		from torch.utils.data import Dataset, DataLoader
		class _SimpleDS(Dataset):
			def __init__(self, tokenizer, texts, labels, max_length):
				self.tokenizer = tokenizer
				self.texts = texts
				self.labels = labels
				self.max_length = max_length
			def __len__(self): return len(self.texts)
			def __getitem__(self, idx):
				enc = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
				item = {k: v.squeeze(0) for k, v in enc.items()}
				item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
				return item
		ds = _SimpleDS(self.tokenizer, texts, labels, max_length)
		return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=None)