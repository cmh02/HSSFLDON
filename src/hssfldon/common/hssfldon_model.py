'''
	# HSSFLDON - Model

	This common module will provide the model for use by the server and clients in HSSFLDON.
'''

### Library Imports
import os
import copy
import torch
from typing import Tuple
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

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
	def __init__(self):

		# Parse dotenv for env variables
		envStatus: bool = load_dotenv()
		if envStatus is False:
			print(f"Warning: .env file not found or failed to load. Make sure to create a .env file with the necessary configuration variables!")
			
		# Get logger
		self.logger: HSSFLDON_Logger = HSSFLDON_Logger(name=f"ModelManager")

		# Get needed values from env
		self.modelDirectory: str = os.getenv("HSSFLDON_MODEL_DIRECTORY", "models")
		self.modelId: str = os.getenv("HSSFLDON_MODEL_ID", "microsoft/deberta-v3-small")
		self.modelNClasses: int = int(os.getenv("HSSFLDON_MODEL_N_CLASSES", 9))
		self.huggingFaceAccessToken: str | None = os.getenv("HSSFLDON_HF_ACCESS_TOKEN", None)

		# Create static paths for model components
		self.modelPath_base: str = os.path.join(self.modelDirectory, f"model_base")
		self.modelFile_base: str = os.path.join(self.modelPath_base, f"pytorch_model.bin")
		self.modelPath_head: str = os.path.join(self.modelDirectory, f"model_head")
		self.modelFile_head: str = os.path.join(self.modelPath_head, f"classification_head.pt")
		self.modelPath_tokenizer: str = os.path.join(self.modelDirectory, f"model_tokenizer")
		self.modelFile_tokenizer: str = os.path.join(self.modelPath_tokenizer, f"tokenizer.pt")

		# Login to HF if needed
		if self.huggingFaceAccessToken:
			login(token=self.huggingFaceAccessToken)
			self.logger.info(f"Logged in to Hugging Face Hub successfully!")
		else:
			self.logger.warning(f"No Hugging Face access token provided. If the model `{self.modelId}` is private, model loading will fail!")

		# Initialize model, tokenizer, and classification head
		self.component_base = self.loadBaseModel()
		self.component_head = self.loadClassificationHead()
		self.model = HSSFLDON_ClassifierHead(base=self.component_base, head=self.component_head)
		self.tokenizer = self.loadTokenizer()

	def loadBaseModel(self):
		"""
		Load the (frozen) base model from file or HF if needed.
		"""
		
		# Check if the model has been saved locally
		if os.path.exists(self.modelPath_base):
			self.logger.info(f"Loading base model from local path: {self.modelPath_base}")
			base_model = AutoModel.from_pretrained(self.modelPath_base)
		else:
			self.logger.info(f"Base model not found locally. Loading from Hugging Face Hub: {self.modelId}")
			base_model = AutoModel.from_pretrained(self.modelId)

		# Determine available device and move model if needed
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		base_model.to(self.device)
		self.logger.info(f"Base model loaded and moved to device: {self.device}")

		# Freeze parameters of the base model
		for param in base_model.parameters():
			param.requires_grad = False
		self.logger.info(f"Base model parameters frozen. Only the classification head will be trainable.")

		# Return
		return base_model

	def saveBaseModel(self):
		"""
		Save the (frozen) base model to file.
		"""

		# Make sure directory exists for model components
		os.makedirs(self.modelPath_base, exist_ok=True)

		# If base not loaded, try to load it first (this will also set tokenizer)
		if not hasattr(self, "base_model"):
			self.logger.warning("Base model not loaded; calling loadBaseModel() before saving.")
			self.base_model = self.loadBaseModel()

		# Save the base model using transformers' save_pretrained when available
		try:
			if hasattr(self.base_model, "save_pretrained"):
				self.base_model.save_pretrained(self.modelPath_base)
			else:
				torch.save(self.base_model.state_dict(), self.modelFile_base)
			self.logger.info(f"Base model saved to {self.modelFile_base}")
		except Exception as e:
			self.logger.error(f"Failed to save base model to {self.modelFile_base}: {e}")

	def loadTokenizer(self):
		"""
		Load the tokenizer from file or HF if needed.
		"""
		
		# Make sure directory exists for model components
		os.makedirs(self.modelPath_tokenizer, exist_ok=True)

		# Check if the tokenizer has been saved locally
		if os.path.exists(self.modelPath_tokenizer):
			self.logger.info(f"Loading tokenizer from local path: {self.modelPath_tokenizer}")
			tokenizer = AutoTokenizer.from_pretrained(self.modelPath_tokenizer)
		else:
			self.logger.info(f"Tokenizer not found locally. Loading from Hugging Face Hub: {self.modelId}")
			tokenizer = AutoTokenizer.from_pretrained(self.modelId)

		# Return
		return tokenizer

	def saveTokenizer(self):
		"""
		Save the tokenizer to file.
		"""
		
		# Make sure directory exists for model components
		os.makedirs(self.modelPath_tokenizer, exist_ok=True)

		# If tokenizer not loaded, try to load it first (this will also set base_model)
		if not hasattr(self, "tokenizer"):
			self.logger.warning("Tokenizer not loaded; calling loadTokenizer() before saving.")
			self.tokenizer = self.loadTokenizer()

		# Save the tokenizer using transformers' save_pretrained when available
		try:
			if hasattr(self.tokenizer, "save_pretrained"):
				self.tokenizer.save_pretrained(self.modelPath_tokenizer)
			else:
				torch.save(self.tokenizer, self.modelFile_tokenizer)
			self.logger.info(f"Tokenizer saved to {self.modelFile_tokenizer}")
		except Exception as e:
			self.logger.error(f"Failed to save tokenizer to {self.modelFile_tokenizer}: {e}")

	def loadClassificationHead(self):
		"""
		Load the (trainable) classification head from file or create a new one if needed.
		"""
		
		# Create classification head
		head = torch.nn.Linear(self.component_base.config.hidden_size, self.modelNClasses)

		# Check if saved locally, and if so, load state dict
		if os.path.exists(self.modelFile_head):
			self.logger.info(f"Loading classification head from local path: {self.modelFile_head}")
			try:
				head.load_state_dict(torch.load(self.modelFile_head, map_location=self.device))
				self.logger.info(f"Classification head loaded successfully from {self.modelFile_head}")
			except Exception as e:
				self.logger.error(f"Failed to load classification head from {self.modelFile_head}: {e}")
				self.logger.info(f"Proceeding with newly initialized classification head.")
		else:
			self.logger.info(f"Classification head not found locally, using new head!")

		# Send to device
		head.to(self.device)

		# Make sure head is trainable
		for param in head.parameters():
			param.requires_grad = True

		return head

	def saveClassificationHead(self):
		"""
		Save the (trainable) classification head to file.
		"""
		# Make sure directory exists for model components
		os.makedirs(self.modelPath_head, exist_ok=True)

		# If classification head not loaded, try to load it first
		if not hasattr(self, "classification_head"):
			self.logger.warning("Classification head not loaded; calling loadClassificationHead() before saving.")
			self.classification_head = self.loadClassificationHead()

		# Save the classification head
		try:
			torch.save(self.classification_head.state_dict(), self.modelFile_head)
			self.logger.info(f"Classification head saved to {self.modelFile_head}")
		except Exception as e:
			self.logger.error(f"Failed to save classification head to {self.modelFile_head}: {e}")

	def getTrainableParameters(self):
		"""
		Get the trainable parameters of the model (i.e. the classification head).
		"""
		return self.component_head.parameters()
	
	def buildOptimizer(self, learningRate: float = 1e-4, weightDecay: float = 0.01):
		"""
		Build an optimizer for the trainable parameters of the model.
		"""
		return torch.optim.AdamW(self.getTrainableParameters(), lr=learningRate, weight_decay=weightDecay)
	
	def buildScheduler(self, optimizer, numWarmupSteps: int = 0, numTrainingSteps: int = 1000):
		"""
		Build a learning rate scheduler for the optimizer.
		"""
		return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=numWarmupSteps, num_training_steps=numTrainingSteps)
	
	def _forwardPass(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Perform a forward pass through the model and return the logits and labels.
		"""
		encodings: dict = {k: v.to(self.device) for k, v in batch.items()}
		logits = self.model(**encodings)
		if isinstance(logits, tuple):
			logits = logits[0]
			self.logger.warning(f"Model output is a tuple; using the first element as logits!")
		return logits, batch["labels"]

	def train(self, trainingDataLoader, validationDataLoader = None, epochs: int = 1, learningRate: float = 1e-4, weightDecay: float = 0.00, maxGradientNorm: float = 1.0):
		"""
		Train the model on the given data loader.
		"""
		self.logger.info(f"Starting training for {epochs} epochs with learning rate {learningRate} and weight decay {weightDecay}!")

		# Build optimizer and scheduler
		optimizer = self.buildOptimizer(learningRate=learningRate, weightDecay=weightDecay)
		scheduler = self.buildScheduler(optimizer, numWarmupSteps=0, numTrainingSteps=epochs * len(trainingDataLoader))

		# Training loop
		epochHistory = {}
		for epoch in range(1, epochs + 1):
			self.logger.debug(f"Epoch {epoch}/{epochs}")

			# Set model into training mode and prepare trackers
			self.model.train()
			epochStats_loss = 0.0
			epochStats_lossAverage = 0.0
			epochStats_accuracy = 0.0
			epochStats_correct = 0
			epochStats_total = 0

			# Iterate over training data
			for batch in trainingDataLoader:

				# Each batch is a dict of tensors with keys like "input_ids", "attention_mask", and "labels"
				batch = {k: v.to(self.device) for k, v in batch.items()}

				# Forward pass
				logits, labels = self._forwardPass(batch)
				loss = torch.nn.functional.cross_entropy(logits, labels)

				# Backward pass
				optimizer.zero_grad()
				loss.backward()

				# Gradient clipping
				torch.nn.utils.clip_grad_norm_(self.getTrainableParameters(), max_norm=maxGradientNorm)

				# Step optimizer and scheduler
				optimizer.step()
				scheduler.step()

				# Track epoch stats
				epochStats_loss += loss.item() * batch["labels"].size(0)
				preds = torch.argmax(logits, dim=1)
				epochStats_correct += (preds == batch["labels"]).sum().item()
				epochStats_total += batch["labels"].size(0)

			# Calculate average loss and accuracy for the epoch
			epochStats_lossAverage = epochStats_loss / epochStats_total
			epochStats_accuracy = epochStats_correct / epochStats_total
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

	def evaluate(self, validationDataLoader) -> tuple[float, float]:
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
		for batch in validationDataLoader:

			# Each batch is a dict of tensors with keys like "input_ids", "attention_mask", and "labels"
			batch = {k: v.to(self.device) for k, v in batch.items()}

			# Forward pass
			logits, labels = self._forwardPass(batch)
			loss = torch.nn.functional.cross_entropy(logits, labels)

			# Track validation stats
			valStats_loss += loss.item() * batch["labels"].size(0)
			preds = torch.argmax(logits, dim=1)
			valStats_correct += (preds == batch["labels"]).sum().item()
			valStats_total += batch["labels"].size(0)

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
				preds = torch.argmax(logits, dim=1)
				outputs.extend(preds.cpu().tolist())
		return outputs