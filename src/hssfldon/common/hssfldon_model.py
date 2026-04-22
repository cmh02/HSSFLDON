'''
	# HSSFLDON - Model

	This common module will provide the model for use by the server and clients in HSSFLDON.
'''

### Library Imports
import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

# Project Imports
from hssfldon.common.hssfldon_logger import HSSFLDON_Logger

class HSSFLDON_ModelManager:
	"""
	The main model class for HSSFLDON.
	"""
	def __init__(self, modelId: str):

		# Parse dotenv for env variables
		envStatus: bool = load_dotenv()
		if envStatus is False:
			print(f"Warning: .env file not found or failed to load. Make sure to create a .env file with the necessary configuration variables!")
			
		# Get logger
		self.logger: HSSFLDON_Logger = HSSFLDON_Logger(name=f"ModelManager")

		# Copy in params
		self.modelId: str = modelId

		# Get needed values from env
		self.loraRank: int = int(os.getenv("HSSFLDON_LORA_RANK", 8))
		self.loraAlpha: int = int(os.getenv("HSSFLDON_LORA_ALPHA", 32))
		self.huggingFaceAccessToken: str = os.getenv("HSSFLDON_HF_ACCESS_TOKEN", None)

		# Login to HF if needed
		if self.huggingFaceAccessToken:
			login(token=self.huggingFaceAccessToken)
			self.logger.info(f"Logged in to Hugging Face Hub successfully!")
		else:
			self.logger.warning(f"No Hugging Face access token provided. If the model `{self.modelId}` is private, model loading will fail!")

		# Setup base model
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.base_model = AutoModelForCausalLM.from_pretrained(
			self.modelId, 
			torch_dtype=torch.float16,
			device_map=self.device
		)
		self.tokenizer = AutoTokenizer.from_pretrained(self.modelId)
		self.lora_config = LoraConfig(r=self.loraRank, lora_alpha=self.loraAlpha, target_modules=["q_proj", "v_proj"])
		self.logger.info(f"Initialized base model `{self.modelId}` on device `{self.device}` with LoRA config: `rank={self.loraRank}`, `alpha={self.loraAlpha}`!")

	def getFreshModel(self) -> PeftModel:
		"""
		Get a fresh instance of the model.

		Returns:
			An instance of the model.
		"""
		return get_peft_model(self.base_model, self.lora_config)

	def loadAdapterFromFile(self, filePath: str) -> PeftModel:
		"""
		Load a model adapter from a file.

		Args:
			filePath: The path to the adapter file.
		
		Returns:
			An instance of the model with the adapter loaded.
		"""
		if not os.path.exists(filePath):
			raise FileNotFoundError(f"Adapter file not found at path: {filePath}")
		
		return PeftModel.from_pretrained(model=self.base_model, model_id=self.modelId, adapter_name=filePath, is_trainable=True)

	def saveAdapterToFile(self, model: PeftModel, filePath: str):
		"""
		Save a model adapter to a file.

		Args:
			model: The model instance with the adapter to save.
			filePath: The path to save the adapter file to.
		"""
		model.save_pretrained(save_directory=filePath)

def aggregateAdapters(self, peftModel: PeftModel, clientPaths: list[str], savePath: str):
		"""
		Uses FedAvg to aggregate client adapters into new global adapter.
		"""
		
		# Load all client adapters into model memory
		adapterNames: list[str] = []
		for i, path in enumerate(clientPaths):
			name: str = f"client_update_{i}"
			adapterNames.append(name)
			peftModel.load_adapter(path, adapter_name=name)
			
		# Calculate weights for averaging based on number of clients
		weightValue: float = 1.0 / len(clientPaths)
		weights: list[float] = [weightValue] * len(clientPaths)
		
		# Average weights with FedAvg into new global adapter
		peftModel.add_weighted_adapter(
			adapters=adapterNames,
			weights=weights,
			adapter_name="new_global_adapter",
			combination_type="linear"
		)
		
		# Update global adapter and save
		peftModel.set_adapter("new_global_adapter")
		self.saveAdapterToFile(peftModel, savePath)
		
		# Cleanup VRAM and model
		for name in adapterNames:
			peftModel.delete_adapter(name)
		peftModel.delete_adapter("new_global_adapter")

		# Log
		self.logger.info(f"Aggregated {len(clientPaths)} client adapters into new global adapter and saved to `{savePath}` successfully!")