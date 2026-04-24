'''
	# HSSFLDON - Model

	This common module will provide the model for use by the server and clients in HSSFLDON.
'''

### Library Imports
import os
import copy
import torch
from dotenv import load_dotenv
from huggingface_hub import login

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
		self.huggingFaceAccessToken: str | None = os.getenv("HSSFLDON_HF_ACCESS_TOKEN", None)

		# Login to HF if needed
		if self.huggingFaceAccessToken:
			login(token=self.huggingFaceAccessToken)
			self.logger.info(f"Logged in to Hugging Face Hub successfully!")
		else:
			self.logger.warning(f"No Hugging Face access token provided. If the model `{self.modelId}` is private, model loading will fail!")

	def loadBaseModel(self):
		"""
		Load the (frozen) base model from file or HF if needed.
		"""
		pass

	def saveModel(self):
		"""
		Save the model to file.
		"""
		pass

	def loadClassificationHead(self):
		"""
		Load the (trainable) classification head from file or create a new one if needed.
		"""
		pass

	def saveClassificationHead(self):
		"""
		Save the classification head to file.
		"""
		pass