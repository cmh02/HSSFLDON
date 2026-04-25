'''
	# HSSFLDON - Data Loader

	This common module will provide common utilities for loading data at both server and clients.
'''

# Library Imports
from dotenv import load_dotenv
from datasets import load_dataset, Dataset #type: ignore 

# Project Imports
from hssfldon.common.hssfldon_logger import HSSFLDON_Logger

class HSSFLDON_DataLoader:
	"""
	The main data loader class for HSSFLDON.
	"""
	def __init__(self):

		# Load env file
		envStatus: bool = load_dotenv()
		if envStatus is False:
			print(f"Warning: .env file not found or failed to load. Make sure to create a .env file with the necessary configuration variables!")
 
		# Get logger
		self.logger: HSSFLDON_Logger = HSSFLDON_Logger(name=f"DataLoader")

	def loadDataset(self, path: str, split: str = "train") -> Dataset | None:
		"""
		Load a dataset from a given path.

		Args:
			path (str): The path to the dataset.
			split (str): The split of the dataset to load.

		Returns:
			Dataset: The loaded dataset.
		"""

		# Load dataset from parquet file
		try:
			dataset: Dataset = load_dataset("parquet", data_files=path, split=split)
		except Exception as e:
			self.logger.error(f"Error loading dataset from path `{path}`: {e}")
			return None
		
		# Validate
		if len(dataset) == 0:
			self.logger.warning(f"Loaded dataset from path `{path}` is empty!")
			return None
		
		# Relabel 'classifications' to 'labels' for consistency
		if "classifications" in dataset.column_names:
			dataset = dataset.rename_column("classifications", "labels")

		# Only keep 'text' and 'labels' columns for training
		columns_to_keep = ["text", "labels"]
		columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
		if columns_to_remove:
			dataset = dataset.remove_columns(columns_to_remove)
			self.logger.info(f"Removed unnecessary columns from dataset: {columns_to_remove}")
		
		# Log and return
		self.logger.info(f"Loaded dataset from path `{path}` with {len(dataset)} samples!")
		return dataset