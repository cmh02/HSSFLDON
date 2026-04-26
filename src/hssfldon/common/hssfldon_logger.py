'''
	# HSSFLDON - Logger

	This common module will provide a logger for use anywhere in HSSFLDON.
'''

### Library Imports
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from transformers import TrainerCallback

class HSSFLDON_Logger:
	"""
	The main logger class for HSSFLDON.
	"""
	def __init__(self, name: str, level: int = logging.DEBUG):

		# Copy in params
		self.name: str = name
		self.level: int = level

		# Load env file
		load_dotenv(dotenv_path='../.env')

		# Setup logger and formatter
		self.logger: logging.Logger = logging.getLogger(name)
		self.formatter: logging.Formatter = logging.Formatter(
			fmt=f'[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
			datefmt='%m-%d %H:%M:%S'
		)

		# Setup console handler
		console_handler = self._setup_getConsoleHandler()
		self.logger.addHandler(console_handler)

		# Setup global file handler
		global_file_handler = self._setup_getGlobalFileHandler()
		self.logger.addHandler(global_file_handler)

		# Setup local file handler
		local_file_handler = self._setup_getLocalFileHandler()
		self.logger.addHandler(local_file_handler)

		# Finalize logger setup
		self.logger.setLevel(level)
		self.logger.info(f"Initialized HSSFLDON Logger with name: {name} and level: {logging.getLevelName(level)}!")

	def getLogger(self) -> logging.Logger:
		"""
		Get the configured logger for HSSFLDON.

		Returns:
			logging.Logger: The configured logger.
		"""
		return self.logger
	
	def log(self, level: int, message: str):
		"""
		Log a message with the specified level.

		Args:
			level (int): The logging level (e.g., logging.DEBUG, logging.INFO).
			message (str): The message to log.
		"""
		self.logger.log(level, message)

	def debug(self, message: str):
		"""
		Log a debug message.

		Args:
			message (str): The message to log.
		"""
		self.logger.debug(message)

	def info(self, message: str):
		"""
		Log an info message.

		Args:
			message (str): The message to log.
		"""
		self.logger.info(message)

	def warning(self, message: str):
		"""
		Log a warning message.

		Args:
			message (str): The message to log.
		"""
		self.logger.warning(message)

	def error(self, message: str):
		"""
		Log an error message.

		Args:
			message (str): The message to log.
		"""
		self.logger.error(message)

	def _setup_getConsoleHandler(self) -> logging.Handler:
		"""
		Setup a console handler for HSSFLDON.

		Returns:
			logging.Logger: The configured console handler.
		"""
		handler = logging.StreamHandler()
		handler.setLevel(self.level)
		handler.setFormatter(self.formatter)
		return handler
	
	def _setup_getGlobalFileHandler(self) -> logging.Handler:
		"""
		Setup a global file handler for HSSFLDON.

		Returns:
			logging.Logger: The configured global file handler.
		"""
		globalLogDirectory: Path = Path(os.getenv("HSSFLDON_LOGDIR", "logs/"))
		globalLogFilePath: Path = Path.joinpath(globalLogDirectory, 'global.log')
		os.makedirs(os.path.dirname(globalLogDirectory), exist_ok=True)
		handler = logging.FileHandler(globalLogFilePath)
		handler.setLevel(self.level)
		handler.setFormatter(self.formatter)
		return handler
	
	def _setup_getLocalFileHandler(self) -> logging.Handler:
		"""
		Setup a local file handler for HSSFLDON.

		Returns:
			logging.Logger: The configured local file handler.
		"""
		localLogDirectory: Path = Path(os.getenv("HSSFLDON_LOGDIR", "logs/"))
		localLogFilePath: Path = Path.joinpath(localLogDirectory, f'{self.name}.log')
		os.makedirs(os.path.dirname(localLogDirectory), exist_ok=True)
		handler = logging.FileHandler(localLogFilePath)
		handler.setLevel(self.level)
		handler.setFormatter(self.formatter)
		return handler
	
class HSSFLDON_TrainerCallbackLogger(TrainerCallback):
	"""
	A TrainerCallback for logging training progress in HSSFLDON.
	"""
	def __init__(self, logger: HSSFLDON_Logger):
		self.logger = logger

	def on_log(self, args, state, control, logs=None, **kwargs):
		if logs is not None:

			# Grab max_steps to create a fraction: e.g., "Step: 10/500"
			step_str = f"Step: {state.global_step}/{state.max_steps}"
			epoch_str = f"Epoch: {round(state.epoch, 2) if state.epoch else 'N/A'}"
			
			# Format the metrics
			metrics_str = " | ".join([f"{k}: {round(v, 6) if isinstance(v, float) else v}" for k, v in logs.items()])
			
			# Send it to the logger
			self.logger.debug(f"[HF Trainer] {step_str} | {epoch_str} | {metrics_str}")