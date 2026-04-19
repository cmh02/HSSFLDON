'''
	# HSSFLDON - Logger

	This common module will provide a logger for use anywhere in HSSFLDON.
'''

### Library Imports
import os
import logging
from pathlib import Path
from dotenv import load_dotenv


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
		self.formatter: logging.Formatter = logging.Formatter(f'%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
		self.logger.log(logging.INFO, f"Initialized HSSFLDON Logger with name: {name} and level: {logging.getLevelName(level)}!")


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
		handler = logging.FileHandler(localLogFilePath)
		handler.setLevel(self.level)
		handler.setFormatter(self.formatter)
		return handler