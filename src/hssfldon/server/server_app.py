'''
	# HSSFLDON - Server Application

	This module will provide the server application for HSSFLDON.
	
	The server is responsible for:
	- Managing unlabeled dataset
	- Distributing unknown datapoints to clients for labeling
	- Aggregating client updates in global model



'''

# Project Imports
from hssfldon.common.hssfldon_logger import HSSFLDON_Logger


class HSSFLDON_ServerApplication:
	"""
	The main server application for HSSFLDON.
	"""
	def __init__(self):

		# Get logger
		self.logger = HSSFLDON_Logger(name="HSSFLDON_ServerApplication").getLogger()

		pass

	