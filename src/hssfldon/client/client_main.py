'''
	# HSSFLDON - Client Application

	This module will provide the client application for HSSFLDON.
	
	The client is responsible for:
	- Managing localized labeled dataset
	- Training local model
	- Receiving active labeling requests
	- Uploading model updates to server
'''

# Project Imports
from hssfldon.client.client_app import HSSFLDON_ClientApplication

if __name__ == "__main__":
	client_app = HSSFLDON_ClientApplication()