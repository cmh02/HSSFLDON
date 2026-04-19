'''
	# HSSFLDON - Server Application

	This will be the main entrypoint for the server in HSSFLDON.
	
	The server is responsible for:
	- Managing unlabeled dataset
	- Distributing unknown datapoints to clients for labeling
	- Aggregating client updates in global model



'''

# Project Imports
from hssfldon.server.server_app import HSSFLDON_ServerApplication

if __name__ == "__main__":
	server = HSSFLDON_ServerApplication()