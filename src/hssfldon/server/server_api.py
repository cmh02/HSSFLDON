'''
	# HSSFLDON - Server API

	This module will provide the API for the HSSFLDON server application.
	
	The server is responsible for:
	- Managing unlabeled dataset
	- Distributing unknown datapoints to clients for labeling
	- Aggregating client updates in global model
'''

# Library Imports
from fastapi import APIRouter

# Create router instance
HSSFLDON_ServerAPIRouter = APIRouter()

@HSSFLDON_ServerAPIRouter.get("/health", tags=["System"])
def health_check():
    """
    API Endpoint: /health
    Method: GET
    Description: Health check endpoint to verify that the server is online and responsive.
    """
    return {"status": "ok", "message": "System is online!"}