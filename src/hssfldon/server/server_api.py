'''
	# HSSFLDON - Server API

	This module will provide the API for the HSSFLDON server application.
	
	The server is responsible for:
	- Managing unlabeled dataset
	- Distributing unknown datapoints to clients for labeling
	- Aggregating client updates in global model
'''

# Library Imports
import anyio
from typing import Optional
from fastapi import APIRouter, Request, status
from pydantic import BaseModel, Field

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

class RegisterClientRequest(BaseModel):
    """
    Request model for client registration.
    """
    client_id: int = Field(..., gt=0, description="Unique identifier for the client")

class RegisterClientResponse(BaseModel):
    """
    Response model for client registration.
    """
    status: str = Field(..., description="Status of the registration request")
    message: str = Field(..., description="Additional information about the registration result")
    client_id: Optional[int] = Field(None, description="The ID of the registered client, if successful")

@HSSFLDON_ServerAPIRouter.post("/register", response_model=RegisterClientResponse, status_code=status.HTTP_201_CREATED, tags=["Client Management"])
async def register_client(request: Request, payload: RegisterClientRequest):
    """
    API Endpoint: /register
    Method: POST
    Description: Register a new client with the server.
    """
    # Access the server application instance
    server_app = request.app.state.server_app

    # Await the registration
    await anyio.to_thread.run_sync(server_app.registerClient, payload.client_id)

    # Return success
    return RegisterClientResponse(status="ok", message=f"Client {payload.client_id} registered successfully!", client_id=payload.client_id)