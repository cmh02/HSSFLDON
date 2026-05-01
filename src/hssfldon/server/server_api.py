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
import torch
from typing import Optional, TYPE_CHECKING
from fastapi import APIRouter, Request, status, Depends
from pydantic import BaseModel, Field

# Project Imports
if TYPE_CHECKING:
    from hssfldon.server.server_app import HSSFLDON_ServerApplication

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
    client_id: int = Field(..., gte=0, description="Unique identifier for the client") # type: ignore

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
    server_app: "HSSFLDON_ServerApplication" = request.app.state.server_app

    # Await the registration
    await anyio.to_thread.run_sync(server_app.registerClient, payload.client_id) # type: ignore

    # Return success
    return RegisterClientResponse(status="ok", message=f"Client {payload.client_id} registered successfully!", client_id=payload.client_id)

class GetTaskRequest(BaseModel):
    """
    Request model for fetching the next task for a client.
    """
    client_id: int = Field(..., gte=0, description="Unique identifier for the client") # type: ignore

class GetTaskResponse(BaseModel):
    """
    Response model for fetching the next task for a client.
    """
    status: str = Field(..., description="Status of the request")
    message: str = Field(..., description="Additional information about the request result")
    task: Optional[str] = Field(None, description="The next task assigned to the client, if available")

@HSSFLDON_ServerAPIRouter.get("/task", response_model=GetTaskResponse, status_code=status.HTTP_200_OK, tags=["Client Management"])
async def getClientTask(request: Request, payload: GetTaskRequest = Depends()):
    """
    API Endpoint: /task
    Method: GET
    Description: Fetch the next task for a client.
    """
    # Access the server application instance
    server_app: "HSSFLDON_ServerApplication" = request.app.state.server_app

    # Get task from mapping
    task = server_app.clientTasks.get(payload.client_id, None)

    if task is not None:
        return GetTaskResponse(status="ok", message=f"Next task for client {payload.client_id} fetched successfully!", task=task.name)
    else:
        return GetTaskResponse(status="ok", message=f"No tasks available for client {payload.client_id} at this time.", task=None)

class SubmitUpdateRequest(BaseModel):
    """
    Request model for submitting a client update.
    """
    client_id: int = Field(..., gte=0, description="Unique identifier for the client") # type: ignore
    head_path: str = Field(..., description="Path to the client's classification head!")

class SubmitUpdateResponse(BaseModel):
    """
    Response model for submitting a client update.
    """
    status: str = Field(..., description="Status of the submission request")
    message: str = Field(..., description="Additional information about the submission result")

@HSSFLDON_ServerAPIRouter.post("/submit_update", response_model=SubmitUpdateResponse, status_code=status.HTTP_200_OK, tags=["Client Management"])
async def submitClientUpdate(request: Request, payload: SubmitUpdateRequest):
    """
    API Endpoint: /submit_update
    Method: POST
    Description: Submit a client update to the server.
    """
    # Access the server application instance
    server_app: "HSSFLDON_ServerApplication" = request.app.state.server_app

    # Mark as received and store path
    server_app.clientUpdateStatus[payload.client_id] = True
    server_app.clientHeadPathCache[payload.client_id] = payload.head_path

    return SubmitUpdateResponse(status="ok", message=f"Client {payload.client_id} update submitted successfully!")

class SubmitEvaluationRequest(BaseModel):
    """
    Request model for submitting a client evaluation.
    """
    client_id: int = Field(..., gte=0, description="Unique identifier for the client") # type: ignore
    evaluation_results: dict = Field(..., description="Evaluation results from the client")

class SubmitEvaluationResponse(BaseModel):
    """
    Response model for submitting a client evaluation.
    """
    status: str = Field(..., description="Status of the submission request")
    message: str = Field(..., description="Additional information about the submission result")

@HSSFLDON_ServerAPIRouter.post("/submit_evaluation", response_model=SubmitEvaluationResponse, status_code=status.HTTP_200_OK, tags=["Client Management"])
async def submitClientEvaluation(request: Request, payload: SubmitEvaluationRequest):
    """
    API Endpoint: /submit_evaluation
    Method: POST
    Description: Submit a client evaluation to the server.
    """
    # Access the server application instance
    server_app: "HSSFLDON_ServerApplication" = request.app.state.server_app

    # Process evaluation results (for now, just log them) TODO: implement
    server_app.logger.info(f"Received evaluation results from client {payload.client_id}: {payload.evaluation_results}")

    return SubmitEvaluationResponse(status="ok", message=f"Client {payload.client_id} evaluation submitted successfully!")

class GetActiveDatapointsRequest(BaseModel):
    """
    Request model for fetching active datapoints for a client.
    """
    client_id: int = Field(..., gte=0, description="Unique identifier for the client") # type: ignore

class GetActiveDatapointsResponse(BaseModel):
    """
    Response model for fetching active datapoint for a client.
    """
    status: str = Field(..., description="Status of the request")
    message: str = Field(..., description="Additional information about the request result")
    datapoint: str | None = Field(None, description="A text to be labeled by the client")
    labels: list | None = Field(None, description="Labels for the active datapoint")

@HSSFLDON_ServerAPIRouter.get("/active_datapoint", response_model=GetActiveDatapointsResponse, status_code=status.HTTP_200_OK, tags=["Client Management"])
async def getActiveDatapoint(request: Request, payload: GetActiveDatapointsRequest = Depends()):
    """
    API Endpoint: /active_datapoint
    Method: GET
    Description: Fetch an active datapoint for a client to label.
    """
    # Access the server application instance
    server_app: "HSSFLDON_ServerApplication" = request.app.state.server_app

    # Get active datapoint for client
    datapoint = server_app.clientActiveLearningDatapointCache.get(payload.client_id, None)

    if datapoint is not None:
        return GetActiveDatapointsResponse(
            status="ok",
            message=f"Active datapoint for client {payload.client_id} fetched successfully!", 
            datapoint=datapoint,
            labels=datapoint.get("labels")
        )
    else:
        return GetActiveDatapointsResponse(
            status="ok", 
            message=f"No active datapoints available for client {payload.client_id} at this time.", 
            datapoint=None, 
            labels=None
        )