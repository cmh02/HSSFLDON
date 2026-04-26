'''
	# HSSFLDON - Enums

	This common module will provide enums for use anywhere in HSSFLDON.
'''

# Library Imports
from enum import Enum

class HSSFLDON_ServerState(Enum):
    """
    Enum for the different states of the HSSFLDON server.
    """

    # General state for server is idle and waiting on some trigger
    IDLE = 0

    # State for when server is waiting on clients to register
    WAITING_CLIENT_REGISTRATION = 1

    # State for when server is aggregating client updates and updating global model
    AGGREGATING = 9

    # State for when server is in passive learning mode
    PASSIVE_LEARNING = 10

    # State for when server is in active learning mode
    ACTIVE_LEARNING = 11

class HSSFLDON_ClientState(Enum):
    """
    Enum for the different states of the HSSFLDON clients.
    """

    # General state for client is idle and waiting on some trigger
    IDLE = 0

    # State for when client is waiting on task from server
    WAITING_FOR_TASK = 1

    # State for when client is performing passive learning on local dataset
    PASSIVE_LEARNING = 2

    # State for when client is performing active learning on assigned datapoint from server
    ACTIVE_LEARNING = 3

    # State for when client is sending an update to server
    SENDING_UPDATE = 4

    # State for when client is evaluating local model
    EVALUATING = 5

    # State for when client is sending evaluation results to server
    SENDING_EVALUATION = 6

class HSSFLDON_ClientTask(Enum):
    """
    Enum for the different tasks that can be assigned to HSSFLDON clients.
    """

    # Task for a client to standby and wait a period of time before checking again
    STANDBY = 0

    # Task for a client to perform passive learning on local dataset and send update to server
    DO_PASSIVE_LEARNING = 1

    # Task for a client to get a datapoint and perform active learning on it and send update to server
    DO_ACTIVE_LEARNING = 2

    # Task for a client to evaluate and send evaluation results to server
    DO_EVALUATION = 3

class HSSFLDON_PredictionOutputType(Enum):
    """
    Enum for the different types of prediction outputs that can be returned by HSSFLDON clients.
    """

    # Output type for a client to return a binary prediction for each category
    BINARY_PREDICTION = 0

    # Output type for a client to return a probability for each category
    PROBABILITY_PREDICTION = 1

    # Output type for a client to return logits for each category
    LOGIT_PREDICTION = 2

    # Output type for a client to return an embedding vector for each datapoint
    EMBEDDING_PREDICTION = 3