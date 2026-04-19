'''
	# HSSFLDON - Simulator

	This module will provide a simulator for HSSFLDON.

	This will allow for testing various conditions with the 
	server and client applications without needing to run them separately.
'''

# Library Imports
import os
import sys
import time
import signal
import subprocess

def main() -> None:
	
	# List for tracking all processes created by simulator
	serverProcess: subprocess.Popen[bytes] | None = None
	clientProcesses: list[subprocess.Popen[bytes]] = []
	

	try:
		# Start server entrypoint
		server: subprocess.Popen[bytes] = subprocess.Popen(
			args=[sys.executable, "-m", "hssfldon.server.server_main"],
			env=os.environ.copy(),
		)
		serverProcess = server

		# Wait for server to boot (replace with health check later)
		time.sleep(2)

		# Start clients
		client_count: int = 1
		for i in range(client_count):
			env: dict[str, str] = os.environ.copy()
			env["HSSFLDON_CLIENT_ID"] = str(i)
			client: subprocess.Popen[bytes] = subprocess.Popen(
				args=[sys.executable, "-m", "hssfldon.client.client_main"],
				env=env,
			)
			clientProcesses.append(client)

		# Wait for all clients to exit
		for process in clientProcesses:
			try:
				process.wait(timeout=30)
			except subprocess.TimeoutExpired:
				process.send_signal(sig=signal.SIGTERM)
				try:
					process.wait(timeout=5)
				except subprocess.TimeoutExpired:
					process.kill()
					process.wait(timeout=5)
			
	finally:

		# Quick grab of all processes
		allProcesses: list[subprocess.Popen[bytes] | None] = [serverProcess] + clientProcesses

		# Send termination signal to all processes
		for process in allProcesses:
			if process is not None and process.poll() is None:
				process.send_signal(sig=signal.SIGTERM)
				try:
					process.wait(timeout=5)
				except subprocess.TimeoutExpired:
					process.kill()

if __name__ == "__main__":
	main()