# world.py — The orchestrator script (NOT an agent).
# Launches the distributed runtime, registers the Creator agent, then uses asyncio.gather()
# to spawn N agents in parallel. Each agent is created by the Creator, registered with the
# runtime, messaged for an idea, and the result is saved to ideaN.md.
# Uses async Python to run all creations concurrently (event loop, not threads).
# Run with: uv run world.py (or python world.py)

from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntimeHost
from agent import Agent
from creator import Creator
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime
from autogen_core import AgentId
import messages
import asyncio

# How many agents to spawn — each costs ~1 API call for creation + 1 for idea generation
HOW_MANY_AGENTS = 20


async def create_and_message(worker, creator_id, i: int):
    """Send a creation request to the Creator, save the resulting idea to a markdown file."""
    try:
        result = await worker.send_message(messages.Message(content=f"agent{i}.py"), creator_id)
        with open(f"idea{i}.md", "w") as f:
            f.write(result.content)
    except Exception as e:
        print(f"Failed to run worker {i} due to exception: {e}")


async def main():
    # Start the gRPC host (message router) and a single worker
    host = GrpcWorkerAgentRuntimeHost(address="localhost:50051")
    host.start()
    worker = GrpcWorkerAgentRuntime(host_address="localhost:50051")
    await worker.start()
    # Register the Creator agent — it will dynamically register child agents
    result = await Creator.register(worker, "Creator", lambda: Creator("Creator"))
    creator_id = AgentId("Creator", "default")
    # Launch all agent creations in parallel using asyncio.gather
    # Each coroutine: Creator writes code → imports it → registers it → messages it → saves idea
    coroutines = [create_and_message(worker, creator_id, i) for i in range(1, HOW_MANY_AGENTS + 1)]
    await asyncio.gather(*coroutines)
    try:
        await worker.stop()
        await host.stop()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    asyncio.run(main())
