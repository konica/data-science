import asyncio
import os
from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file if present

async def main():
    async with (
        ChatAgent(
            chat_client=AzureAIAgentClient(
                endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
                api_key=os.getenv("AZURE_AI_PROJECT_KEY"),
                api_version=os.getenv("AZURE_AI_API_VERSION"),
            ),
            instructions="You are good at telling jokes."
        ) as agent,
    ):
        result = await agent.run("Tell me a joke about a pirate.")
        print(result.text)

if __name__ == "__main__":
    asyncio.run(main())