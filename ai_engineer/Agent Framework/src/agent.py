import os
import asyncio
from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


async def main():
    try:
        # Initialize a chat agent with Azure OpenAI Responses
        agent = AzureOpenAIResponsesClient(
            endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
            deployment_name=os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_AI_API_VERSION"),
            api_key=os.getenv("AZURE_AI_PROJECT_KEY"),
        ).create_agent(
            name="HaikuBot",
            instructions="You are an upbeat assistant that writes beautifully.",
        )

        # Ensure the prompt is a string
        prompt = "Write a haiku about Microsoft Agent Framework."
        result = await agent.run(prompt)
        print(result)
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e)}")
        
        # Debug: Print environment variables (without sensitive data)
        print(f"Endpoint: {os.getenv('AZURE_AI_PROJECT_ENDPOINT') is not None}")
        print(f"Deployment: {os.getenv('AZURE_AI_MODEL_DEPLOYMENT_NAME')}")
        print(f"API Version: {os.getenv('AZURE_AI_API_VERSION')}")


if __name__ == "__main__":
    asyncio.run(main())