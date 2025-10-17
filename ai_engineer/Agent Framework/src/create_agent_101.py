import asyncio
import os
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv
from agent_framework import ChatMessage, TextContent, UriContent, Role

load_dotenv()  # Load environment variables from a .env file if present

endpoint = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")

agent = AzureOpenAIChatClient(
        endpoint=endpoint,
        credential=AzureCliCredential()
    ).create_agent(
    instructions="You are good at telling jokes.",
    name="Joker"
)

# async def main():
#     result = await agent.run("Tell me a joke about a pirate.")
#     print(result.text)



message = ChatMessage(
    role=Role.USER,
    contents=[
        TextContent(text="Tell me a joke about this image?"),
        UriContent(uri="https://img-s-msn-com.akamaized.net/tenant/amp/entityid/AA1Oe52j.img?w=768&h=512&m=6&x=1040&y=245&s=322&d=322", media_type="image/jpeg")
    ]
)

async def main():
    result = await agent.run(message)
    print(result.text)

asyncio.run(main())