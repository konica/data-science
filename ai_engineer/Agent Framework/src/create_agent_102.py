import asyncio
import os
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv
from agent_framework import ChatMessage, TextContent, UriContent, Role, DataContent

load_dotenv()  # Load environment variables from a .env file if present

endpoint = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")

agent = AzureOpenAIChatClient(
        endpoint=endpoint,
        credential=AzureCliCredential()
    ).create_agent(
    name="VisionAgent",
    instructions="You are a helpful agent that can analyze images"
)

# message = ChatMessage(
#     role=Role.USER,
#     contents=[
#         TextContent(text="What do you see in this image?"),
#         UriContent(
#             uri="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
#             media_type="image/jpeg"
#         )
#     ]
# )

# Load image from local file
with open("./data/AA1OcRaZ.jpg", "rb") as f:
    image_bytes = f.read()

message = ChatMessage(
    role=Role.USER,
    contents=[
        TextContent(text="What do you see in this image?"),
        DataContent(
            data=image_bytes,
            media_type="image/jpeg"
        )
    ]
)
async def main():
    result = await agent.run(message)
    print(result.text)

asyncio.run(main())