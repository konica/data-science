from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

project = AIProjectClient(
  endpoint="https://test-azure-aif-101.services.ai.azure.com/api/projects/firstProject",
  credential=DefaultAzureCredential())
# The AIProjectClient lets you access models, data, and services in your project.


models = project.get_openai_client(api_version="2025-01-01-preview")
response = models.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful writing assistant"},
        {"role": "user", "content": "Write me a poem about flowers"},
    ],
)

print(response.choices[0].message.content)