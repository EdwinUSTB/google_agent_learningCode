import os
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_toolset import MCPToolset,HttpServerParameters

FASTMCP_SERVER_URL = "http://localhost:8000"

root_agent = LlmAgent(
    model="gemini-2.5-flash",
    system_prompt="你是一个助手，可以通过“greet”工具向人问好。",
    tools = [
        MCPToolset(
            connection_params=HttpServerParameters(
                url=FASTMCP_SERVER_URL,
            ),
            tool_filter=['greet'],
        )
    ],
)

