import asyncio
import os
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


server_params = StdioServerParameters(
    command="python",
    args=["mcp_server.py"],  # 👈 your MCP server
)


async def run_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            # Initialize MCP connection
            await session.initialize()

            # Load MCP tools dynamically
            tools = await load_mcp_tools(session)

            # Create LangGraph ReAct agent
            agent = create_react_agent(model, tools)

            # Example query
            user_input = input("Ask your HR assistant: ")

            agent_response = await agent.ainvoke({
                "messages": user_input
            })

            return agent_response


if __name__ == "__main__":
    result = asyncio.run(run_agent())


    print("Final Answer:")
    print(result["messages"][-1].content)