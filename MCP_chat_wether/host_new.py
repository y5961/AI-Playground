import asyncio
from contextlib import AsyncExitStack
from typing import Any

import google.generativeai as genai
from google.generativeai import GenerativeModel
from client import MCPClient
from dotenv import load_dotenv
import os

load_dotenv()


class ChatHost:
    def __init__(self):
        self.mcp_clients: list[MCPClient] = [
            MCPClient("./weather_USA.py"),
            MCPClient("./weather_Israel.py")
        ]
        self.tool_clients: dict[str, tuple[MCPClient, str]] = {}
        self.clients_connected = False
        self.exit_stack = AsyncExitStack()
        
        # Google Gemini setup
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        
        genai.configure(api_key=api_key)
        self.model = GenerativeModel("gemini-1.5-flash")

    async def connect_mcp_clients(self):
        """Connect all configured MCP clients once."""
        if self.clients_connected:
            return

        for client in self.mcp_clients:
            if client.session is None:
                await client.connect_to_server()

        if not self.mcp_clients:
            raise RuntimeError("No MCP clients are connected")

        self.clients_connected = True

    async def get_available_tools(self) -> list[dict[str, Any]]:
        """Collect tools from all MCP clients and map them back to their owner."""
        await self.connect_mcp_clients()
        self.tool_clients = {}
        available_tools: list[dict[str, Any]] = []

        for client in self.mcp_clients:
            if client.session is None:
                print(f"Warning: MCP client {client.client_name} is not connected, skipping")
                continue

            try:
                response = await client.session.list_tools()
                for tool in response.tools:
                    exposed_name = f"{client.client_name}__{tool.name}"
                    if exposed_name in self.tool_clients:
                        raise RuntimeError(f"Duplicate tool name detected: {exposed_name}")

                    self.tool_clients[exposed_name] = (client, tool.name)
                    available_tools.append(
                        {
                            "name": exposed_name,
                            "description": f"[{client.client_name}] {tool.description}",
                            "input_schema": tool.inputSchema,
                        }
                    )
            except Exception as e:
                print(f"Warning: Failed to get tools from {client.client_name}: {str(e)}")
                continue

        if not available_tools:
            raise RuntimeError("No tools available from any MCP client")

        return available_tools

    async def process_query(self, query: str) -> str:
        """Process a query using Google Gemini and available tools"""
        system_prompt = """You are a focused weather assistant. Your role is to:
1. Answer weather questions concisely and directly
2. Use the available weather tools to get current information
3. Provide ONLY relevant weather information without extra commentary
4. Keep responses brief and focused on the user's question
5. Use Hebrew if user writes in Hebrew, English if user writes in English"""
        
        messages = [{"role": "user", "parts": [{"text": system_prompt + "\n\nUser question: " + query}]}]
        available_tools = await self.get_available_tools()
        final_text = []

        while True:
            # Prepare tools for Gemini
            tool_list = [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
                for tool in available_tools
            ]

            response = self.model.generate_content(
                messages,
                tools=tool_list if tool_list else None,
                generation_config={"temperature": 0.7}
            )

            # Handle text content
            if response.text:
                final_text.append(response.text)

            # Handle tool calls
            tool_results = []
            saw_tool_use = False
            
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    saw_tool_use = True
                    tool_name = tool_call.function_name
                    tool_args = dict(tool_call.args) if tool_call.args else {}

                    if tool_name not in self.tool_clients:
                        raise RuntimeError(f"Unknown tool requested by model: {tool_name}")

                    client, original_tool_name = self.tool_clients[tool_name]
                    if client.session is None:
                        raise RuntimeError(f"MCP client {client.client_name} is not connected")

                    try:
                        result = await client.session.call_tool(original_tool_name, tool_args)
                        
                        # Extract result content
                        result_text = ""
                        if result.content:
                            for content_block in result.content:
                                if hasattr(content_block, 'text'):
                                    result_text += content_block.text
                        
                        tool_results.append(result_text if result_text else "Tool executed successfully")
                    except Exception as tool_error:
                        error_text = f"Tool execution error: {str(tool_error)}"
                        tool_results.append(error_text)
                        
                # Add tool results to messages for next iteration
                for result in tool_results:
                    messages.append({
                        "role": "user",
                        "parts": [{"text": f"Tool result: {result}"}]
                    })
            
            if not saw_tool_use:
                break
            
            # Add assistant response to messages for context (even if no text, tool was called)
            messages.append({
                "role": "model",
                "parts": [{"text": response.text or "[Tool called]"}]
            })

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\n🌤️ MCP Weather Client Started!")
        print("Type your queries or 'quit' to exit.\n")
        
        while True:
            try:
                query = input("Query: ").strip()
                
                if query.lower() == 'quit':
                    break
                
                response = await self.process_query(query)
                print("\n" + response)
                
            except Exception as e:
                print(f"\nError: {str(e)}")
                
    async def cleanup(self):
        """Clean up resources"""
        for client in reversed(self.mcp_clients):
            await client.cleanup()
        await self.exit_stack.aclose()
        
        
async def main():
    host = ChatHost()
    try:
        await host.chat_loop()
    finally:
        await host.cleanup()
        
if __name__ == "__main__":
    asyncio.run(main())
