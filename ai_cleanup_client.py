# copy_cleanup_client.py - Complete Fixed Version
import asyncio
import os
import uuid
import json
import signal
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_fireworks import ChatFireworks
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langgraph.prebuilt import InjectedStore
from rich.console import Console
from typing_extensions import Annotated
from typing import List, Dict, Any, Optional
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph_bigtool.utils import convert_positional_only_function_to_tool
from ai_cleanup_graph import create_agent
from langgraph.errors import GraphRecursionError
import yaml

# Configuration
MAX_ITERATIONS = 50
RETRY_LIMIT = 1
RECURSION_LIMIT = 100

console = Console()
load_dotenv()


def preprocess_query(user_query: str) -> str:
    """Adds a [TREND_REQUEST] prefix for trend/history queries."""
    trend_keywords = ["trend", "history", "over the last", "past", "minutes", "hours"]
    if any(kw in user_query.lower() for kw in trend_keywords):
        return f"[TREND_REQUEST] {user_query}"
    return user_query


def load_yaml_context(file_path: str):
    """Load YAML file into a Python dict."""
    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        console.print(f"[green]✓ Loaded YAML context from {file_path}[/green]")
        return data
    except FileNotFoundError:
        console.print(f"[yellow]⚠ No YAML file found at {file_path}[/yellow]")
        return None
    except Exception as e:
        console.print(f"[red]✖ Error loading YAML file: {e}[/red]")
        return None


@dataclass
class MCPServerConfig:
    name: str
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    transport: str = "stdio"
    headers: Optional[Dict[str, str]] = None


class MCPServerManager:
    def __init__(self, config_path: str = "/root/amflw_chatbot/jenkins_automate/mcp_c_trial1/mcp_server_v2.json"):
        self.config_path = Path(config_path)
        self.servers: Dict[str, MCPServerConfig] = {}
        self.load_servers()

    def load_servers(self):
        if not self.config_path.exists():
            console.print(f"[yellow]⚠ No configuration file found at {self.config_path}[/yellow]")
            return

        try:
            with open(self.config_path, 'r') as f:
                servers_data = json.load(f)
                for name, data in servers_data.items():
                    self.servers[name] = MCPServerConfig(
                        name=name,
                        url=data.get("url"),
                        command=data.get("command"),
                        args=data.get("args", []),
                        transport=data.get("transport", "stdio"),
                        headers=data.get("headers")
                    )
            console.print(f"[green]✓ Loaded {len(self.servers)} server configurations[/green]")
        except Exception as e:
            console.print(f"[red]✖ Error loading server configurations: {e}[/red]")


class RemoteLangGraphAgent:
    def __init__(self, tools: Dict[str, Any]):
        self.tool_registry = tools
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.store = self._initialize_store()
        self.llm = self._get_llm()
        self.graph = None
        # Initialize a clean stable state
        self.current_state = {
            "messages": [],
            "selected_tool_ids": [],
            "circuit_breaker": {},
            "current_query": "",
            "query_context": {},
            "pending_tool_calls": []
        }

    def _initialize_store(self) -> InMemoryStore:
        store = InMemoryStore(
            index={
                "embed": self.embeddings,
                "dims": 384,
                "fields": ["description"],
            }
        )

        # Add all tools to vector store
        for tool_name, tool in self.tool_registry.items():
            desc = getattr(tool, "description", "No description")
            store.put(("tools",), tool_name, {"description": f"{tool_name}: {desc}"})

        # Add math tools
        import math
        import types
        for func_name in dir(math):
            func = getattr(math, func_name)
            if isinstance(func, types.BuiltinFunctionType):
                if lc_tool := convert_positional_only_function_to_tool(func):
                    tool_id = str(uuid.uuid4())
                    self.tool_registry[tool_id] = lc_tool
                    store.put(
                        ("tools",),
                        tool_id,
                        {"description": f"{lc_tool.name}: {lc_tool.description}"}
                    )

        return store

    def _get_llm(self):
        return ChatFireworks(
            model="accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
            max_tokens=131072,
            temperature=0.0
        )

    def _retrieve_tools_function(
        self,
        query: str,
        *,
        store: Annotated[BaseStore, InjectedStore],
    ) -> List[str]:
        results = store.search(("tools",), query=query, limit=10)
        return [result.key for result in results[:5]]

    async def _retrieve_tools_coroutine(
        self,
        query: str,
        *,
        store: Annotated[BaseStore, InjectedStore],
    ):
        return self._retrieve_tools_function(query, store=store)

    def initialize_agent(self):
        """Initialize agent with debug mode enabled"""
        builder = create_agent(
            self.llm,
            self.tool_registry,
            retrieve_tools_function=self._retrieve_tools_function,
            retrieve_tools_coroutine=self._retrieve_tools_coroutine,
            limit=5
        )
        self.graph = builder.compile(store=self.store, debug=True)
        console.print("[yellow]🔍 Debug mode ENABLED - will show message stack at each iteration[/yellow]")
        return self.graph

    def reset_conversation(self):
        self.current_state = {
            "messages": [],
            "selected_tool_ids": [],
            "circuit_breaker": {},
            "current_query": "",
            "query_context": {},
            "pending_tool_calls": []
        }
        console.print("[yellow]🔄 Conversation state reset[/yellow]")

    async def process_query(self, query: str):
        """Fast, reliable query handler with cleanup, short-circuiting, and reduced recursion."""
        import re

        query = query.strip()
        if not query:
            return "⚠️ Empty query provided. Please enter a valid question or command."

        # Maintain only the last few messages
        self.current_state["messages"] = self.current_state["messages"][-2:] + [HumanMessage(content=query)]

        attempt, final_response = 0, None
        tool_call_count = 0  # Track tool executions

        while attempt <= RETRY_LIMIT:
            iteration = 0
            try:
                print(f"🔁 Starting execution attempt {attempt + 1}...")
                async for step in self.graph.astream(
                    self.current_state,
                    config={"recursion_limit": RECURSION_LIMIT},
                    stream_mode="updates",
                ):
                    iteration += 1
                    if iteration > MAX_ITERATIONS:
                        print("⚠️ Circuit breaker triggered — too many iterations.")
                        raise GraphRecursionError("Exceeded MAX_ITERATIONS limit")

                    for node_name, update in step.items():
                        if node_name in ("agent", "tools", "output"):
                            print(f"🧠 Node: {node_name}")

                        for msg in update.get("messages", []):
                            if isinstance(msg, AIMessage):
                                if msg.tool_calls:
                                    tool_call_count += len(msg.tool_calls)  # Count tool calls
                                elif msg.content.strip():
                                    final_response = msg.content
                                    print(f"💡 AI: {msg.content[:200]}...")
                                    
                                    # Only short-circuit if we've executed tools AND got final answer
                                    if tool_call_count > 0 and re.search(
                                        r"(successfully|completed|created|here is|done|result)",
                                        msg.content,
                                        re.I
                                    ):
                                        print("🛑 Short-circuit: Final response after tool execution.")
                                        raise StopAsyncIteration
                                        
                            elif isinstance(msg, ToolMessage):
                                print(f"🛠️ Tool Result: {msg.content[:200]}...")

                print(f"✅ Completed in {iteration} iterations.")
                break

            except StopAsyncIteration:
                print("✅ Early termination — final answer found.")
                break
            except GraphRecursionError as e:
                print(f"⚠️ Recursion detected on attempt {attempt + 1}: {str(e)}")
                if attempt < RETRY_LIMIT:
                    print("🔁 Retrying once with clean state...")
                    self.reset_conversation()
                    attempt += 1
                    await asyncio.sleep(0.3)
                    continue
                else:
                    print("❌ Max retries reached.")
                    break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                import traceback
                traceback.print_exc()
                break

        # Clean up & remove duplicates before returning
        if isinstance(final_response, str):
            final_response = final_response.strip()
            lines = final_response.splitlines()
            seen, cleaned = set(), []
            for line in lines:
                if line.strip() and line.strip() not in seen:
                    cleaned.append(line)
                    seen.add(line.strip())
            final_response = "\n".join(cleaned)

        return final_response or "No final AI response received."


async def main():
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        console.print("\n[yellow]👋 Shutting down gracefully...[/yellow]")
        raise KeyboardInterrupt
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    server_manager = MCPServerManager()
    os.environ["FIREWORKS_API_KEY"] = "fw_3ZPF2zBvheEvfSGRRoChVZPc"
    console.print("[blue]🚀 Starting MCP client with remote support...[/blue]")

    if not server_manager.servers:
        console.print("[yellow]⚠ No servers configured in mcp_servers.json[/yellow]")
        return

    # Show available servers
    console.print("\n[bold]Available Servers:[/bold]")
    server_names = list(server_manager.servers.keys())
    for i, name in enumerate(server_names, 1):
        server = server_manager.servers[name]
        if server.url:
            console.print(f"{i}. {name} → {server.url} ({server.transport})")
        elif server.command:
            cmd = f"{server.command} {' '.join(server.args or [])}"
            console.print(f"{i}. {name} → {cmd} ({server.transport})")

    # Select servers
    console.print("\n[bold]Select servers to connect:[/bold]")
    console.print("• Enter numbers (e.g., 1,2)")
    console.print("• Enter 'all' to connect to all")
    selection = input("\n🔢 Select: ").strip()

    selected_servers = []
    if selection.lower() == 'all':
        selected_servers = server_names
    else:
        try:
            indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
            for idx in indices:
                if 0 <= idx < len(server_names):
                    selected_servers.append(server_names[idx])
                else:
                    console.print(f"[red]✖ Invalid: {idx + 1}[/red]")
        except ValueError:
            console.print("[red]✖ Invalid input[/red]")
            return

    if not selected_servers:
        console.print("[red]✖ No servers selected[/red]")
        return

    # Build MCP client config
    mcp_client_config = {}
    console.print(f"[green]✓ Selected {len(selected_servers)} servers: {', '.join(selected_servers)}[/green]")
    
    for name in selected_servers:
        server = server_manager.servers[name]
        if server.url:
            mcp_client_config[name] = {
                "transport": server.transport,
                "url": server.url,
                "headers": server.headers or {}
            }
        elif server.command:
            mcp_client_config[name] = {
                "transport": server.transport,
                "command": server.command,
                "args": server.args or [],
            }

    # Create MultiServerMCPClient
    console.print("[cyan]🔗 Connecting to servers via MultiServerMCPClient...[/cyan]")
    client = None
    try:
        client = MultiServerMCPClient(mcp_client_config)
        tools = await client.get_tools()
        tool_dict = {tool.name: tool for tool in tools}
        console.print(f"[green]✓ Loaded {len(tools)} tools from {len(selected_servers)} servers[/green]")
        console.print("Available tools:")
        for tool in tools:
            console.print(f"  • {tool.name}: {getattr(tool, 'description', 'No description')[:80]}...")
            
    except Exception as e:
        console.print(f"[red]✖ Failed to connect: {e}[/red]")
        import traceback
        traceback.print_exc()
        return

    # Initialize agent
    console.print("[blue]⏳ Initializing LangGraph BigTool Agent...[/blue]")
    agent = RemoteLangGraphAgent(tool_dict)
    agent.initialize_agent()

    console.print("[bold green]🧠 Agent Ready![/bold green]")
    console.print("Type 'exit' to quit, 'reset' to clear history.")
    
    # Load and inject YAML context if Prometheus server is selected
    if "prometheus-grafana" in selected_servers:
        yaml_path = "prometheus_123.yml"
        yaml_data = load_yaml_context(yaml_path)
        if yaml_data:
            system_msg = SystemMessage(content=f"YAML Context:\n{yaml_data}")
            agent.current_state["messages"] = [system_msg]

    try:
        while True:
            # Get user input
            raw_query = input("\n🔎 Query: ").strip()

            # Handle special commands
            if raw_query.lower() in ('exit', 'quit'):
                break
            elif raw_query.lower() == 'reset':
                agent.reset_conversation()
                continue

            # Preprocess for trend/history queries
            user_query = preprocess_query(raw_query)

            # Send to agent
            result = await agent.process_query(user_query)
            console.print(f"\n[green]📋 Final Result:[/green]\n{result}")

    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Interrupted by user[/yellow]")
    finally:
        # Cleanup
        console.print("[yellow]🔄 Closing connections...[/yellow]")
        if client:
            try:
                if hasattr(client, 'aclose'):
                    await client.aclose()
                elif hasattr(client, 'close'):
                    await client.close()
            except Exception as e:
                console.print(f"[yellow]⚠️ Cleanup warning: {e}[/yellow]")

        # Force close any remaining tasks
        pending = [t for t in asyncio.all_tasks() if not t.done() and t != asyncio.current_task()]
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        # Give time for cleanup
        await asyncio.sleep(0.5)
        console.print("[green]✓ Cleanup complete[/green]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Exiting...[/yellow]")
    except Exception as e:
        console.print(f"[red]✖ Fatal error: {e}[/red]")
        import traceback
        traceback.print_exc()