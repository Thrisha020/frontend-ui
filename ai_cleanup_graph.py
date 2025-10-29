# cleanup_graph.py

from typing import Annotated, Callable, Optional, List, Dict, Any
import re
import uuid

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Send
from langgraph.utils.runnable import RunnableCallable

from langgraph_bigtool.tools import get_default_retrieval_tool, get_store_arg


def _add_new(left: list, right: list) -> list:
    """Extend left list with new items from right list."""
    return left + [item for item in right if item not in set(left)]


class State(MessagesState):
    selected_tool_ids: Annotated[list[str], _add_new]
    current_query: str = ""
    query_context: dict = {}
    pending_tool_calls: list = []
    circuit_breaker: dict = {}


def _format_selected_tools(selected_tools: dict, tool_registry: Dict[str, BaseTool]) -> tuple[list[ToolMessage], list[str]]:
    tool_messages, tool_ids = [], []
    for tool_call_id, batch in selected_tools.items():
        tool_names = []
        for result in batch:
            tool_obj = tool_registry.get(result)
            if hasattr(tool_obj, 'name'):
                tool_names.append(tool_obj.name)
            elif hasattr(tool_obj, '__name__'):
                tool_names.append(tool_obj.__name__)
            else:
                tool_names.append(str(result))
        tool_messages.append(
            ToolMessage(f"Available tools: {', '.join(tool_names)}", tool_call_id=tool_call_id)
        )
        tool_ids.extend(batch)
    return tool_messages, tool_ids


def _is_similar_query(prev_query: str, current_query: str, similarity_threshold: float = 0.7) -> bool:
    """Check if two queries are similar enough to maintain tool context."""
    if not prev_query or not current_query:
        return False
    prev_words = set(prev_query.lower().split())
    current_words = set(current_query.lower().split())
    union = prev_words.union(current_words)
    if not union:
        return False
    similarity = len(prev_words.intersection(current_words)) / len(union)
    return similarity >= similarity_threshold


def _validate_tool_parameters(tool_call: dict, tool_registry: Dict[str, BaseTool]) -> Optional[str]:
    """Validate tool parameters and return missing required parameters."""
    tool_name = tool_call["name"]
    args = tool_call["args"]

    tool = None
    for t in tool_registry.values():
        if getattr(t, 'name', '') == tool_name:
            tool = t
            break
    if not tool or not hasattr(tool, 'args_schema'):
        return None

    try:
        schema = tool.args_schema.schema()
        required_params = schema.get('required', [])
        missing_params = []
        for param in required_params:
            value = args.get(param)
            if (
                value is None
                or value == ""
                or (isinstance(value, str) and any(w in value.lower() for w in ["new", "sample", "example", "test", "temp"]))
            ):
                missing_params.append(param)
        if missing_params:
            return f"Missing required parameters for {tool_name}: {', '.join(missing_params)}"
    except Exception:
        pass
    return None


def validate_tool_selection(state: State, config: RunnableConfig, *, store: BaseStore) -> State:
    """
    ✅ CRITICAL: Remove AI messages using RemoveMessage
    Manager's instruction: Keep only System → Human → Tool messages
    """
    messages = state["messages"]
    
    # 🧹 Find all AI messages to remove
    messages_to_remove = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            messages_to_remove.append(RemoveMessage(id=msg.id))
    
    if messages_to_remove:
        print(f"🧹 [VALIDATION] Removing {len(messages_to_remove)} AI messages")
    
    # Debug: Show what remains
    remaining = [m for m in messages if not isinstance(m, AIMessage)]
    print(f"[VALIDATION] Message order after cleanup:")
    for i, msg in enumerate(remaining):
        print(f"  {i+1}. {type(msg).__name__}")
    
    # Get current query
    user_queries = [msg.content for msg in remaining if isinstance(msg, HumanMessage)]
    current_query = user_queries[-1] if user_queries else ""
    
    # Reset on new query
    if state.get("current_query") and not _is_similar_query(state.get("current_query"), current_query):
        return {
            "messages": messages_to_remove,
            "selected_tool_ids": [],
            "current_query": current_query,
            "pending_tool_calls": [],
            "circuit_breaker": {}
        }
    
    return {
        "messages": messages_to_remove,
        "current_query": current_query
    }


async def avalidate_tool_selection(state: State, config: RunnableConfig, *, store: BaseStore) -> State:
    """Async version of validate_tool_selection."""
    return validate_tool_selection(state, config, store=store)


def validate_parameters(state: State, config: RunnableConfig, *, store: BaseStore, tool_registry: dict) -> State:
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"pending_tool_calls": []}

    pending_tool_calls, valid_tool_calls = [], []
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "retrieve_relevant_tools":
            continue
        validation_error = _validate_tool_parameters(tool_call, tool_registry)
        if validation_error:
            pending_tool_calls.append({"tool_call": tool_call, "missing_info": validation_error})
        else:
            valid_tool_calls.append(tool_call)

    if pending_tool_calls:
        missing_params_message = "I need more information to proceed:\n"
        for pending in pending_tool_calls:
            missing_params_message += f"- {pending['missing_info']}\n"
        missing_params_message += "\nPlease provide the required information."
        return {
            "messages": [AIMessage(content=missing_params_message)],
            "pending_tool_calls": pending_tool_calls
        }
    return {"pending_tool_calls": []}


async def avalidate_parameters(state: State, config: RunnableConfig, *, store: BaseStore, tool_registry: dict) -> State:
    return validate_parameters(state, config, store=store, tool_registry=tool_registry)


def handle_user_input(state: State, config: RunnableConfig, *, store: BaseStore, tool_registry: dict) -> State:
    """Handle user input for missing parameters and create proper tool calls."""
    messages = state["messages"]
    pending_tool_calls = state.get("pending_tool_calls", [])
    if not pending_tool_calls or not messages:
        return state
    last_message = messages[-1]
    if not isinstance(last_message, HumanMessage):
        return state

    user_input = last_message.content
    extracted_params: Dict[str, Any] = {}

    for pending in pending_tool_calls:
        tool_call = pending["tool_call"]
        tool_name = tool_call["name"]

        tool = None
        for t in tool_registry.values():
            if getattr(t, 'name', '') == tool_name:
                tool = t
                break

        if tool and hasattr(tool, 'args_schema'):
            try:
                schema = tool.args_schema.schema()
                properties = schema.get('properties', {})
                for param_name in properties:
                    patterns = [
                        rf"{param_name}\s*[:=]\s*([^\n]+)",
                        rf"{param_name}.*?\bis\b\s*([^\n]+)",
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, user_input, re.IGNORECASE)
                        if match:
                            extracted_params[param_name] = match.group(1).strip()
                            break
            except:
                continue

    updated_tool_calls = []
    for pending in pending_tool_calls:
        new_call = pending["tool_call"].copy()
        new_call["args"] = {**new_call.get("args", {}), **extracted_params}
        updated_tool_calls.append(new_call)

    new_ai_message = AIMessage(
        content="Proceeding with provided parameters.",
        tool_calls=updated_tool_calls
    )
    return {"messages": [new_ai_message], "pending_tool_calls": []}


async def ahandle_user_input(state: State, config: RunnableConfig, *, store: BaseStore, tool_registry: dict) -> State:
    return handle_user_input(state, config, store=store, tool_registry=tool_registry)


def call_model(
    state: State,
    config: RunnableConfig,
    *,
    store: BaseStore,
    llm: LanguageModelLike,
    retrieve_tools: BaseTool,
    tool_registry: dict,
    active_server: str = "jenkins"
) -> State:
    """
    ✅ Call model with CLEAN message stack
    Should only see: SystemMessage → HumanMessage → ToolMessage(s)
    NO AI messages
    """
    
    print("\n[AGENT] 📋 Messages entering call_model:")
    for i, m in enumerate(state["messages"]):
        msg_type = type(m).__name__
        content_preview = str(m.content)[:80] if hasattr(m, 'content') else ""
        print(f"  {i+1}. {msg_type}: {content_preview}")
    
    # Count AI messages (should be 0)
    ai_count = sum(1 for msg in state["messages"] if isinstance(msg, AIMessage))
    if ai_count > 0:
        print(f"⚠️  WARNING: Found {ai_count} AIMessage(s) - cleanup failed!")
    else:
        print(f"✅ No AI messages found - cleanup working correctly")
    
    # Circuit breaker
    breaker_count = state.get("circuit_breaker", {}).get("count", 0)
    breaker_count += 1
    if breaker_count > 10:
        return {
            "messages": [AIMessage(content="⚠️ Stopping due to excessive reasoning loops.")],
            "circuit_breaker": {"count": breaker_count},
        }

    selected_tools = [tool_registry[id] for id in state["selected_tool_ids"]]
    llm_with_tools = llm.bind_tools([retrieve_tools, *selected_tools])

    messages = state["messages"].copy()
    system_message_content = f"""IMPORTANT: You are a multi-server assistant with access to tools.
Active platform default: {active_server}

### 🔑 CORE RULES:
1. NEVER respond with shell commands like `git clone ...` — use real tools.
2. Use correct server-specific tools (e.g., jenkins_trigger_job).
3. If platform is ambiguous (e.g., "clone repo"), ASK FOR CLARIFICATION.
4. NEVER invent parameter values. If missing, ask the user.
5. DO NOT FALL BACK TO TEXT INSTRUCTIONS IF A TOOL EXISTS.

When calling tools:
- NEVER generate random or placeholder values (like "job_name", "default").
- If parameters are missing, ASK THE USER for clarification.

For ambiguous actions across platforms (create/list repository, issue creation, etc.), ALWAYS ask which server to use.

EXAMPLE:
User: "create a new repository"
→ Ask: "Which platform would you like to create the repository on? Options: GitHub, JFrog Artifactory"

Always confirm the platform first when request applies to multiple systems!"""

    system_message = SystemMessage(content=system_message_content)

    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages.insert(0, system_message)

    response = llm_with_tools.invoke(messages)
    return {"messages": [response], "circuit_breaker": {"count": breaker_count}}


async def acall_model(
    state: State,
    config: RunnableConfig,
    *,
    store: BaseStore,
    llm: LanguageModelLike,
    retrieve_tools: BaseTool,
    tool_registry: dict,
    active_server: str = "jenkins"
) -> State:
    """Async version of call_model."""
    
    print("\n[AGENT] 📋 Messages entering call_model (async):")
    for i, m in enumerate(state["messages"]):
        print(f"  {i+1}. {type(m).__name__}")
    
    ai_count = sum(1 for msg in state["messages"] if isinstance(msg, AIMessage))
    if ai_count > 0:
        print(f"⚠️  WARNING: {ai_count} AIMessage(s) present!")
    else:
        print(f"✅ Clean - no AI messages")
    
    breaker_count = state.get("circuit_breaker", {}).get("count", 0)
    breaker_count += 1
    if breaker_count > 10:
        return {
            "messages": [AIMessage(content="⚠️ Stopping due to excessive reasoning loops.")],
            "circuit_breaker": {"count": breaker_count},
        }

    selected_tools = [tool_registry[id] for id in state["selected_tool_ids"]]
    llm_with_tools = llm.bind_tools([retrieve_tools, *selected_tools])

    messages = state["messages"].copy()
    system_message_content = f"""IMPORTANT: You are a multi-server assistant.
Active platform: {active_server}

Rules:
- Use real tools only
- Ask for clarification if platform is ambiguous
- Never invent parameter values
- Do not fall back to text instructions"""

    system_message = SystemMessage(content=system_message_content)

    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages.insert(0, system_message)

    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response], "circuit_breaker": {"count": breaker_count}}


def format_output(state: State, config: RunnableConfig, *, store: BaseStore) -> State:
    """Format final output node."""
    msgs = state["messages"]
    for m in reversed(msgs):
        if isinstance(m, AIMessage):
            cleaned = (m.content or "").strip()
            return {"messages": [AIMessage(content=cleaned or "No response generated.")]}
    for m in reversed(msgs):
        if isinstance(m, ToolMessage):
            return {"messages": [AIMessage(content=str(m.content))]}
    return {"messages": [AIMessage(content="No response generated.")]}


async def aformat_output(state: State, config: RunnableConfig, *, store: BaseStore) -> State:
    """Async version of format_output."""
    return format_output(state, config, store=store)


def create_agent(
    llm: LanguageModelLike,
    tool_registry: Dict[str, BaseTool | Callable],
    *,
    limit: int = 2,
    filter: Dict[str, Any] | None = None,
    namespace_prefix: tuple[str, ...] = ("tools",),
    retrieve_tools_function: Callable | None = None,
    retrieve_tools_coroutine: Callable | None = None,
) -> StateGraph:
    """Create an agent with a registry of tools."""
    if retrieve_tools_function is None and retrieve_tools_coroutine is None:
        retrieve_tools_function, retrieve_tools_coroutine = get_default_retrieval_tool(
            namespace_prefix, limit=limit, filter=filter
        )

    retrieve_tools = StructuredTool.from_function(
        func=retrieve_tools_function,
        coroutine=retrieve_tools_coroutine,
        name="retrieve_relevant_tools",
        description="Internal tool router: finds most relevant tools based on user query.",
    )

    store_arg = get_store_arg(retrieve_tools)
    tool_node = ToolNode(list(tool_registry.values()))

    def call_model_wrapper(state: State, config: RunnableConfig, *, store: BaseStore):
        return call_model(
            state, config, store=store, llm=llm,
            retrieve_tools=retrieve_tools, tool_registry=tool_registry
        )

    async def acall_model_wrapper(state: State, config: RunnableConfig, *, store: BaseStore):
        return await acall_model(
            state, config, store=store, llm=llm,
            retrieve_tools=retrieve_tools, tool_registry=tool_registry
        )

    def select_tools(tool_calls: list[dict], config: RunnableConfig, *, store: BaseStore) -> State:
        selected_tools = {}
        for tool_call in tool_calls:
            kwargs = {**tool_call["args"]}
            if store_arg:
                kwargs[store_arg] = store
            result = retrieve_tools.invoke(kwargs)
            selected_tools[tool_call["id"]] = result

        tool_messages, tool_ids = _format_selected_tools(selected_tools, tool_registry)
        return {"messages": tool_messages, "selected_tool_ids": tool_ids}

    async def aselect_tools(tool_calls: list[dict], config: RunnableConfig, *, store: BaseStore) -> State:
        selected_tools = {}
        for tool_call in tool_calls:
            kwargs = {**tool_call["args"]}
            if store_arg:
                kwargs[store_arg] = store
            result = await retrieve_tools.ainvoke(kwargs)
            selected_tools[tool_call["id"]] = result

        tool_messages, tool_ids = _format_selected_tools(selected_tools, tool_registry)
        return {"messages": tool_messages, "selected_tool_ids": tool_ids}

    def should_continue(state: State, *, store: BaseStore):
        messages = state["messages"]
        last_message = messages[-1] if messages else None

        # Safety: stop if repeated loops detected
        loop_count = state.get("circuit_breaker", {}).get("count", 0)
        if loop_count > 10:
            print("⚠️  [CIRCUIT BREAKER] Stopping - too many loops")
            return "output"

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return "output"

        pending_tool_calls = state.get("pending_tool_calls", [])
        if pending_tool_calls:
            return "parameter_validation"

        destinations = []
        for call in last_message.tool_calls:
            if call["name"] == "retrieve_relevant_tools":
                destinations.append(Send("select_tools", [call]))
            else:
                tool_call = tool_node.inject_tool_args(call, state, store)
                destinations.append(Send("tools", [tool_call]))
        return destinations

    # Build nodes
    select_tools_node = RunnableCallable(select_tools, aselect_tools)
    validation_node = RunnableCallable(validate_tool_selection, avalidate_tool_selection)

    def parameter_validation_wrapper(state: State, config: RunnableConfig, *, store: BaseStore):
        return validate_parameters(state, config, store=store, tool_registry=tool_registry)

    async def aparameter_validation_wrapper(state: State, config: RunnableConfig, *, store: BaseStore):
        return await avalidate_parameters(state, config, store=store, tool_registry=tool_registry)

    parameter_validation_node = RunnableCallable(parameter_validation_wrapper, aparameter_validation_wrapper)

    def user_input_wrapper(state: State, config: RunnableConfig, *, store: BaseStore):
        return handle_user_input(state, config, store=store, tool_registry=tool_registry)

    async def auser_input_wrapper(state: State, config: RunnableConfig, *, store: BaseStore):
        return await ahandle_user_input(state, config, store=store, tool_registry=tool_registry)

    user_input_node = RunnableCallable(user_input_wrapper, auser_input_wrapper)
    output_node = RunnableCallable(format_output, aformat_output)

    # Build graph
    builder = StateGraph(State)

    builder.add_node("validation", validation_node)
    builder.add_node("agent", RunnableCallable(call_model_wrapper, acall_model_wrapper))
    builder.add_node("select_tools", select_tools_node)
    builder.add_node("tools", tool_node)
    builder.add_node("parameter_validation", parameter_validation_node)
    builder.add_node("user_input", user_input_node)
    builder.add_node("output", output_node)

    builder.set_entry_point("validation")
    builder.add_edge("validation", "agent")

    builder.add_conditional_edges(
        "agent",
        should_continue,
        path_map={
            "select_tools": "select_tools",
            "tools": "tools",
            "parameter_validation": "parameter_validation",
            "output": "output",
        },
    )

    builder.add_edge("select_tools", "validation")
    builder.add_edge("tools", "validation")
    builder.add_edge("parameter_validation", "user_input")
    builder.add_edge("user_input", "validation")
    builder.add_edge("output", END)

    return builder