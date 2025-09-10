# graph_builder.py
import logging
from typing import Any, Dict, List, Optional, Callable
import json
import os
import re
from uuid import uuid4

from pyagenity.graph import (
    StateGraph,
    CompiledGraph,
    Node,
    ToolNode,
    Edge,
)
from pyagenity.state import AgentState
from pyagenity.utils import (
    Message,
    END,
    START,
    CallbackManager,
    DependencyContainer,
    ResponseGranularity,
)
from pyagenity.checkpointer import InMemoryCheckpointer
from pyagenity.publisher import ConsolePublisher # Example publisher

# Import your custom state and agent nodes
from agents.resume_builder_state import ResumeBuilderState
from agents.general_chat_section_routing import general_chat_and_section_routing

# Import new specialized section nodes
from agents.section_nodes import section_chat_node, section_updater_node, section_applier_node

# Import LLM helpers from the routing module
from agents.general_chat_section_routing import call_llm_json_decision, safe_extract_text, maybe_print_usage, OFFLINE_MODE, GOOGLE_API_KEY, LLM_MODEL
from litellm import acompletion

def build_resume_graph(
    checkpointer: InMemoryCheckpointer[ResumeBuilderState] | None = None,
    publisher: ConsolePublisher | None = None,
    dependency_container: DependencyContainer | None = None,
    callback_manager: CallbackManager | None = None,
    initial_state: ResumeBuilderState | None = None,
) -> CompiledGraph[ResumeBuilderState]:
    """Build & compile the resume assistant graph with simplified routing.

    Parameters:
        initial_state: If provided, this pre-populated state (with jd_summary, section_objects,
            resume_sections, etc.) is used instead of a fresh blank state.
    """
    print("Building the resume builder graph...")

    checkpointer = checkpointer or InMemoryCheckpointer[ResumeBuilderState]()
    publisher = publisher or ConsolePublisher()
    dependency_container = dependency_container or DependencyContainer()
    callback_manager = callback_manager or CallbackManager()

    graph = StateGraph[ResumeBuilderState](
        state=initial_state or ResumeBuilderState(),
        publisher=publisher,
        dependency_container=dependency_container,
    )

    # --- Add Nodes ---
    # General chat and initial section routing only
    graph.add_node("GeneralChat", general_chat_and_section_routing)

    # Add specialized section nodes that handle their own routing
    graph.add_node("SectionChat", section_chat_node)
    graph.add_node("SectionUpdater", section_updater_node) 
    graph.add_node("SectionApplier", section_applier_node)

    # --- Define Edges ---
    # Start with general chat
    graph.set_entry_point("GeneralChat")

    def route_from_general_chat(state):
        """Route from GeneralChat - only handles initial routing TO sections."""
        next_action = getattr(state, 'next_action', None)
        
        print(f"[GENERAL_CHAT_ROUTER] Next action: {next_action}")
        print(f"[GENERAL_CHAT_ROUTER] Current section: {state.current_section}")
        
        # Check routing attempt limits
        max_attempts = getattr(state, 'routing_attempts', 0)
        if max_attempts >= 3:
            print("[GENERAL_CHAT_ROUTER] Max routing attempts reached, ending conversation")
            return END
        
        valid_sections = {
            "skills", "experiences", "education", "projects",
            "summary", "contact", "certificates", "publications", 
            "languages", "recommendations", "custom"
        }
        
        # Only route TO sections from general chat
        if next_action == "section_chat" and state.current_section in valid_sections:
            return "SectionChat"
        
        print("[GENERAL_CHAT_ROUTER] Staying in general chat or ending")
        return END

    def route_from_section_chat(state):
        """Route from SectionChat - handles internal section routing."""
        next_action = getattr(state, 'next_action', None)
        print(f"[SECTIONCHAT_ROUTER] Next action: {next_action}")
        
        if next_action == "section_updater":
            return "SectionUpdater"
        elif next_action == "section_applier":
            return "SectionApplier"
        elif next_action == "exit_to_general":
            return "GeneralChat"
        elif next_action == "section_chat":
            # Stay in section chat (loop back to self)
            return "SectionChat"
        else:
            # Default: stay in section chat
            return "SectionChat"

    def route_from_section_updater(state):
        """Route from SectionUpdater."""
        next_action = getattr(state, 'next_action', None)
        print(f"[SECTIONUPDATER_ROUTER] Next action: {next_action}")
        
        if next_action == "section_applier":
            return "SectionApplier"
        elif next_action == "exit_to_general":
            return "GeneralChat"
        else:
            # Default: back to section chat
            return "SectionChat"

    def route_from_section_applier(state):
        """Route from SectionApplier."""
        next_action = getattr(state, 'next_action', None)
        print(f"[SECTIONAPPLIER_ROUTER] Next action: {next_action}")
        
        if next_action == "exit_to_general":
            return "GeneralChat"
        else:
            # Default: back to section chat with fresh data
            return "SectionChat"

    # Add conditional edges
    graph.add_conditional_edges(
        "GeneralChat",
        route_from_general_chat,
        {
            "SectionChat": "SectionChat",
            END: END
        }
    )
    
    graph.add_conditional_edges(
        "SectionChat", 
        route_from_section_chat,
        {
            "SectionUpdater": "SectionUpdater",
            "SectionApplier": "SectionApplier", 
            "GeneralChat": "GeneralChat",
            "SectionChat": "SectionChat",  # Self-loop for staying in section
            END: END  # End execution when staying in section without specific action
        }
    )
    
    graph.add_conditional_edges(
        "SectionUpdater",
        route_from_section_updater, 
        {
            "SectionChat": "SectionChat",
            "SectionApplier": "SectionApplier",
            "GeneralChat": "GeneralChat"
        }
    )

    graph.add_conditional_edges(
        "SectionApplier",
        route_from_section_applier,
        {
            "SectionChat": "SectionChat",
            "GeneralChat": "GeneralChat"
        }
    )

    # --- Compile the graph ---
    print("Compiling the graph...")
    compiled_graph = graph.compile(checkpointer=checkpointer)
    print("Graph compiled successfully.")

    return compiled_graph

# --- Interactive Terminal Chat Runner ---
async def run_interactive_session(compiled_graph: CompiledGraph[ResumeBuilderState]):
    """
    Runs an interactive chat session with the compiled PyAgenity graph.
    """
    print("\nWelcome to the Resume Builder Chat!")
    print("Type 'quit' or 'exit' to end the session.")
    print("You can ask questions about your JD or resume sections.")
    print("You can also route to sections like 'skills', 'experiences', etc.")
    print("To exit a section, say 'back to general chat' or 'exit section'.")

    # Helper: extract last assistant message from a list
    def _last_assistant(msgs: List[Message]) -> Message | None:
        for m in reversed(msgs):
            if getattr(m, "role", None) == "assistant":
                return m
        return None

    # Helper: print assistant message from invocation result
    def _print_from_result(result: Dict[str, Any]) -> None:
        msgs = result.get("messages", []) if isinstance(result, dict) else []
        assistant = _last_assistant(msgs)
        if not assistant and isinstance(result, dict):
            st = result.get("state")
            if st is not None:
                ctx = getattr(st, "context", None) or (st.get("context") if isinstance(st, dict) else None)
                if ctx:
                    assistant = _last_assistant(ctx)
        if assistant:
            content = assistant.content
            # Check if it's a JSON routing message and skip printing it
            try:
                parsed = json.loads(content)
                if "route" in parsed:
                    # Don't print routing JSON
                    return
            except:
                pass
            print(f"AI: {content}")
        else:
            print("AI: (No response generated)")

    # Initial setup
    initial_input: Dict[str, Any] = {"messages": [Message.from_text("SESSION_START", role="system")]}
    config = {"thread_id": str(uuid4()), "recursion_limit": 50}  # Reasonable recursion limit

    # First automatic invocation (assistant greets)
    try:
        first_result = await compiled_graph.ainvoke(initial_input, config, response_granularity="full")
        initial_input["messages"] = first_result.get("messages", [])
        if (new_state := first_result.get("state")):
            initial_input["state"] = new_state
        _print_from_result(first_result)
    except Exception as e:
        print(f"Startup error: {e}")

    # Interactive loop
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"quit", "exit"}:
            break

        initial_input["messages"].append(Message.from_text(user_input, role="user"))
        try:
            turn_result = await compiled_graph.ainvoke(initial_input, config, response_granularity="full")
            initial_input["messages"] = turn_result.get("messages", [])
            
            # Update state if returned
            if (new_state := turn_result.get("state")):
                initial_input["state"] = new_state
            
            # Trim messages to avoid unbounded growth (keep last 20 instead of 30)
            if len(initial_input["messages"]) > 20:
                initial_input["messages"] = initial_input["messages"][-20:]
            
            _print_from_result(turn_result)
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
            break

    print("\nChat session ended. Goodbye!")