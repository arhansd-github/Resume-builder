# src/graph_builder.py

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

# Import LLM helpers from the routing module
from agents.general_chat_section_routing import call_llm_json_decision, safe_extract_text, maybe_print_usage, OFFLINE_MODE, GOOGLE_API_KEY, LLM_MODEL
from litellm import acompletion

# Single section node that handles all sections based on section_name parameter
async def section_select_node(state: ResumeBuilderState, config: Dict[str, Any], section_name: str) -> ResumeBuilderState:
    """Single node that handles all sections based on the section_name parameter using LLM."""
    print(f"\n--- In {section_name.title()} Section ---")
    
    # Check if we're in the correct section
    if state.current_section != section_name:
        # This shouldn't happen with proper routing, but handle gracefully
        print(f"Warning: Expected section '{section_name}' but current_section is '{state.current_section}'")
        state.current_section = None
        state.context.append(state.make_message("assistant", "Returning to general chat. What would you like to do?"))
        return state
    
    # Get the last user message (if any)
    last_msg = state.context[-1] if state.context else None
    is_first_entry = not last_msg or getattr(last_msg, "role", "") != "user"
    user_text = ""
    
    if is_first_entry:
        # First time entering this section
        user_text = f"SECTION_ENTRY: User just entered {section_name} section. Provide initial guidance."
    else:
        # User has sent a message while in this section
        user_text = getattr(last_msg, "content", "") or ""
    
    # Get section-specific data
    section_data = state.section_objects.get(section_name, {})
    resume_content = state.resume_sections.get(section_name, "No content yet")
    
    # Prepare context for LLM with clear variable blocks
    jd_summary = state.jd_summary or "No JD provided"
    alignment_score = section_data.get("alignment_score", "Not analyzed")
    missing_requirements = section_data.get("missing_requirements", [])
    recommended_questions = section_data.get("recommended_questions", None)
    
    # System prompt with clear variable formatting
    system_prompt = f"""You are a resume assistant currently helping with the {section_name} section.

        CONTEXT:
        - JD Summary: {jd_summary}
        - Section Alignment Score: {alignment_score}
        - Missing Requirements: {missing_requirements}
        - Recommended Questions: {recommended_questions}
        - Current Resume Content: {resume_content}
        - User Message: {user_text}

        TASKS:
        1. If user_message starts with "SECTION_ENTRY", provide a welcoming message that:
        - Confirms they're in the {section_name} section
        - Shows alignment score: {alignment_score}
        - Lists missing requirements: {', '.join(missing_requirements) if missing_requirements else 'None identified'}
        - Suggests next steps
        - Keep it concise and actionable (under 100 words)

        2. For other messages, help improve this specific section by:
        - Answering questions about {section_name}
        - Suggesting improvements based on JD alignment
        - Helping rewrite or enhance content
        - Staying focused on {section_name} content

        3. If user asks to switch sections or go back, acknowledge but stay helpful.

        Respond in natural language, be helpful and specific. No JSON output needed."""

    # Prepare section context for LLM
    section_context = {
        "current_section": section_name,
        "jd_summary": jd_summary,
        "section_alignment": alignment_score,
        "missing_requirements": missing_requirements,
        "recommended_questions": recommended_questions,
        "current_resume_content": resume_content,
        "user_message": user_text
    }
    
    # Prepare messages for LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(section_context, indent=2)}
    ]
    
    # Call LLM with error handling
    if OFFLINE_MODE:
        missing_reqs_text = ', '.join(missing_requirements) if missing_requirements else 'none identified'
        response_text = f"(Offline) You're in the {section_name} section. Alignment score: {alignment_score}. Missing: {missing_reqs_text}."
    else:
        try:
            extra = {}
            if GOOGLE_API_KEY:
                extra["api_key"] = GOOGLE_API_KEY
                
            resp = await acompletion(
                model=LLM_MODEL,
                messages=messages,
                max_completion_tokens=300,
                **extra
            )
            maybe_print_usage(resp, f"section_{section_name}")
            response_text = safe_extract_text(resp) or f"Let's work on your {section_name} section."
            
        except Exception as e:
            print(f"LLM error in section node: {e}")
            missing_reqs_text = ', '.join(missing_requirements) if missing_requirements else 'not calculated'
            response_text = f"Let's work on your {section_name} section. Your alignment score is {alignment_score}. Missing requirements: {missing_reqs_text}."
    
    # Add assistant response to context
    state.context.append(state.make_message("assistant", response_text))
    
    # Keep current_section set so we stay in this section
    # The routing node will handle section switches
    
    return state


def build_resume_graph(
    checkpointer: InMemoryCheckpointer[ResumeBuilderState] | None = None,
    publisher: ConsolePublisher | None = None,
    dependency_container: DependencyContainer | None = None,
    callback_manager: CallbackManager | None = None,
    initial_state: ResumeBuilderState | None = None,
) -> CompiledGraph[ResumeBuilderState]:
    """Build & compile the resume assistant graph with routing + section nodes.

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
    # The main routing node
    graph.add_node("AnalyzeUserQuery", general_chat_and_section_routing)

    # Add section nodes using lambda functions to pass section_name parameter
    graph.add_node("SkillsSection", lambda state, config: section_select_node(state, config, "skills"))
    graph.add_node("ExperiencesSection", lambda state, config: section_select_node(state, config, "experiences"))
    graph.add_node("EducationSection", lambda state, config: section_select_node(state, config, "education"))
    graph.add_node("ProjectsSection", lambda state, config: section_select_node(state, config, "projects"))
    graph.add_node("SummarySection", lambda state, config: section_select_node(state, config, "summary"))
    graph.add_node("ContactSection", lambda state, config: section_select_node(state, config, "contact"))
    graph.add_node("CertificatesSection", lambda state, config: section_select_node(state, config, "certificates"))
    graph.add_node("PublicationsSection", lambda state, config: section_select_node(state, config, "publications"))
    graph.add_node("LanguagesSection", lambda state, config: section_select_node(state, config, "languages"))
    graph.add_node("RecommendationsSection", lambda state, config: section_select_node(state, config, "recommendations"))
    graph.add_node("CustomSection", lambda state, config: section_select_node(state, config, "custom"))

    # --- Define Edges ---
    # Initial flow: Start by analyzing the user query
    graph.set_entry_point("AnalyzeUserQuery")

    def route_to_section(state):
        """
        Route based on current_section with improved loop prevention.
        Returns END if routing attempts exceeded or no valid section.
        """
        valid_sections = {
            "skills", "experiences", "education", "projects",
            "summary", "contact", "certificates", "publications",
            "languages", "recommendations", "custom"
        }
        
        # Debug: Print current state
        print(f"[ROUTER] Current section: {state.current_section}")
        print(f"[ROUTER] Routing attempts: {getattr(state, 'routing_attempts', 0)}")
        
        # Check routing attempt limits
        max_attempts = getattr(state, 'routing_attempts', 0)
        if max_attempts >= 3:  # Matching MAX_ROUTING_ATTEMPTS from routing module
            print("[ROUTER] Max routing attempts reached, ending conversation")
            return END
        
        # Check if we should route to a section
        if state.current_section in valid_sections:
            # Check if we have a new message from the user or it's initial entry
            last_message = state.context[-1] if state.context else None
            is_user_message = last_message and getattr(last_message, "role", "") == "user"
            is_initial = not state.context or len(state.context) <= 1
            
            if is_user_message or is_initial:
                print(f"[ROUTER] Routing to section: {state.current_section}")
                return state.current_section
                
        print("[ROUTER] No valid section routing, ending")
        return END

    # Add conditional edges from AnalyzeUserQuery with section name mapping
    section_node_mapping = {
        "skills": "SkillsSection",
        "experiences": "ExperiencesSection", 
        "education": "EducationSection",
        "projects": "ProjectsSection",
        "summary": "SummarySection",
        "contact": "ContactSection",
        "certificates": "CertificatesSection",
        "publications": "PublicationsSection",
        "languages": "LanguagesSection",
        "recommendations": "RecommendationsSection",
        "custom": "CustomSection"
    }
    
    # Build the routing map for conditional edges
    routing_map = {section: node for section, node in section_node_mapping.items()}
    routing_map[END] = END
    
    graph.add_conditional_edges(
        "AnalyzeUserQuery",
        route_to_section,
        routing_map
    )

    # IMPORTANT: Loop sections back to AnalyzeUserQuery for continuous interaction
    # This allows users to continue conversations within sections
    for section_node in section_node_mapping.values():
        graph.add_edge(section_node, "AnalyzeUserQuery")

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