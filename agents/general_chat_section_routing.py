"""
general_chat_section_routing.py

Updated routing logic with APPLY functionality that:
1. Detects when user wants to apply changes
2. Calls section analysis function to update alignment scores and recommendations
3. Updates state with new data
4. Loops back to section with fresh analysis
5. FIXED: Better section name matching for typos and variations
"""

import os
import json
import re
from typing import Any, Dict, Optional, List, Tuple
from difflib import SequenceMatcher

# litellm (sync/async wrappers) - use completion or acompletion based on your runtime preference
from litellm import completion, acompletion

#local file imports
from agents.resume_builder_state import ResumeBuilderState

# -----------------------
# Constants
# -----------------------
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini/gemini-2.5-flash-lite")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
OFFLINE_MODE = GOOGLE_API_KEY is None
MAX_ROUTING_ATTEMPTS = 3  # Prevent infinite loops

def safe_extract_text(resp: Any) -> Optional[str]:
    """Extract the assistant text from common litellm ModelResponse shapes."""
    try:
        # Handle ModelResponse objects directly
        if hasattr(resp, "choices") and resp.choices:
            c0 = resp.choices[0]
            if hasattr(c0, "message") and c0.message:
                return getattr(c0.message, "content", None)
        # Handle streaming chunk-like objects or other common structures
        if hasattr(resp, "choices") and resp.choices:
            c0 = resp.choices[0]
            # Check for delta content first (common in streaming)
            if hasattr(c0, "delta") and c0.delta:
                return getattr(c0.delta, "content", None)
            # Fallback to message content if delta is not present
            if hasattr(c0, "message") and c0.message:
                return getattr(c0.message, "content", None)
        if hasattr(resp, "text") and resp.text:
            return resp.text
        # Handle dictionary responses
        if isinstance(resp, dict):
            if "choices" in resp and resp["choices"]:
                ch = resp["choices"][0]
                if isinstance(ch, dict):
                    if "message" in ch and isinstance(ch["message"], dict):
                        return ch["message"].get("content")
            if "candidates" in resp and resp["candidates"]:
                return resp["candidates"][0].get("content")
            if "content" in resp: # Direct content in dict
                return resp.get("content")
    except Exception as e:
        # Log the exception if any error occurs during extraction
        print(f"Warning: Error extracting text from response: {e}")
        pass
    # Return None if content cannot be extracted
    return None

def maybe_print_usage(resp: Any, label: str = ""):
    """Best-effort token usage printer for common shapes returned by litellm."""
    try:
        # 1) ModelResponse.usage
        if hasattr(resp, "usage") and resp.usage:
            u = resp.usage
            prompt = getattr(u, "prompt_tokens", getattr(u, "prompt_token_count", None))
            completion_t = getattr(u, "completion_tokens", getattr(u, "candidates_token_count", None))
            total = getattr(u, "total_tokens", getattr(u, "total_token_count", None))
            print(f"[Token Usage {label}] prompt={prompt} completion={completion_t} total={total}")
            return
        # 2) dict usage
        if isinstance(resp, dict) and "usage" in resp:
            u = resp["usage"]
            print(f"[Token Usage {label}] prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')} total={u.get('total_tokens')}")
    except Exception as e:
        # Log any errors during usage printing
        print(f"Warning: Error printing token usage: {e}")
        pass

def normalize_section_name(section_name: str, available_sections: List[str]) -> Optional[str]:
    """
    Normalize and match section names with fuzzy matching for typos.
    Returns the correct section name or None if no good match found.
    """
    if not section_name or not available_sections:
        return None
    
    # Clean the input
    clean_input = section_name.lower().strip()
    
    # Exact match first
    if clean_input in available_sections:
        return clean_input
    
    # Common variations and aliases
    section_aliases = {
        'skill': 'skills',
        'experience': 'experiences', 
        'exp': 'experiences',
        'work': 'experiences',
        'edu': 'education',
        'school': 'education',
        'project': 'projects',
        'cert': 'certificates',
        'certification': 'certificates',
        'certs': 'certificates',
        'pub': 'publications',
        'publication': 'publications',
        'papers': 'publications',
        'lang': 'languages',
        'language': 'languages',
        'rec': 'recommendations',
        'recommendation': 'recommendations',
        'refs': 'recommendations',
        'references': 'recommendations',
        'contact': 'contact',
        'contacts': 'contact',
        'summary': 'summary',
        'about': 'summary',
        'custom': 'custom',
        'other': 'custom',
        'additional': 'custom'
    }
    
    # Check aliases
    if clean_input in section_aliases:
        canonical = section_aliases[clean_input]
        if canonical in available_sections:
            return canonical
    
    # Fuzzy matching for typos (similarity > 0.6)
    best_match = None
    best_similarity = 0.0
    
    for section in available_sections:
        similarity = SequenceMatcher(None, clean_input, section).ratio()
        if similarity > best_similarity and similarity > 0.6:  # 60% similarity threshold
            best_similarity = similarity
            best_match = section
    
    return best_match

def extract_and_validate_json(raw_text: str) -> Dict[str, Any]:
    """
    Extract JSON from raw text and validate basic structure.
    Raises ValueError if no valid JSON found or required keys missing.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty response from LLM")
    
    # Try to find JSON block in response
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_text, re.DOTALL)
    if not json_match:
        # If no JSON found, treat entire text as answer
        return {"action": "answer", "route": None, "answer": raw_text.strip()}
    
    json_text = json_match.group(0)
    try:
        parsed = json.loads(json_text)
        
        # Validate required keys
        if not isinstance(parsed, dict):
            raise ValueError("Response is not a JSON object")
        
        if "action" not in parsed:
            raise ValueError("Missing required 'action' key in JSON response")
        
        # Ensure valid action values - ADD 'apply' to valid actions
        valid_actions = {"answer", "route", "stay", "switch", "exit", "apply"}
        if parsed["action"] not in valid_actions:
            print(f"Warning: Unexpected action '{parsed['action']}', treating as 'answer'")
            parsed["action"] = "answer"
        
        return parsed
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

def detect_question_matches(user_answer: str, questions: List[str]) -> List[Tuple[int, str, float]]:
    """
    Detect which questions the user's answer might be responding to using keyword matching.
    Returns list of (question_index, question_text, confidence_score) sorted by confidence.
    """
    matches = []
    user_words = set(user_answer.lower().split())
    
    for idx, question in enumerate(questions):
        question_words = set(question.lower().split())
        
        # Calculate keyword overlap
        common_words = user_words & question_words
        if len(common_words) > 0:
            # Simple confidence based on word overlap ratio
            confidence = len(common_words) / max(len(question_words), 1)
            matches.append((idx, question, confidence))
    
    # Sort by confidence (highest first)
    return sorted(matches, key=lambda x: x[2], reverse=True)

def safe_initialize_answers(state: ResumeBuilderState, section_name: str, questions: List[str]) -> None:
    """Safely initialize answers array for a section with proper synchronization."""
    if section_name not in state.recommended_answers:
        state.recommended_answers[section_name] = [""] * len(questions)
    else:
        # Ensure array length matches questions count
        current_answers = state.recommended_answers[section_name]
        if len(current_answers) != len(questions):
            # Resize array, preserving existing answers
            new_answers = [""] * len(questions)
            for i in range(min(len(current_answers), len(questions))):
                new_answers[i] = current_answers[i]
            state.recommended_answers[section_name] = new_answers

async def apply_section_changes(state: ResumeBuilderState, section_name: str, updated_content: str) -> Dict[str, Any]:
    """
    Apply changes to a section and re-analyze it against the job description.
    Updates alignment score, missing requirements, and recommended questions.
    
    Args:
        state: Current resume builder state
        section_name: Name of the section to update
        updated_content: New content for the section
    
    Returns:
        Dictionary with new analysis results
    """
    print(f"\n🔄 APPLYING changes to {section_name}...")
    
    # Update the resume section content
    if not hasattr(state, 'resume_sections'):
        state.resume_sections = {}
    state.resume_sections[section_name] = updated_content
    
    # Prepare analysis payload
    analysis_payload = {
        "jd_summary": state.jd_summary or "",
        "section_name": section_name,
        "section_content": updated_content,
        "original_section_data": state.section_objects.get(section_name, {})
    }
    
    # System prompt for re-analysis
    analysis_prompt = f"""You are analyzing how well an updated resume section aligns with job requirements.

    JOB DESCRIPTION SUMMARY: {analysis_payload['jd_summary']}
    SECTION NAME: {section_name}
    UPDATED SECTION CONTENT: {updated_content}

    Analyze the updated content and provide:

    RESPONSE FORMAT (JSON):
    {{
        "alignment_score": <number 0-100>,
        "missing_requirements": ["req1", "req2", ...],
        "recommended_questions": ["question1", "question2", ...],
        "analysis_summary": "Brief summary of improvements and remaining gaps"
    }}

    ANALYSIS RULES:
    1. Compare updated content against job requirements
    2. Calculate alignment score based on how well it matches JD needs
    3. Identify remaining missing requirements
    4. Generate 2-4 targeted questions for remaining gaps
    5. Keep questions specific and actionable
    6. Focus on the most impactful missing elements
    """
    
    try:
        # Call LLM for section re-analysis
        messages = [
            {"role": "system", "content": analysis_prompt},
            {"role": "user", "content": json.dumps(analysis_payload, separators=(",", ":"))}
        ]
        
        extra = {}
        if GOOGLE_API_KEY:
            extra["api_key"] = GOOGLE_API_KEY

        if OFFLINE_MODE:
            # Offline fallback
            return {
                "alignment_score": 75,
                "missing_requirements": ["More specific examples needed"],
                "recommended_questions": [f"Can you add more specific examples to your {section_name}?"],
                "analysis_summary": "Offline mode - please configure API key for full analysis"
            }
        
        resp = await acompletion(model=LLM_MODEL, messages=messages, max_completion_tokens=500, **extra)
        maybe_print_usage(resp, f"apply_{section_name}")
        raw = safe_extract_text(resp) or ""
        
        # Extract and validate analysis results - use direct JSON parsing for analysis
        analysis_result = {}
        try:
            # Try to find JSON block in response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                analysis_result = json.loads(json_text)
            else:
                print(f"No JSON found in analysis response: {raw}")
        except json.JSONDecodeError as e:
            print(f"JSON decode error in analysis: {e}")
            print(f"Raw response: {raw}")
        
        # Ensure required fields exist
        analysis_result.setdefault("alignment_score", 70)
        analysis_result.setdefault("missing_requirements", [])
        analysis_result.setdefault("recommended_questions", [])
        analysis_result.setdefault("analysis_summary", "Section updated successfully")
        
        # Update state with new analysis
        if section_name not in state.section_objects:
            state.section_objects[section_name] = {}
        
        state.section_objects[section_name].update({
            "alignment_score": analysis_result["alignment_score"],
            "missing_requirements": analysis_result["missing_requirements"],
            "recommended_questions": analysis_result["recommended_questions"],
            "last_updated": "just_now"  # Timestamp or flag to show it's fresh
        })
        
        # Reset answers for new questions
        if analysis_result["recommended_questions"]:
            state.recommended_answers[section_name] = [""] * len(analysis_result["recommended_questions"])
        else:
            # No more questions - section is complete
            state.recommended_answers[section_name] = []
        
        print(f"✅ {section_name} updated - New alignment score: {analysis_result['alignment_score']}%")
        print(f"📋 New questions: {len(analysis_result['recommended_questions'])}")
        
        return analysis_result
        
    except Exception as e:
        print(f"❌ Error in apply_section_changes: {e}")
        # Fallback response
        return {
            "alignment_score": 70,
            "missing_requirements": ["Analysis error - please try again"],
            "recommended_questions": [f"Let's continue improving your {section_name}. What specific details can you add?"],
            "analysis_summary": f"There was an error analyzing the updated {section_name}. The changes have been saved."
        }

async def call_llm_json_decision(system_prompt: str, user_payload: Dict[str, Any], max_tokens: int = 300) -> Dict[str, Any]:
    """
    Call the LLM (async) and expect JSON decision output with robust error handling.
    """
    # Build message: use a compact JSON payload to keep tokens small
    user_text = json.dumps(user_payload, separators=(",", ":"), ensure_ascii=False)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text}
    ]
    extra = {}
    if GOOGLE_API_KEY:
        extra["api_key"] = GOOGLE_API_KEY

    # Use async completion (acompletion) so this node can be async-compatible in the graph
    if OFFLINE_MODE:
        # Provide deterministic offline answer so app still works
        return {"action": "answer", "route": None, "answer": "(Offline) I received your query and will help once an API key is configured."}
    
    try:
        resp = await acompletion(model=LLM_MODEL, messages=messages, max_completion_tokens=max_tokens, **extra)
        maybe_print_usage(resp, "router")
        raw = safe_extract_text(resp) or ""
        
        # Use robust JSON extraction and validation
        return extract_and_validate_json(raw)
        
    except Exception as e:
        # Connectivity or auth issue -> degrade gracefully
        print(f"LLM call error: {e}")
        return {"action": "answer", "route": None, "answer": f"I encountered an error processing your request. Please try again."}

# -----------------------
# Node: general_chat_and_section_routing
# -----------------------
async def general_chat_and_section_routing(state: ResumeBuilderState, config: Dict[str, Any]) -> Any:
    """
    Master routing node with APPLY functionality, infinite loop prevention, and fuzzy section matching.
    """
    # Initialize routing attempt counter if not exists
    if not hasattr(state, 'routing_attempts'):
        state.routing_attempts = 0
    
    # Prevent infinite loops
    if state.routing_attempts >= MAX_ROUTING_ATTEMPTS:
        print(f"Maximum routing attempts ({MAX_ROUTING_ATTEMPTS}) reached, forcing exit to general chat")
        state.current_section = None
        state.routing_attempts = 0
        state.context.append(state.make_message(
            "assistant",
            "I'm having trouble processing your request. Let's start fresh - how can I help you with your resume?"
        ))
        return state
    
    state.routing_attempts += 1
    
    # Detect if this is the first turn (no user message yet)
    last_msg = state.context[-1] if state.context else None
    first_turn = not state.context or (last_msg and getattr(last_msg, "role", "") != "user")
    user_text = ""
    
    if not first_turn:
        user_text = getattr(last_msg, "content", "") or ""
    else:
        # Seed a synthetic user query to drive initial LLM greeting
        user_text = "INITIAL_GREETING: Greet the user, summarize JD, list sections with alignment and missing requirements, invite next action."

    # Build compact payload for LLM
    compact_sections = {}
    for s, data in state.section_objects.items():
        compact_sections[s] = {
            "section_name": s,
            "alignment_score": data.get("alignment_score"),
            "missing_requirements": data.get("missing_requirements", []),
            "recommended_questions": data.get("recommended_questions", []),
        }

    # Prepare recommended questions and answers for the current section
    recommended_questions = []
    current_answers = []
    if state.current_section and state.current_section in compact_sections:
        section_data = compact_sections[state.current_section]
        recommended_questions = section_data.get("recommended_questions", [])
        
        # Safe initialization of answers
        safe_initialize_answers(state, state.current_section, recommended_questions)
        current_answers = state.recommended_answers[state.current_section]
    
    # Get last 5 AI and human messages for context
    ai_messages = []
    human_messages = []
    
    # Go through context in reverse to get the most recent messages
    for msg in reversed(state.context):
        if msg.role == 'assistant' and len(ai_messages) < 5:
            ai_messages.insert(0, msg.content)
        elif msg.role == 'user' and len(human_messages) < 5:
            human_messages.insert(0, msg.content)
        
        # Stop if we've collected enough messages
        if len(ai_messages) >= 5 and len(human_messages) >= 5:
            break
    
    # Prepare section-specific data
    if state.current_section:
        # If in a specific section, only send that section's data
        sections_data = {state.current_section: compact_sections.get(state.current_section, {})}
        resume_content = {state.current_section: state.resume_sections.get(state.current_section)} \
            if hasattr(state, 'resume_sections') and state.resume_sections else {}
    else:
        # If in general chat, send all sections
        sections_data = compact_sections
        resume_content = state.resume_sections if hasattr(state, 'resume_sections') else {}

    # Get current section's resume content if available
    current_section_content = ""
    if state.current_section and hasattr(state, 'resume_sections') and state.current_section in state.resume_sections:
        current_section_content = state.resume_sections[state.current_section]
        if isinstance(current_section_content, list):
            current_section_content = "\n".join(str(item) for item in current_section_content)
        elif isinstance(current_section_content, dict):
            current_section_content = "\n".join(f"{k}: {v}" for k, v in current_section_content.items())

    # Prepare payload for LLM
    payload = {
        "user_query": user_text,
        "conversation_context": {
            "ai_messages": ai_messages,
            "human_messages": human_messages
        }
    }
    
    # Check if all questions are answered for the current section
    all_questions_answered = (
        state.current_section and 
        state.current_section in state.recommended_answers and 
        all(state.recommended_answers[state.current_section]) and
        len(state.recommended_answers[state.current_section]) == len(recommended_questions)
    )

    # Check if we're already in a section - use different prompts for different contexts
    if state.current_section:
        current_section_data = compact_sections.get(state.current_section, {})
        
        # User is in a section - detect intent to switch, exit, apply, or stay
        system_prompt = f"""You are analyzing user intent while they are currently in the '{state.current_section}' section of resume editing.

        CURRENT SECTION DATA: {current_section_data}
        CURRENT RESUME CONTENT FOR THIS SECTION: {current_section_content}
        
        CONVERSATION CONTEXT:
        Use the last 5 AI and human messages provided for context, and to maintain conversation flow.

        RESPONSE FORMAT:
        Respond with STRICT JSON: {{"action": "stay|switch|exit|apply", "route": "section_name_or_null", "answer": "response_text", "updated_section_content": "new_content_or_null"}}

        INTENT DETECTION RULES:
        1. action='stay': User wants to continue working on current section or asking questions about it
        2. action='switch': User clearly wants to move to a different specific section  
        3. action='exit': User wants to go back to general chat/main menu
        4. action='apply': User wants to APPLY/SAVE the changes to their resume (look for words like "apply", "save", "yes", "confirm", "update resume")
        5. If action='switch', set 'route' to the target section name (must be from available sections)
        6. If action='apply', include the final updated section content in 'updated_section_content'
        7. If action='stay', provide helpful response in 'answer' field
        8. If action='exit', set 'answer' to confirmation message
        9. Be liberal with 'stay' - only switch/exit/apply if user clearly indicates they want to
        10. Look for clear intent words like 'go to', 'switch to', 'work on [different section]', 'back to main', 'exit', 'apply', 'save', 'confirm'

        STRICT REQUIREMENT: 
        At the END of EVERY response, you MUST output the remaining unanswered questions in the exact format 
        in a clear NUMBERED LIST format, exactly like this:

        Unanswered Questions:
        1. [first unanswered question here]
        2. [second unanswered question here]
        3. [third unanswered question here]

        - If there are no unanswered questions, instead write:
        "All questions have been answered."

        QUESTION-ANSWER MATCHING:
        Current questions: {recommended_questions}
        Current answers: {current_answers}

        When user provides information that could answer any recommended questions:
        - As soon as u detect a question is being answered , show and update the 'question_matches' and 'updated_answers'.
        1. Include 'question_matches' array in response with matched question indices
        2. Include 'updated_answers' array with new answers in correct positions
        3. Acknowledge the provided information in your answer

        APPLY DETECTION:
        - When user says "yes", "apply" or something related. 
        - Set action='apply' and include the complete updated section content in 'updated_section_content'
        - The updated content should incorporate all the answered questions

        CHAT TIMELINE: 
        * initial message: is always a welcome message  , alignment score , then STRICT REQUIREMENT 
        * Keep answers under 120 words, no markdown
        * Always show numbered list of unanswered questions at end of each message if any remain ,(take questions from recommended_questions not previous chats)
        * If all questions are answered, SHOW THE COMPLETE UPDATED SECTION CONTENT and ask for confirmation to APPLY
        * Maintain conversational context from previous messages
        - If user is chatting and their messages are not related to any recommended question, answer them, but always include the STRICT REQUIREMENT
        - After all questions are answered:
            1. Acknowledge that all questions have been answered
            2. Generate an updated version of the section content based on the answers, keep the same format as the previous resume_section
            3. Display the updated_section_content with line breaks before and after
            4. Ask if they want to APPLY these changes to the section
            
            - If user confirms (says 'yes', 'apply', etc.):
                * Set action='apply' with the updated content
                * This will trigger re-analysis and return to section with fresh data
                
            - If user declines (says 'no', 'not now', etc.):
                * Keep the changes in the updated_section_content
                * Continue the conversation
                * Gently remind them they can apply the changes later
                
        - If user asks to modify any answers after seeing the preview:
            * Update the specific answer
            * Regenerate the updated section content
            * Show the new preview and ask for confirmation again
        """

    else:
        available_sections = list(compact_sections.keys())
        
        # User is in general chat - route to section or answer
        system_prompt = f"""You are a resume analysis assistant helping users improve their resume sections.

        AVAILABLE SECTIONS: {', '.join(available_sections)}
        CURRENT RESUME SECTIONS DATA: {json.dumps(compact_sections, separators=(",", ":"))}

        RESPONSE FORMAT: 
        Respond with STRICT JSON: {{"action": "answer|route", "route": "section_name_or_null", "answer": "response_text"}}

        RULES:
        1. If user_query starts with 'INITIAL_GREETING', ALWAYS set action='answer' with friendly greeting and key gaps summary
        2. Use action='route' ONLY when user explicitly asks to work on a section or clearly wants to edit one
        3. When routing, use EXACT section name from available sections
        4. If user intent is unclear or just asking questions, use action='answer'
        5. Look for clear intent to work on/edit/improve specific section before routing
        6. Keep answers helpful and under 120 words, no markdown
        7. IMPORTANT: If user mentions a section name that doesn't exactly match, try to find the closest match from available sections
        """
            
    # Call LLM for decision
    try:
        parsed = await call_llm_json_decision(system_prompt, payload, max_tokens=800)
        
        # Debug: Print the parsed response
        print("\nLLM Response:", json.dumps(parsed, indent=2))
        if( state.recommended_answers):
            print("\n" + "="*80)
            print( "\n\n\n ANSWERS " ,  state.recommended_answers , "\n\n\n")
            print("\n" + "="*80)
        
        # Handle APPLY action first - this is the key new functionality
        if parsed.get("action") == "apply" and state.current_section:
            updated_content = parsed.get("updated_section_content", "")
            if updated_content:
                # Apply the changes and re-analyze the section
                analysis_result = await apply_section_changes(state, state.current_section, updated_content)
                
                # Create confirmation message
                confirmation_msg = (
                    f"Applied changes to {state.current_section}!\n\n"
                    f"New alignment score: {analysis_result['alignment_score']}%\n"
                    f"{analysis_result.get('analysis_summary', '')}\n\n"
                )
                
                if analysis_result.get("recommended_questions"):
                    confirmation_msg += f"Let's continue improving this section with {len(analysis_result['recommended_questions'])} more targeted questions."
                else:
                    confirmation_msg += "This section looks complete! You can switch to another section or continue refining."
                
                state.context.append(state.make_message("assistant", confirmation_msg))
                
                # Stay in the same section but with fresh data
                print(f"Staying in {state.current_section} with updated analysis")
                return state
            else:
                # No updated content provided
                state.context.append(state.make_message(
                    "assistant", 
                    "I couldn't find the updated content to apply. Could you please confirm the changes you'd like to make?"
                ))
                return state
        
        # Handle question-answer matching for section context
        if state.current_section and ('question_matches' in parsed or 'updated_answers' in parsed):
            # Initialize recommended_answers for current section if not exists
            if state.current_section not in state.recommended_answers:
                state.recommended_answers[state.current_section] = [''] * len(recommended_questions)
            
            # Process explicit question matches
            if 'question_matches' in parsed and 'updated_answers' in parsed:
                question_matches = parsed.get('question_matches', [])
                updated_answers = parsed.get('updated_answers', [])
                
                if isinstance(question_matches, list) and isinstance(updated_answers, list):
                    for match_idx in question_matches:
                        if (isinstance(match_idx, int) and 
                            0 <= match_idx < len(state.recommended_answers[state.current_section]) and
                            match_idx < len(updated_answers) and
                            updated_answers[match_idx]):
                            
                            state.recommended_answers[state.current_section][match_idx] = updated_answers[match_idx]
                            print(f"Updated answer {match_idx} to: {updated_answers[match_idx]}")
            
            # Handle direct answer updates (without explicit question matches)
            elif 'updated_answers' in parsed and isinstance(parsed['updated_answers'], list) and not first_turn:
                updated_answers = parsed['updated_answers']
                # Find the first empty slot or the first question without a good answer
                for i, answer in enumerate(state.recommended_answers[state.current_section]):
                    if i < len(updated_answers) and updated_answers[i] and (not answer or len(str(answer).strip()) < 5):
                        state.recommended_answers[state.current_section][i] = updated_answers[i]
                        print(f"Updated answer {i} to: {updated_answers[i]}")
                        break
                else:
                    # If no empty slots, try to match by question content
                    matches = detect_question_matches(user_text, recommended_questions)
                    if matches and updated_answers:
                        best_match_idx = matches[0][0]
                        new_answer = next((ans for ans in updated_answers if ans), None)
                        if new_answer and best_match_idx < len(state.recommended_answers[state.current_section]):
                            state.recommended_answers[state.current_section][best_match_idx] = new_answer
                            print(f"Auto-matched answer to question {best_match_idx}: {new_answer}")
            
            # Debug output after updates
            print("\n" + "="*80)
            print("CURRENT ANSWERS:")
            for i, (q, a) in enumerate(zip(recommended_questions, state.recommended_answers[state.current_section])):
                print(f"{i+1}. Q: {q}\n   A: {a}\n")
            print("="*80 + "\n")
        
    except ValueError as e:
        print(f"JSON parsing error: {e}")
        fallback_text = "I'm having trouble understanding your request. Could you please rephrase?"
        state.context.append(state.make_message("assistant", fallback_text))
        return state
    except Exception as e:
        print(f"Unexpected error in LLM call: {e}")
        fallback_text = "I encountered an error. How can I help you with your resume?"
        state.context.append(state.make_message("assistant", fallback_text))
        return state

    # Reset routing attempts on successful processing
    state.routing_attempts = 0

    # Process LLM decision
    action = parsed.get("action")
    route_to = parsed.get("route")
    answer_text = parsed.get("answer", "")

    if state.current_section:
        # Handle section context actions
        if action == "switch" and route_to:
            # Get available sections for fuzzy matching
            available_sections = list(compact_sections.keys())
            
            # Normalize the target section name
            normalized_section = normalize_section_name(route_to, available_sections)
            
            if normalized_section:
                state.current_section = normalized_section
                print(f"Switching to section: {normalized_section}")
                if normalized_section != route_to:
                    print(f"Note: Corrected '{route_to}' to '{normalized_section}'")
                return state
            else:
                # No good match found - stay in current section and explain
                state.context.append(state.make_message(
                    "assistant",
                    f"I couldn't find a section that matches '{route_to}'. "
                    f"Available sections are: {', '.join(sorted(available_sections))}. "
                    "Please try again with one of these section names."
                ))
                return state
        elif action == "exit":
            # User wants to exit to general chat
            state.current_section = None
            exit_msg = answer_text or "Back to general chat. How can I help you with your resume?"
            state.context.append(state.make_message("assistant", exit_msg))
            return state
        else:
            # action == "stay" or fallback - remain in current section
            if answer_text:
                state.context.append(state.make_message("assistant", answer_text))
            print(f"Staying in section: {state.current_section}")
            return state
    else:
        # Handle general chat actions
        if action == "route" and route_to:
            # Get available sections for fuzzy matching
            available_sections = list(compact_sections.keys())
            
            # Normalize the target section name
            normalized_section = normalize_section_name(route_to, available_sections)
            
            if normalized_section:
                state.current_section = normalized_section
                print(f"Routing to section: {normalized_section}")
                if normalized_section != route_to:
                    print(f"Note: Corrected '{route_to}' to '{normalized_section}'")
                return state
            else:
                # No good match found - stay in general chat and explain
                state.context.append(state.make_message(
                    "assistant",
                    f"I couldn't find a section that matches '{route_to}'. "
                    f"Available sections are: {', '.join(sorted(available_sections))}. "
                    "Please try again with one of these section names."
                ))
                return state
        else:
            # action == "answer" or fallback
            if not answer_text:
                answer_text = "I'm here to help with your resume. Would you like to work on a specific section?"
            state.context.append(state.make_message("assistant", answer_text))
            return state