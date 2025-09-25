from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from uipath.models import CreateAction
from uipath_langchain.chat import UiPathChat
from langchain_core.messages import SystemMessage, HumanMessage
import json, logging

logging.basicConfig(level=logging.INFO)

# Use UiPathChat for making LLM calls
llm = UiPathChat(model="gpt-4o-2024-08-06")

# ---------------- State ----------------
class GraphState(BaseModel):
    leave_request: str | None = None
    leave_start: str | None = None
    leave_end: str | None = None
    leave_reason: str | None = None
    leave_category: str | None = None
    hitl_required: bool = False


# ---------------- Nodes ----------------
async def start_node(state: GraphState) -> GraphState:
    system_prompt = """You are a smart HR assistant tasked with extracting leave information from a user's message. 

    Your goal is to extract the following three fields:
    1. leave_start - the start date of the leave (try to parse informal dates like "next Monday", "Sep 25", etc. into YYYY-MM-DD if possible; otherwise return null)
    2. leave_end - the end date of the leave (same rules as leave_start)
    3. leave_reason - the reason for the leave

    Instructions:
    - Only return a JSON object with keys: leave_start, leave_end, leave_reason.
    - If a field cannot be determined, return null.
    - If dates are vague (like "next Monday"), try to infer the actual date relative to today, but if not possible, return null.
    - Only output the JSON. Do not include any explanations, commentary, or extra text.
    - Extract the most relevant reason from the user's message.

    Examples:

    User message: "I want to take leave from 2025-09-25 to 2025-09-27 for attending a wedding."
    Output:
    {
    "leave_start": "2025-09-25",
    "leave_end": "2025-09-27",
    "leave_reason": "attending a wedding"
    }

    User message: "I need a day off next Monday."
    Output:
    {
    "leave_start": "2025-09-30",
    "leave_end": "2025-09-30",
    "leave_reason": "a day off"
    }

    User message: "I want leave for a family function."
    Output:
    {
    "leave_start": null,
    "leave_end": null,
    "leave_reason": "family function"
    }
    """

    output = await llm.ainvoke(
        [SystemMessage(system_prompt),
         HumanMessage(state.leave_request)]
    )

    # Parse JSON
    leave_data = json.loads(output.content)

    return state.model_copy(update={"leave_start": leave_data.get("leave_start"),
                                    "leave_end": leave_data.get("leave_end"),
                                    "leave_reason": leave_data.get("leave_reason"),
                                    }
                                )


def check_fields_node(state: GraphState) -> GraphState:
    # If any required field is missing, mark HITL as required
    hitl_required = not state.leave_start or not state.leave_end or not state.leave_reason
    # Return updated state
    return state.model_copy(update={"hitl_required":hitl_required})


def hitl_node(state: GraphState) -> Command:
    # Escalate to human to provide missing details
    action_data = interrupt(
        CreateAction(
            app_name="LeaveRequestApp",  # Your Action App name
            title="Please fill missing leave details",
            data={
                "LeaveStart": state.leave_start or "",
                "LeaveEnd": state.leave_end or "",
                "LeaveReason": state.leave_reason
            },
            app_version=1,
            app_folder_path="Shared"
        )
    )

    # Update state with returned values from human
    updates = {
        "leave_start": action_data.get("LeaveStart", state.leave_start),
        "leave_end": action_data.get("LeaveEnd", state.leave_end),
        "leave_reason": action_data.get("LeaveReason", state.leave_reason)
    }

    return Command(update=updates)


async def categorize_node(state: GraphState) -> GraphState:
    system_prompt = """You are a helpful and professional HR assistant. Your task is to categorize leave requests submitted by employees. 
    Each request contains reason for leave. 

    You should categorize each leave request into one of the following categories:

    1. Sick Leave - when the employee is unable to work due to illness, injury, or medical appointments.
    2. Vacation - when the employee requests time off for personal leisure, travel, or rest.
    3. Maternity/Paternity Leave - when the leave is related to childbirth, adoption, or childcare.
    4. Other - for any leave that does not clearly fit into the above categories (e.g., bereavement, jury duty, personal emergencies).

    Instructions for categorization:
    - Read the reason carefully and determine the most appropriate category.
    - If a request could fit multiple categories, choose the one that best matches the employee's main reason for leave.
    - Be concise and return only the category name (Sick Leave, Vacation, Maternity/Paternity, or Other) as output.
    - Maintain a professional tone and avoid adding extra commentary.

    Example:
    Reason: "I have a medical checkup scheduled next week." → Output: Sick Leave
    Reason: "I want to go on a family trip." → Output: Vacation
    """

    output = await llm.ainvoke(
        [SystemMessage(system_prompt),
         HumanMessage(f"Reason: {state.leave_reason}")]
    )

    return state.model_copy(update={"leave_category": output.content})


def end_node(state: GraphState) -> GraphState:
    logging.info(f"Leave request categorized: {state.leave_category}")
    return state


# --- The CONDITION function ---
# Its only job is to READ the hitl_required state and return a routing key.
def should_go_to_hitl(state: GraphState):
    """Reads the 'hitl_required' flag to decide the next path."""

    if state.hitl_required:
        return "hitl_needed"  # This is a routing key
    else:
        return "hitl_not_needed"   # This is another routing key


# ---------------- Build Graph ----------------
graph = StateGraph(GraphState)
graph.add_node("start", start_node)
graph.add_node("check_fields", check_fields_node)
graph.add_node("hitl", hitl_node)
graph.add_node("categorize", categorize_node)
graph.add_node("end", end_node)

graph.set_entry_point("start")
graph.add_edge("start", "check_fields")
graph.add_conditional_edges(
    "check_fields",         # Start branching AFTER this node runs.
    should_go_to_hitl,      # Use this function to decide WHERE to go.
    path_map={
        "hitl_needed": "hitl",         # If function returns this string, go to 'hitl'
        "hitl_not_needed": "categorize"   # If function returns this string, go to 'categorize'
    }
)
graph.add_edge("hitl", "check_fields")
graph.add_edge("categorize", "end")
graph.add_edge("end", END)

agent = graph.compile()
