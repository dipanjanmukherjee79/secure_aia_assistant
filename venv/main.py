import os
import json
from datetime import datetime,timezone,timedelta
from typing import Optional, List, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, Field

# ----------------------------
# Setup
# ----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"
MAX_HISTORY = 10  # excluding the system message
TASKS_FILE = "tasks.json"

SYSTEM_PROMPT = """
You are my secure daily AI assistant.

When the user asks to create a reminder/task/to-do, use the create_task tool.
When the user asks to show/list/check tasks, use the list_tasks tool.

If the user is just chatting or asking questions, do NOT call tools.

If you call a tool, keep tool arguments minimal and correct.
""".strip()

messages: List[Dict[str, Any]] = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

# ----------------------------
# Tool schema (argument validation)
# ----------------------------
class CreateTaskArgs(BaseModel):
    title: str = Field(..., min_length=1, description="Short title for the task")
    due_date: Optional[str] = Field(
        None,
        description="Due date in YYYY-MM-DD (optional)"
    )
    notes: Optional[str] = Field(None, description="Extra notes (optional)")

class ListTasksArgs(BaseModel):
    status: str = Field("open", description="one of: open, done, all")


# ----------------------------
# Tool implementation (your code runs this)
# ----------------------------
def _load_tasks() -> List[dict]:
    if not os.path.exists(TASKS_FILE):
        return []
    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_tasks(tasks: List[dict]) -> None:
    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)


def create_task(title: str, due_date: Optional[str] = None, notes: Optional[str] = None) -> dict:
    # Basic due_date validation (optional)
    if due_date is not None:
        try:
            datetime.strptime(due_date, "%Y-%m-%d")
        except ValueError:
            return {"ok": False, "error": "due_date must be in YYYY-MM-DD format (e.g., 2026-02-25)"}

    tasks = _load_tasks()
    task = {
        "id": f"tsk_{len(tasks) + 1}",
        "title": title.strip(),
        "due_date": due_date,
        "notes": notes,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "done": False,
    }
    tasks.append(task)
    _save_tasks(tasks)
    return {"ok": True, "task": task}


def list_tasks(status: str = "open") -> dict:
    tasks = _load_tasks()

    status = (status or "open").strip().lower()
    if status not in ("open", "done", "all"):
        return {"ok": False, "error": "status must be one of: open, done, all"}

    if status == "open":
        filtered = [t for t in tasks if not t.get("done", False)]
    elif status == "done":
        filtered = [t for t in tasks if t.get("done", False)]
    else:
        filtered = tasks

    # Sort: due_date first (None goes last), then created_at
    def sort_key(t):
        due = t.get("due_date") or "9999-12-31"
        created = t.get("created_at") or ""
        return (due, created)

    filtered_sorted = sorted(filtered, key=sort_key)

    return {"ok": True, "count": len(filtered_sorted), "tasks": filtered_sorted}


# ----------------------------
# Tool definitions for the model
# ----------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": "Create a task/reminder and store it locally.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Task title"},
                    "due_date": {"type": "string", "description": "Due date in YYYY-MM-DD (optional)"},
                    "notes": {"type": "string", "description": "Extra notes (optional)"},
                },
                "required": ["title"],
                "additionalProperties": False,
            },
            
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "List tasks with optional filtering by status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "description": "Status to filter tasks by (open, done, all)"},
                },
                "required": [],
                "additionalProperties": False,
            },
            
        },
    }
]


def trim_messages(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(msgs) <= MAX_HISTORY + 1:
        return msgs
    return [msgs[0]] + msgs[-MAX_HISTORY:]


print("Secure AI Assistant + Tasks (type 'exit' to quit)\n")
print(f"Tasks will be saved to: {TASKS_FILE}\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    messages.append({"role": "user", "content": user_input})
    messages = trim_messages(messages)

    # 1) Ask the model (it may decide to call a tool)
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
    )

assistant_msg = response.choices[0].message
messages.append(assistant_msg)

# 2) If tool calls exist, execute them and send results back
tool_calls = getattr(assistant_msg, "tool_calls", None) or []

if tool_calls:
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        raw_args = tool_call.function.arguments

        if tool_name == "create_task":
            try:
                args = CreateTaskArgs.model_validate_json(raw_args)
                result = create_task(title=args.title, due_date=args.due_date, notes=args.notes)
            except ValidationError as e:
                result = {"ok": False, "error": "Invalid tool arguments", "details": e.errors()}
            except Exception as e:
                result = {"ok": False, "error": f"Tool execution failed: {str(e)}"}

        elif tool_name == "list_tasks":
            try:
                args = ListTasksArgs.model_validate_json(raw_args) if raw_args else ListTasksArgs()
                result = list_tasks(status=args.status)
            except ValidationError as e:
                result = {"ok": False, "error": "Invalid tool arguments", "details": e.errors()}
            except Exception as e:
                result = {"ok": False, "error": f"Tool execution failed: {str(e)}"}

        else:
            result = {"ok": False, "error": f"Unknown tool: {tool_name}"}

        # Provide tool output back to the model (IMPORTANT)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            }
        )

    # 3) Ask the model again so it can respond to the user using tool results
    response2 = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
    )
    final_reply = response2.choices[0].message.content
    print("Assistant:", final_reply)
    messages.append({"role": "assistant", "content": final_reply})
    messages = trim_messages(messages)

else:
    # Normal reply, no tools used
    print("Assistant:", assistant_msg.content)
    messages = trim_messages(messages)