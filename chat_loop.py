import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from database import SessionLocal
import models

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = "qwen/qwen3-30b-a3b-instruct-2507"


# ---------------------------------------------------------------------------
# The real database function — returns ALL fields so the model HAS everything.
# How much it actually reveals is controlled by the system message below.
# ---------------------------------------------------------------------------
def get_student_by_id(student_id: int):
    db = SessionLocal()
    try:
        student = db.query(models.Student).filter(
            models.Student.student_id == student_id
        ).first()

        if not student:
            return {"error": f"No student found with id {student_id}"}

        return {
            "student_id": student.student_id,
            "full_name": student.full_name,
            "email": student.email,
            "date_of_birth": str(student.date_of_birth),
            "gender": student.gender,
            "stream": student.stream,
            "year_of_enrollment": student.year_of_enrollment,
            "current_year": student.current_year,
            "current_semester": student.current_semester,
            "address": student.address,
            "status": student.status,
        }
    finally:
        db.close()


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_student_by_id",
            "description": "Get a student's full details using their student ID number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {
                        "type": "integer",
                        "description": "The student's ID number",
                    }
                },
                "required": ["student_id"],
            },
        },
    }
]


def run_tool(function_name, arguments):
    if function_name == "get_student_by_id":
        return get_student_by_id(arguments["student_id"])
    return {"error": "Unknown tool"}


# ---------------------------------------------------------------------------
# The system message is where the tiered-reveal behaviour lives.
# We describe the behaviour in plain English; the model applies it to
# whatever the user actually types, however they phrase it.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant for a college management system that answers "
    "questions about students.\n\n"
    "When a user gives you a student ID, use the get_student_by_id tool to look "
    "them up. You will receive ALL of the student's fields, but you must reveal "
    "them GRADUALLY based on what the user asks for:\n\n"
    "1. FIRST REVEAL: When a student is first looked up, respond ONLY by "
    "confirming who the ID belongs to, using their name and nothing else. "
    "Phrase your reply naturally so it responds to HOW the user asked. If they "
    "asked a yes/no question like 'do we have a student with id 9?', begin by "
    "answering it, e.g. 'Yes, we do - that's [name]. Is this the student you're "
    "looking for?'. If they just gave a bare ID, you can say 'This ID belongs "
    "to [name]. Is this who you're looking for?'. Either way, do NOT mention "
    "their stream, year, semester, email, or any other field at this stage. Do "
    "NOT welcome or address the student directly. Keep it to just the name plus "
    "a short confirmation question.\n"
    "2. AFTER CONFIRMATION: If the user simply confirms the person (e.g. 'yes', "
    "'yeah', 'that's him', 'correct', 'yep'), do NOT reveal any details yet. "
    "Just acknowledge briefly and ask what they'd like to know - for example: "
    "'Great! What would you like to know about them?'. Wait for the user to ask "
    "for specific information before revealing anything.\n"
    "3. If the user then asks for more details, share: email, date of birth, "
    "gender, and stream.\n"
    "4. If the user asks for even more / everything else, then share the "
    "remaining details: year of enrollment, current year, current semester, "
    "address, and status.\n\n"
    "IMPORTANT: The user will NOT use fixed commands. They may phrase requests "
    "in any way - for example 'just the name', 'what's their email?', 'which "
    "stream?', 'tell me everything', 'anything more?'. Understand their intent "
    "and answer accordingly. If they ask for one specific field, give just that "
    "field. Be natural, friendly, and conversational."
)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

print("=" * 55)
print(" College Student Chatbot")
print("=" * 55)
print("Hello! Which student would you like to look up?")
print("Please enter the student ID (or type 'quit' to exit).\n")

while True:
    user_input = input("You: ")

    if user_input.strip().lower() in ("quit", "exit"):
        print("Goodbye!")
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
    )
    reply = response.choices[0].message

    if reply.tool_calls:
        messages.append(reply)

        for tool_call in reply.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            result = run_tool(function_name, arguments)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

        final_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        answer = final_response.choices[0].message.content
        messages.append({"role": "assistant", "content": answer})
        print(f"Bot: {answer}\n")

    else:
        answer = reply.content
        messages.append({"role": "assistant", "content": answer})
        print(f"Bot: {answer}\n")
