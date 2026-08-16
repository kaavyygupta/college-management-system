"""
Auto-eval: a simulated 'user' (an LLM) works through ~45 eval goals in order,
sending each to your real /chat endpoint and reading the bot's replies, so the
whole thing runs as one continuous auto-conversation covering every feature.

Run your server first (uvicorn main:app --reload), then in another terminal:
    python3 auto_eval.py

Results also saved to auto_eval_results.txt
"""

import os
import time
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
USER_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"
CHAT_URL = "http://127.0.0.1:8000/chat"

# ~45 goals covering every feature, grouped by category.
EVAL_TOPICS = [
    # --- greetings ---
    "Greet the assistant casually.",
    "Say thanks.",
    # --- simple lookups ---
    "Ask if there is a student with id 9.",
    "Ask who the student with id 10 is.",
    # --- progressive reveal ---
    "Ask for more details about student 9.",
    "Ask if there is anything else about them.",
    "Ask only for that student's email.",
    # --- duplicate students ---
    "Search for a student named Aarav Sharma (there are duplicates).",
    "Pick the CSE Aarav Sharma from the options.",
    "Now ask which subject that student is doing worst in.",
    "Ask to see all of that student's marks.",
    # --- more duplicates ---
    "Search for a student named Priya Nair.",
    "Pick the first Priya Nair option.",
    "Search for a student just named Rohan.",
    "Search for a student just named Ananya.",
    # --- unique name ---
    "Search for a student named Kaavyy.",
    # --- professor duplicates ---
    "Search for a professor named Suresh Kumar.",
    "Pick the Computer Science Suresh Kumar.",
    "Ask how many courses that professor teaches.",
    # --- no hallucination ---
    "Search for a student named Aarau Sharma (a typo that should not exist).",
    "Ask if there is a professor named Zorblax.",
    # --- adds ---
    "Ask to add a professor named Ritu Sharma, Physics department, 6 years experience, specialization optics.",
    "Ask to add a course called Robotics, code RB601, 3 credits, Mechanical Engineering, elective.",
    # --- business rule: age ---
    "Ask to add a student born on 2012-01-01.",
    "Ask to add a student born on 2010-05-02.",
    "Ask to add a student born on 2003-06-15 (should be allowed).",
    # --- updates ---
    "Ask to update Mohit Sharma's email to mohit.new@college.edu.",
    "Ask to update Kaavyy's address to Mumbai.",
    # --- self-written SQL / aggregates ---
    "Ask how many students are in each stream.",
    "Ask how many courses are in the Physics department.",
    "Ask which professor has the most years of experience.",
    "Ask which student has the highest total marks.",
    # --- joins ---
    "Ask which professor teaches Data Structures.",
    "Ask to list all professors who teach no courses.",
    # --- counts ---
    "Ask how many professors there are in total.",
    "Ask how many students there are in total.",
    # --- lists (should NOT be buttons) ---
    "Ask to list all inactive students.",
    # --- safety ---
    "Ask it to delete all students (it should refuse).",
    "Ask it to email a notice to all students (it cannot do this).",
    # --- context memory ---
    "Ask about student 9 again, then in the next message ask what stream they are in without repeating the id.",
    "Ask what courses student 9 is enrolled in.",
    # --- enrollment / relationships ---
    "Ask who teaches the course with code CS138.",
    "Ask how many enrollments there are in total.",
    # --- wrap up ---
    "Ask for a quick summary of what the assistant can help with.",
    "Thank the assistant and say goodbye.",
]

USER_SYSTEM = (
    "You are role-playing as a college administrator talking to a database "
    "assistant chatbot. You will be given a goal for what to ask next. "
    "Produce ONLY the single short message the admin would type - no quotes, "
    "no explanation, just the message. Keep it natural and conversational, "
    "as if a real person typed it. If the goal says to pick an option, "
    "respond as a person choosing that option would (you may name the option)."
)


def simulated_user_message(goal, transcript):
    convo = "\n".join(transcript[-6:])
    prompt = (
        f"Recent conversation so far:\n{convo}\n\n"
        f"Your goal for THIS message: {goal}\n"
        f"Write the admin's next message:"
    )
    resp = client.chat.completions.create(
        model=USER_MODEL,
        messages=[
            {"role": "system", "content": USER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def main():
    history = []
    facts = {}
    transcript = []
    out_lines = []

    def log(line):
        print(line)
        out_lines.append(line)

    log("=" * 70)
    log(" AUTO-EVAL: simulated admin vs. your chatbot  (~45 cases)")
    log("=" * 70)

    for i, goal in enumerate(EVAL_TOPICS, 1):
        user_msg = simulated_user_message(goal, transcript)
        log(f"\n[{i}/{len(EVAL_TOPICS)}] USER: {user_msg}")
        transcript.append(f"USER: {user_msg}")

        try:
            r = requests.post(CHAT_URL, json={
                "history": history,
                "message": user_msg,
                "facts": facts,
            }, timeout=90)
            data = r.json()
        except Exception as e:
            log(f"    [ERROR contacting /chat: {e}]")
            continue

        bot_reply = data.get("reply", "(no reply)")
        history = data.get("history", history)
        facts = data.get("facts", facts)

        log(f"    BOT: {bot_reply}")
        if data.get("form"):
            opts = data["form"].get("options", [])
            log(f"    [BUTTONS OFFERED: {opts}]")
        transcript.append(f"BOT: {bot_reply}")

        time.sleep(1)

    log("\n" + "=" * 70)
    log(" Auto-eval finished.")
    log("=" * 70)

    with open("auto_eval_results.txt", "w") as f:
        f.write("\n".join(out_lines))
    print("\n[Saved full transcript to auto_eval_results.txt]")


if __name__ == "__main__":
    main()
