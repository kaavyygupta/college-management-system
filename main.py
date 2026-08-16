from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
from datetime import date
from pydantic import BaseModel
from typing import List, Optional, TypedDict
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from sqlalchemy import cast, String, text
from langgraph.graph import StateGraph, END

load_dotenv()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

load_dotenv()
 
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
 
CHAT_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"
 
CHAT_SYSTEM_PROMPT = (
    # ---------- ROLE ----------
    "You are an assistant for a college management system. You help the "
    "college administrator look up, add, and update records for students, "
    "professors, courses, and enrollments.\n\n"

    # ---------- REASONING (CoT) ----------
    "REASONING PROCESS - work through this silently before every reply:\n"
    "1. INTENT: What is the user actually asking for? Is it a question, a "
    "request to change data, a clarification of something earlier, or just "
    "small talk?\n"
    "2. CONTEXT: Do I already have the answer from a tool result earlier in "
    "THIS conversation, or from known facts? If yes, use it - don't refetch.\n"
    "3. GAPS: What is missing? Can a tool get it, or does only the user "
    "have it? Tools answer data questions; users answer preference and "
    "detail questions.\n"
    "4. MINIMAL ACTION: What is the smallest number of tool calls that "
    "answers this? Fetch only what was asked for - nothing extra.\n"
    "5. VERIFY: Before replying, actually read the tool result and count "
    "the rows. If more than one row matches a name, treat it as multiple "
    "matches and show selection buttons - never say 'there is one' without "
    "checking the count. Only state values that appear in the result.\n\n"
    "CLARIFYING QUESTIONS:\n"
    "- If information is missing and only the user can provide it, ask for "
    "it. Keep asking until you have what you need.\n"
    "- Do not ask for anything a tool can already retrieve.\n"
    "- Ask for one thing at a time where possible.\n\n"

    # ---------- TOOLS ----------
    "TOOLS:\n"
    "- run_query: write your own SQL SELECT for ANY read question, "
    "including joins and aggregates. This is your primary tool.\n"
    "- write_record: insert a new row.\n"
    "- update_record: change or fill in fields on an existing row.\n\n"

    # ---------- SCHEMA ----------
    "DATABASE SCHEMA (exact table and column names):\n"
    "students: student_id, full_name, email, date_of_birth, gender, stream, "
    "year_of_enrollment, current_year, current_semester, address, status\n"
    "professors: professor_id, full_name, email, department, "
    "years_of_experience, specialization\n"
    "courses: course_id, course_name, course_code, credits, department, "
    "course_type, description\n"
    "course_offering: offering_id, course_id, professor_id, semester, "
    "academic_year, status\n"
    "enrollment: enrollment_id, student_id, offering_id, attendance, "
    "internal_marks, exam_marks, total_marks, grade, grade_points, status, "
    "enrollment_date\n\n"

    # ---------- IDENTIFYING STUDENTS ----------
    "IDENTIFYING PEOPLE BY NAME:\n"
    "- When searching for a person by name, query ONLY the relevant table "
    "(students for students, professors for professors). Do NOT search "
    "courses, offerings, or enrollments for a person's name. One targeted "
    "query is enough.\n"
    "- Before answering ANY request that refers to a person by name - "
    "lookups, marks, grades, courses, enrollments, anything - FIRST run a "
    "query listing ALL records matching that name, selecting identifying "
    "columns like email and stream. Never use LIMIT 1 to skip this.\n"
    "- If more than one record matches, STOP: do not run any further query "
    "and do not answer yet. Set text to a short line asking the user to "
    "pick one, and put EVERY match in the form field as buttons. This "
    "applies EVERY time duplicates occur, on every turn, no matter how many "
    "times it has already happened in this conversation. Never list matches "
    "in text - they appear only as buttons.\n"
    "- Only after the user selects one may you run the query that answers "
    "their actual question, filtered to that exact record.\n"
    "- Once a match is selected (by button or email), treat it as confirmed "
    "- never ask again who they mean.\n"
    "- After the user selects a specific person from the disambiguation "
    "buttons, do NOT reveal their full details. Confirm the selection by "
    "name only and ask what they would like to know - then follow the "
    "gradual reveal rules from there.\n"
    "- If only one record matches, continue normally.\n\n"

    # ---------- PROGRESSIVE REVEAL ----------
    "REVEALING STUDENT DETAILS - reveal gradually, never all at once:\n"
    "1. First mention: give the NAME only, phrased to match how they asked "
    "(e.g. 'Yes, that's [name]. Is this who you're looking for?'). Share "
    "nothing else yet - no stream, year, email, or semester.\n"
    "2. If they simply confirm ('yes', 'correct'), acknowledge and ask what "
    "they'd like to know. Still reveal nothing.\n"
    "3. If they ask for more: email, date of birth, gender, stream.\n"
    "4. If they ask for everything else: year of enrollment, current year, "
    "current semester, address, status.\n"
    "5. If they ask for one specific field, give just that field.\n"
    "This applies only to individual student profiles - lists, counts, and "
    "general questions are answered directly.\n\n"

    # ---------- BUSINESS RULES ----------
    "ENROLLMENT RULES:\n"
    "- Whenever a date of birth is provided or known, FIRST determine the "
    "student's age before doing anything else - before asking for name, "
    "stream, or any other detail.\n"
    "- Never calculate age yourself. Use run_query with CURRENT_DATE to let "
    "the database compute it, and use only the value it returns.\n"
    "- If that age is under 17, reject the enrollment immediately, state the "
    "age, and do not ask for any further details.\n"
    "- Only if the student is 17 or older, continue collecting their "
    "remaining details.\n"
    "- If no date of birth has been given yet, ask for it before anything "
    "else.\n\n"

    # ---------- EFFICIENCY ----------
    "EFFICIENCY:\n"
    "- Use as few tool calls as possible. Stop calling tools the moment you "
    "can answer.\n"
    "- When identifying a student, look up ONLY that student. Do not fetch "
    "their enrollments, courses, or professors unless asked.\n"
    "- If the message is just a greeting or small talk, reply briefly and "
    "call no tools at all.\n\n"
    "- Being efficient never means skipping a lookup. If the user asks "
    "about data, you MUST call a tool before answering. Never say you "
    "don't have information without querying first.\n\n"

    # ---------- ACCURACY ----------
    "ACCURACY RULES:\n"
    "- Never state a specific value (ID, name, email, date, marks, grade, "
    "count) unless it came from a tool result in this conversation.\n"
    "- If a tool result doesn't contain what was asked, say you don't have "
    "that information. Never fill the gap with a plausible guess.\n"
    "- Never claim an action succeeded unless a tool returned success.\n"
    "- If a query returns NO_RECORDS_FOUND or an empty result, tell the "
    "user plainly that no such record exists. NEVER invent students, "
    "emails, or any details to fill the gap.\n"
    "- You have NO memory of database contents. For EVERY question about "
    "any record - even one discussed earlier - you MUST run a fresh "
    "run_query THIS turn. Never answer about a record's existence, count, "
    "or details from memory. When searching a person by name, query the "
    "matching table for ALL rows with that name, never LIMIT 1.\n"
    "- When querying a student's marks, subjects, courses, or enrollments, "
    "never filter by a status column. Return all their records regardless "
    "of status, unless the user explicitly asks for a specific status.\n"
    "- If no tool can do what the user asked, say so plainly - do not "
    "pretend to perform the action.\n"
    "- CRITICAL: Only filter a query by values the user explicitly gave you "
    "or that you retrieved from a previous tool result. NEVER add WHERE "
    "conditions based on assumptions - do not guess status values, "
    "semesters, academic years, or any other filter the user didn't ask "
    "for. When following a relationship (like an enrollment to its "
    "professor), use the linking IDs only. Extra guessed filters exclude "
    "real data and cause wrong 'not found' answers.\n\n"

    # ---------- TONE ----------
    "TONE: Natural and conversational. Users phrase things any way they "
    "like - understand intent, don't expect fixed commands.\n\n"

    # ---------- OUTPUT ----------
    "OUTPUT FORMAT: Respond with ONLY a valid JSON object. No markdown, no "
    "backticks, no text outside the JSON. Exactly these four keys:\n"
    '{"thinking": "...", "facts": {}, "text": "...", "form": null}\n'
    "- thinking: your reasoning from the process above. Never shown to the "
    "user.\n"
    "- facts: concrete values confirmed this turn, e.g. "
    '{"student_id": 9, "full_name": "Kaavyy"}. Use {} if none.\n'
    "- text: the reply the user sees. Always required.\n"
    "- form: null normally. For a choice between fixed options use "
    '{"type": "buttons", "options": ["A", "B"]}. Always use buttons for '
    "yes/no questions and for choosing between multiple matches.\n"
)
 
chat_tools = [
    {
        "type": "function",
        "function": {
            "name": "run_query",
            "description": "Run a read-only SQL SELECT query against the college database. Use this for ANY question about data. Write standard PostgreSQL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A SELECT query. Must start with SELECT."}
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_record",
            "description": "Insert a new row into a table. Use when the user wants to add a student, professor, course, offering, or enrollment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "One of: students, professors, courses, course_offering, enrollment"},
                    "data": {"type": "object", "description": "Column name to value mapping"},
                },
                "required": ["table", "data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_record",
            "description": "Update one or more fields on an existing row in any table. Use whenever the user wants to change, correct, set, or fill in existing data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "students, professors, courses, course_offering, or enrollment"},
                    "id_column": {"type": "string", "description": "Primary key column name, e.g. professor_id"},
                    "id_value": {"type": "integer", "description": "The row's primary key value"},
                    "data": {"type": "object", "description": "Column names mapped to their new values"},
                },
                "required": ["table", "id_column", "id_value", "data"],
            },
        },
    },
]
print("TOOL COUNT:", len(chat_tools), [t["function"]["name"] for t in chat_tools])

def chat_get_student_by_id(student_id: int, db: Session):
    student = db.query(models.Student).filter(models.Student.student_id == student_id).first()
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
 
def chat_get_all_students(db: Session):
    students = db.query(models.Student).all()
    return [
        {
            "student_id": s.student_id,
            "full_name": s.full_name,
            "stream": s.stream,
            "current_year": s.current_year,
            "status": s.status,
        }
        for s in students
    ]
 
def chat_count_students(db: Session):
    return {"total_students": db.query(models.Student).count()}
 

def chat_get_professor_by_id(professor_id: int, db: Session):
    professor = db.query(models.Professor).filter(models.Professor.professor_id == professor_id).first()
    if not professor:
        return {"error": f"No professor found with id {professor_id}"}
    return {
        "professor_id": professor.professor_id,
        "full_name": professor.full_name,
        "email": professor.email,
        "department": professor.department,
        "years_of_experience": professor.years_of_experience,
        "specialization": professor.specialization,
    }
 
def chat_get_all_professors(db: Session):
    professors = db.query(models.Professor).all()
    return [
        {
            "professor_id": p.professor_id,
            "full_name": p.full_name,
            "department": p.department,
            "specialization": p.specialization,
        }
        for p in professors
    ]
 
def chat_get_course_by_id(course_id: int, db: Session):
    course = db.query(models.Course).filter(models.Course.course_id == course_id).first()
    if not course:
        return {"error": f"No course found with id {course_id}"}
    return {
        "course_id": course.course_id,
        "course_name": course.course_name,
        "course_code": course.course_code,
        "credits": course.credits,
        "department": course.department,
        "course_type": course.course_type,
        "description": course.description,
    }

def chat_get_enrollments_by_student(student_id: int, db: Session):
    enrollments = db.query(models.Enrollment).filter(
        models.Enrollment.student_id == student_id
    ).all()
    if not enrollments:
        return {"error": f"No enrollments found for student {student_id}"}
    return [
        {
            "course_name": e.offering.course.course_name,
            "total_marks": e.total_marks,
            "grade": e.grade,
        }
        for e in enrollments
    ]
 
def chat_get_all_courses(db: Session):
    courses = db.query(models.Course).all()
    return [
        {
            "course_id": c.course_id,
            "course_name": c.course_name,
            "course_code": c.course_code,
            "department": c.department,
        }
        for c in courses
    ]
 
def chat_get_enrollment_details(enrollment_id: int, db: Session):
    enrollment = db.query(models.Enrollment).filter(
        models.Enrollment.enrollment_id == enrollment_id
    ).first()
    if not enrollment:
        return {"error": f"No enrollment found with id {enrollment_id}"}
    return {
        "enrollment_id": enrollment.enrollment_id,
        "student_name": enrollment.student.full_name,
        "course_id": enrollment.offering.course.course_id,
        "course_name": enrollment.offering.course.course_name,
        "course_code": enrollment.offering.course.course_code,
        "professor_name": enrollment.offering.professor.full_name,
        "semester": enrollment.offering.semester,
        "academic_year": enrollment.offering.academic_year,
        "attendance": enrollment.attendance,
        "internal_marks": enrollment.internal_marks,
        "exam_marks": enrollment.exam_marks,
        "total_marks": enrollment.total_marks,
        "grade": enrollment.grade,
        "status": enrollment.status,
    }
 
def chat_get_students_by_course(course_identifier: str, db: Session):
    course = db.query(models.Course).filter(
        (models.Course.course_code == course_identifier) |
        (cast(models.Course.course_id, String) == course_identifier)
        ).first()
    if not course:
        return {"error": f"No course found matching '{course_identifier}'"}

    course_id = course.course_id
 
    enrollments = (
        db.query(models.Enrollment)
        .join(models.CourseOffering)
        .filter(models.CourseOffering.course_id == course_id)
        .all()
    )
 
    if not enrollments:
        return {"course_name": course.course_name, "students": [], "message": "No students currently enrolled."}
 
    return {
        "course_name": course.course_name,
        "students": [
            {"student_id": e.student.student_id, "full_name": e.student.full_name}
            for e in enrollments
        ],
    }

ALLOWED_TABLES = {"students", "professors", "courses", "course_offering", "enrollment"}
 
 
def chat_run_query(sql: str, db: Session):
    clean = sql.strip().rstrip(";")
    if not clean.lower().startswith("select"):
        return {"error": "Only SELECT queries are allowed."}
    low = clean.lower()
    if "full_name" in low and "limit 1" in low:
        return {"error": "Do not use LIMIT 1 on a name search. First run the query without LIMIT to see all matching records, then handle duplicates."}
    
    import re
    if "enrollment" in low or "e.status" in low:
        clean = re.sub(r"(?i)\s+AND\s+e?\.?status\s*=\s*'(active|enrolled)'", "", clean)

    try:
        rows = db.execute(text(clean)).mappings().all()
        return [{k: str(v) for k, v in r.items()} for r in rows] if rows else {"error": "NO_RECORDS_FOUND: zero matching records exist. Tell the user no such record exists. Do NOT invent any names, emails, or details."}
    except Exception as e:
        db.rollback()
        return {"error": f"Query failed: {e}. Do NOT repeat the same query. Fix the SQL based on this error, or if you cannot, tell the user you couldn't retrieve it."}
 
 
def chat_write_record(table: str, data: dict, db: Session):
    if table not in ALLOWED_TABLES:
        return {"error": f"Table '{table}' not allowed."}

    if table == "students" and data.get("date_of_birth"):
        from datetime import date as _date
        dob = _date.fromisoformat(str(data["date_of_birth"]))
        age = (_date.today() - dob).days // 365
        if age < 17:
            return {"error": f"Cannot enroll: student is {age} years old. Minimum age is 17."}

    try:
        cols = ", ".join(data.keys())
        vals = ", ".join(f":{k}" for k in data.keys())
        db.execute(text(f"INSERT INTO {table} ({cols}) VALUES ({vals})"), data)
        db.commit()
        return {"success": True, "table": table, "inserted": data}
    except Exception as e:
        db.rollback()
        return {"error": str(e)}

def chat_update_record(table: str, id_column: str, id_value: int, data: dict, db: Session):
    if table not in ALLOWED_TABLES:
        return {"error": f"Table '{table}' not allowed."}
    try:
        sets = ", ".join(f"{k} = :{k}" for k in data.keys())
        params = {**data, "_id": id_value}
        db.execute(text(f"UPDATE {table} SET {sets} WHERE {id_column} = :_id"), params)
        db.commit()
        return {"success": True, "table": table, "updated": data}
    except Exception as e:
        db.rollback()
        return {"error": str(e)}

def run_chat_tool(function_name: str, arguments: dict, db: Session):
    if function_name == "run_query":
        return chat_run_query(arguments["sql"], db)
    elif function_name == "write_record":
        return chat_write_record(arguments["table"], arguments["data"], db)
    elif function_name == "update_record":
        return chat_update_record(arguments["table"], arguments["id_column"], arguments["id_value"], arguments["data"], db)
    else:
        return {"error": "Unknown tool"}

#  LANGGRAPH SETUP
class ChatState(TypedDict):
    messages: list
    force_first_tool: bool


def build_chat_graph(db: Session):

    def agent_node(state: ChatState):
        has_tool_result = any(m.get("role") == "tool" for m in state["messages"][-3:])
        response = openrouter_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=state["messages"],
            tools=chat_tools,
            tool_choice="auto" if has_tool_result else "required",
        )
        reply = response.choices[0].message
        print("TOOL CALLS:", reply.tool_calls)
        return {"messages": state["messages"] + [reply.model_dump()]}

    def tools_node(state: ChatState):
        last_message = state["messages"][-1]
        new_messages = []
        for tool_call in last_message["tool_calls"]:
            function_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            result = run_chat_tool(function_name, arguments, db)
            new_messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(result),
            })
        return {"messages": state["messages"] + new_messages}

    def should_continue(state: ChatState):
        last_message = state["messages"][-1]
        if last_message.get("tool_calls"):
            return "tools"
        return "end"

    graph = StateGraph(ChatState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph.compile()
 
def get_student_by_id(student_id: int, db: Session):
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

class StudentCreate(BaseModel):
    full_name: str
    email: str
    date_of_birth: date
    gender: str
    stream: str
    year_of_enrollment: int
    current_year: int
    current_semester: int
    address: str
    status: str = "active"


class StudentResponse(BaseModel):
    student_id: int
    full_name: str
    email: str
    date_of_birth: date
    gender: str
    stream: str
    year_of_enrollment: int
    current_year: int
    current_semester: int
    address: str
    status: str = "active"
    class Config:
        from_attributes = True

@app.get("/students", response_model=List[StudentResponse])
def get_all_students(db: Session = Depends(get_db)):
    students = db.query(models.Student).all()
    return students

@app.get("/students/{student_id}", response_model=StudentResponse)
def get_student(student_id: int, db: Session = Depends(get_db)):
    student = db.query(models.Student).filter(models.Student.student_id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return student

@app.post("/students", response_model=StudentResponse, status_code=201)
def create_student(student: StudentCreate, db: Session = Depends(get_db)):
    db_student = models.Student(**student.dict())
    db.add(db_student)
    db.commit()
    db.refresh(db_student)
    return db_student

@app.put("/students/{student_id}", response_model=StudentResponse)
def update_student(student_id: int, student: StudentCreate, db: Session = Depends(get_db)):
    db_student = db.query(models.Student).filter(models.Student.student_id == student_id).first()
    if not db_student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    for key, value in student.dict().items():
        setattr(db_student, key, value)
    
    db.commit()
    db.refresh(db_student)
    return db_student

@app.delete("/students/{student_id}", status_code=204)
def delete_student(student_id: int, db: Session = Depends(get_db)):
    db_student = db.query(models.Student).filter(models.Student.student_id == student_id).first()
    if not db_student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    db.delete(db_student)
    db.commit()
    return None


class ProfessorCreate(BaseModel):
    full_name: str
    email: str
    department: str
    years_of_experience: int
    specialization: str

class ProfessorResponse(BaseModel):
    professor_id: int
    full_name: str
    email: str
    department: str
    years_of_experience: int
    specialization: str

    class Config:
        from_attributes = True

@app.get("/professors", response_model=List[ProfessorResponse])
def get_all_professors(db: Session = Depends(get_db)):
    professors = db.query(models.Professor).all()
    return professors

@app.get("/professors/{professor_id}", response_model=ProfessorResponse)
def get_professor(professor_id: int, db: Session = Depends(get_db)):
    professor = db.query(models.Professor).filter(models.Professor.professor_id == professor_id).first()
    if not professor:
        raise HTTPException(status_code=404, detail="Professor not found")
    return professor

@app.post("/professors", response_model=ProfessorResponse, status_code=201)
def create_professor(professor: ProfessorCreate, db: Session = Depends(get_db)):
    db_professor = models.Professor(**professor.dict())
    db.add(db_professor)
    db.commit()
    db.refresh(db_professor)
    return db_professor

@app.put("/professors/{professor_id}", response_model=ProfessorResponse)
def update_professor(professor_id: int, professor: ProfessorCreate, db: Session = Depends(get_db)):
    db_professor = db.query(models.Professor).filter(models.Professor.professor_id == professor_id).first()
    if not db_professor:
        raise HTTPException(status_code=404, detail="Professor not found")
    
    for key, value in professor.dict().items():
        setattr(db_professor, key, value)
    
    db.commit()
    db.refresh(db_professor)
    return db_professor

@app.delete("/professors/{professor_id}", status_code=204)
def delete_professor(professor_id: int, db: Session = Depends(get_db)):
    db_professor = db.query(models.Professor).filter(models.Professor.professor_id == professor_id).first()
    if not db_professor:
        raise HTTPException(status_code=404, detail="Professor not found")
    
    db.delete(db_professor)
    db.commit()
    return None


class CourseCreate(BaseModel):
    course_name: str
    course_code: str
    credits: int
    department: str
    course_type: str
    description: str

class CourseResponse(BaseModel):
    course_id: int
    course_name: str
    course_code: str
    credits: int
    department: str
    course_type: str
    description: str

    class Config:
        from_attributes = True

@app.get("/courses", response_model=List[CourseResponse])
def get_all_courses(db: Session = Depends(get_db)):
    courses = db.query(models.Course).all()
    return courses

@app.get("/courses/{course_id}", response_model=CourseResponse)
def get_course(course_id: int, db: Session = Depends(get_db)):
    course = db.query(models.Course).filter(models.Course.course_id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    return course

@app.post("/courses", response_model=CourseResponse, status_code=201)
def create_course(course: CourseCreate, db: Session = Depends(get_db)):
    existing = db.query(models.Course).filter(models.Course.course_code == course.course_code).first()
    if existing:
        raise HTTPException(status_code=400, detail="Course code already exists")

    db_course = models.Course(**course.dict())
    db.add(db_course)
    db.commit()
    db.refresh(db_course)
    return db_course

@app.put("/courses/{course_id}", response_model=CourseResponse)
def update_course(course_id: int, course: CourseCreate, db: Session = Depends(get_db)):
    db_course = db.query(models.Course).filter(models.Course.course_id == course_id).first()
    if not db_course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    for key, value in course.dict().items():
        setattr(db_course, key, value)
    
    db.commit()
    db.refresh(db_course)
    return db_course

@app.delete("/courses/{course_id}", status_code=204)
def delete_course(course_id: int, db: Session = Depends(get_db)):
    db_course = db.query(models.Course).filter(models.Course.course_id == course_id).first()
    if not db_course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    db.delete(db_course)
    db.commit()
    return None


class EnrollmentCreate(BaseModel):
    student_id: int
    offering_id: int

class EnrollmentResponse(BaseModel):
    enrollment_id: int
    student_id: int
    offering_id: int
    attendance: Optional[int]
    internal_marks: Optional[int]
    exam_marks: Optional[int]
    total_marks: Optional[int]
    grade: Optional[str]
    grade_points: Optional[float]
    status: str
    enrollment_date: date

    class Config:
        orm_mode = True   

class EnrollmentMarksUpdate(BaseModel):
    attendance: Optional[int] = None
    internal_marks: Optional[int] = None
    exam_marks: Optional[int] = None
    total_marks: Optional[int] = None
    grade: Optional[str] = None
    grade_points: Optional[float] = None


class EnrollmentStatusUpdate(BaseModel):
    status: str

class StudentBrief(BaseModel):
    full_name: str
    email: str
    current_year: int

    class Config:
        from_attributes = True

class CourseBrief(BaseModel):
    course_name: str
    course_code: str
    department: str

    class Config:
        from_attributes = True


class ProfessorBrief(BaseModel):
    full_name: str
    email: str

    class Config:
        from_attributes = True

class CourseOfferingBrief(BaseModel):
    semester: int
    academic_year: str
    course: CourseBrief
    professor: ProfessorBrief

    class Config:
        from_attributes = True

class EnrollmentDetailResponse(BaseModel):
    enrollment_id: int
    student: StudentBrief
    offering: CourseOfferingBrief
    attendance: Optional[int]
    internal_marks: Optional[int]
    exam_marks: Optional[int]
    total_marks: Optional[int]
    grade: Optional[str]
    grade_points: Optional[float]
    status: str
    enrollment_date: date

    class Config:
        from_attributes = True  

@app.get("/enrollment", response_model=List[EnrollmentResponse])
def get_all_enrollments(db: Session = Depends(get_db)):
    return db.query(models.Enrollment).all()


@app.get("/enrollment/{enrollment_id}", response_model=EnrollmentResponse)
def get_enrollment(enrollment_id: int, db: Session = Depends(get_db)):
    enrollment = db.query(models.Enrollment).filter(models.Enrollment.enrollment_id == enrollment_id).first()
    if not enrollment:
        raise HTTPException(status_code=404, detail="Enrollment not found")
    return enrollment

@app.get("/enrollment/{enrollment_id}/details", response_model=EnrollmentDetailResponse)
def get_enrollment_details(enrollment_id: int, db: Session = Depends(get_db)):
    enrollment = db.query(models.Enrollment).filter(models.Enrollment.enrollment_id == enrollment_id).first()
    if not enrollment:
        raise HTTPException(status_code=404, detail="Enrollment not found")
    return enrollment

@app.post("/enrollment", response_model=EnrollmentResponse, status_code=201)
def create_enrollment(enrollment: EnrollmentCreate, db: Session = Depends(get_db)):
    # check student exists
    student = db.query(models.Student).filter(models.Student.student_id == enrollment.student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    # check offering exists
    offering = db.query(models.CourseOffering).filter(models.CourseOffering.offering_id == enrollment.offering_id).first()
    if not offering:
        raise HTTPException(status_code=404, detail="Course offering not found")

    # prevent duplicate enrollment in the same offering
    existing = db.query(models.Enrollment).filter(
        models.Enrollment.student_id == enrollment.student_id,
        models.Enrollment.offering_id == enrollment.offering_id
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Student already enrolled in this offering")

    db_enrollment = models.Enrollment(
        student_id=enrollment.student_id,
        offering_id=enrollment.offering_id,
        enrollment_date=date.today(),
        status="ongoing"
    )
    db.add(db_enrollment)
    db.commit()
    db.refresh(db_enrollment)
    return db_enrollment


@app.put("/enrollment/{enrollment_id}/marks", response_model=EnrollmentResponse)
def update_enrollment_marks(enrollment_id: int, marks: EnrollmentMarksUpdate, db: Session = Depends(get_db)):
    db_enrollment = db.query(models.Enrollment).filter(models.Enrollment.enrollment_id == enrollment_id).first()
    if not db_enrollment:
        raise HTTPException(status_code=404, detail="Enrollment not found")

    for key, value in marks.dict(exclude_unset=True).items():
        setattr(db_enrollment, key, value)

    db.commit()
    db.refresh(db_enrollment)
    return db_enrollment


@app.put("/enrollment/{enrollment_id}/status", response_model=EnrollmentResponse)
def update_enrollment_status(enrollment_id: int, status_update: EnrollmentStatusUpdate, db: Session = Depends(get_db)):
    db_enrollment = db.query(models.Enrollment).filter(models.Enrollment.enrollment_id == enrollment_id).first()
    if not db_enrollment:
        raise HTTPException(status_code=404, detail="Enrollment not found")

    db_enrollment.status = status_update.status
    db.commit()
    db.refresh(db_enrollment)
    return db_enrollment


@app.delete("/enrollment/{enrollment_id}", status_code=204)
def delete_enrollment(enrollment_id: int, db: Session = Depends(get_db)):
    db_enrollment = db.query(models.Enrollment).filter(models.Enrollment.enrollment_id == enrollment_id).first()
    if not db_enrollment:
        raise HTTPException(status_code=404, detail="Enrollment not found")

    db.delete(db_enrollment)
    db.commit()
    return None


class CourseOfferingCreate(BaseModel):
    course_id: int
    professor_id: int
    semester: int
    academic_year: str


class CourseOfferingResponse(BaseModel):
    offering_id: int
    course_id: int
    professor_id: int
    semester: int
    academic_year: str
    status: str

    class Config:
        orm_mode = True  

class CourseOfferingStatusUpdate(BaseModel):
    status: str  

@app.get("/course-offering", response_model=List[CourseOfferingResponse])
def get_all_offerings(db: Session = Depends(get_db)):
    return db.query(models.CourseOffering).all()


@app.get("/course-offering/{offering_id}", response_model=CourseOfferingResponse)
def get_offering(offering_id: int, db: Session = Depends(get_db)):
    offering = db.query(models.CourseOffering).filter(models.CourseOffering.offering_id == offering_id).first()
    if not offering:
        raise HTTPException(status_code=404, detail="Course offering not found")
    return offering


@app.post("/course-offering", response_model=CourseOfferingResponse, status_code=201)
def create_offering(offering: CourseOfferingCreate, db: Session = Depends(get_db)):
    # check course exists
    course = db.query(models.Course).filter(models.Course.course_id == offering.course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    # check professor exists
    professor = db.query(models.Professor).filter(models.Professor.professor_id == offering.professor_id).first()
    if not professor:
        raise HTTPException(status_code=404, detail="Professor not found")

    # prevent duplicate offering: same course + same professor + same semester + same year
    existing = db.query(models.CourseOffering).filter(
        models.CourseOffering.course_id == offering.course_id,
        models.CourseOffering.professor_id == offering.professor_id,
        models.CourseOffering.semester == offering.semester,
        models.CourseOffering.academic_year == offering.academic_year
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="This offering already exists")

    db_offering = models.CourseOffering(**offering.dict(), status="ongoing")
    db.add(db_offering)
    db.commit()
    db.refresh(db_offering)
    return db_offering


@app.put("/course-offering/{offering_id}", response_model=CourseOfferingResponse)
def update_offering(offering_id: int, offering: CourseOfferingCreate, db: Session = Depends(get_db)):
    db_offering = db.query(models.CourseOffering).filter(models.CourseOffering.offering_id == offering_id).first()
    if not db_offering:
        raise HTTPException(status_code=404, detail="Course offering not found")

    for key, value in offering.dict().items():
        setattr(db_offering, key, value)

    db.commit()
    db.refresh(db_offering)
    return db_offering


@app.put("/course-offering/{offering_id}/status", response_model=CourseOfferingResponse)
def update_offering_status(offering_id: int, status_update: CourseOfferingStatusUpdate, db: Session = Depends(get_db)):
    db_offering = db.query(models.CourseOffering).filter(models.CourseOffering.offering_id == offering_id).first()
    if not db_offering:
        raise HTTPException(status_code=404, detail="Course offering not found")

    db_offering.status = status_update.status
    db.commit()
    db.refresh(db_offering)
    return db_offering


@app.delete("/course-offering/{offering_id}", status_code=204)
def delete_offering(offering_id: int, db: Session = Depends(get_db)):
    db_offering = db.query(models.CourseOffering).filter(models.CourseOffering.offering_id == offering_id).first()
    if not db_offering:
        raise HTTPException(status_code=404, detail="Course offering not found")

    db.delete(db_offering)
    db.commit()
    return None

class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
 
 
class ChatRequest(BaseModel):
    history: List[ChatMessage] = []
    message: str
    facts: dict = {}
 
 
class ChatResponse(BaseModel):
    reply: str
    history: List[dict]
    form: Optional[dict] = None
    facts: dict = {}
 
 
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    for m in request.history:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": request.message})

    if request.facts:
        messages.insert(1, {"role": "system", "content": f"Known facts so far: {json.dumps(request.facts)}"})

    graph = build_chat_graph(db)
    try:
        result = graph.invoke({
            "messages": messages,
            "force_first_tool": len(request.history) == 0
        }, config={"recursion_limit": 15})
    except Exception:
        safe_history = [{"role": m.role, "content": m.content} for m in request.history]
        safe_history.append({"role": "user", "content": request.message})
        return ChatResponse(reply="Let me try that again — could you rephrase?", history=safe_history, form=None, facts=request.facts)
    raw = result["messages"][-1]["content"]
    try:
        parsed = json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
        answer = parsed.get("text", raw)
        form = parsed.get("form")
        facts = parsed.get("facts") or {}
    except Exception:
        answer = raw
        form = None
        facts = {}

    if not form:
        for m in result["messages"]:
            if isinstance(m, dict) and m.get("role") == "tool":
                try:
                    data = json.loads(m["content"])
                except Exception:
                    continue
                if isinstance(data, list) and len(data) > 1:
                    names = set(r.get("full_name") for r in data if isinstance(r, dict) and r.get("full_name"))
                    # only disambiguate if EVERY row is the same single name
                    if len(names) == 1 and len(data) <= 6:
                        opts = []
                        for r in data:
                            label = r.get("full_name", "")
                            extra = r.get("email") or r.get("stream") or r.get("department") or ""
                            if extra:
                                label += f" - {extra}"
                            opts.append(label)
                        form = {"type": "buttons", "options": opts}
                        if not answer or "select" not in answer.lower():
                            answer = "There are multiple matches. Please select one:"
                        break

    returned_history = []
    for m in request.history:
        returned_history.append({"role": m.role, "content": m.content})
    returned_history.append({"role": "user", "content": request.message})
    returned_history.append({"role": "assistant", "content": answer})

    merged_facts = {**request.facts, **facts}

    return ChatResponse(reply=answer, history=returned_history, form=form, facts=merged_facts)