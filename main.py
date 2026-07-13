from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
from datetime import date
from pydantic import BaseModel
from typing import List, Optional
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from sqlalchemy import cast, String

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
    "You are a helpful assistant for a college management system. You can "
    "answer questions about students, professors, courses, and enrollments "
    "using the tools provided. Always use a tool to get real data - never "
    "guess or make up information.\n\n"
    "STUDENT LOOKUPS (get_student_by_id): reveal details GRADUALLY:\n"
    "1. FIRST REVEAL: confirm identity by name only. Phrase it naturally "
    "based on how the user asked (answer yes/no questions directly, e.g. "
    "'Yes, we do - that's [name]. Is this who you're looking for?'). Do NOT "
    "share stream, year, semester, email, or anything else yet.\n"
    "2. If the user just confirms (e.g. 'yes'), do NOT reveal details - ask "
    "what they'd like to know.\n"
    "3. If they ask for more, share: email, date of birth, gender, stream.\n"
    "4. If they ask for everything else, share: year of enrollment, current "
    "year, current semester, address, status.\n\n"
    "OTHER TOOLS (get_all_students, count_students, get_professor_by_id, "
    "get_all_professors, get_course_by_id, get_all_courses, "
    "get_enrollment_details): these are general information lookups, not "
    "individual student profiles - just answer directly and naturally, no "
    "need for gradual reveal.\n\n"
    "The user will phrase things in any way, not fixed commands. Understand "
    "their intent. Be natural, friendly, and conversational."
)
 
chat_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_student_by_id",
            "description": "Get a single student's full details using their student ID number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {"type": "integer", "description": "The student's ID number"}
                },
                "required": ["student_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_students",
            "description": "Get a list of ALL students in the college, with basic info for each.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_students",
            "description": "Get the total number of students currently enrolled in the college.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_professor_by_id",
            "description": "Get a single professor's details using their professor ID number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "professor_id": {"type": "integer", "description": "The professor's ID number"}
                },
                "required": ["professor_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_professors",
            "description": "Get a list of ALL professors in the college.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_course_by_id",
            "description": "Get a single course's details using its course ID number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "course_id": {"type": "integer", "description": "The course's ID number"}
                },
                "required": ["course_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_courses",
            "description": "Get a list of ALL courses offered by the college.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_enrollment_details",
            "description": (
                "Get full details of a specific enrollment record, including "
                "which student is enrolled in which course, taught by which "
                "professor, plus marks/grades/attendance. Use this for "
                "questions like 'which professor taught student X' or "
                "'what course is student X taking'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "enrollment_id": {"type": "integer", "description": "The enrollment record's ID number"}
                },
                "required": ["enrollment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_students_by_course",
            "description": (
                "Get the list of students enrolled in a specific course. "
                "Accepts EITHER the course's numeric ID (e.g. '1') OR its "
                "course code (e.g. 'CS138'). Use this for questions like "
                "'who is enrolled in course X' or 'which students are "
                "taking this course'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "course_identifier": {
                        "type": "string",
                        "description": "The course's ID number OR its course code (e.g. '1' or 'CS138')"
                    }
                },
                "required": ["course_identifier"],
            },
        },
    },
]

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
 
def run_chat_tool(function_name: str, arguments: dict, db: Session):
    if function_name == "get_student_by_id":
        return chat_get_student_by_id(arguments["student_id"], db)
    elif function_name == "get_all_students":
        return chat_get_all_students(db)
    elif function_name == "count_students":
        return chat_count_students(db)
    elif function_name == "get_professor_by_id":
        return chat_get_professor_by_id(arguments["professor_id"], db)
    elif function_name == "get_all_professors":
        return chat_get_all_professors(db)
    elif function_name == "get_course_by_id":
        return chat_get_course_by_id(arguments["course_id"], db)
    elif function_name == "get_all_courses":
        return chat_get_all_courses(db)
    elif function_name == "get_enrollment_details":
        return chat_get_enrollment_details(arguments["enrollment_id"], db)
    elif function_name == "get_students_by_course":
        return chat_get_students_by_course(arguments["course_identifier"], db)
    else:
        return {"error": "Unknown tool"}
 
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
 
 
class ChatResponse(BaseModel):
    reply: str
    history: List[dict]
 
 
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    for m in request.history:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": request.message})
 
    response = openrouter_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=chat_tools,
        tool_choice="required",
    )
    reply = response.choices[0].message
    print("TOOL CALLS:", reply.tool_calls)
 
    if reply.tool_calls:
        messages.append(reply.model_dump())
 
        for tool_call in reply.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            result = run_chat_tool(function_name, arguments, db)
            print("TOOL RESULT:", result)
 
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })
 
        final_response = openrouter_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
        )
        answer = final_response.choices[0].message.content
    else:
        answer = reply.content
 
    returned_history = []
    for m in request.history:
        returned_history.append({"role": m.role, "content": m.content})
    returned_history.append({"role": "user", "content": request.message})
    returned_history.append({"role": "assistant", "content": answer})
 
    return ChatResponse(reply=answer, history=returned_history)
 