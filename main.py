from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
from datetime import date
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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
