from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

# STUDENTS TABLE
class Student(Base):
    __tablename__ = "students"

    student_id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    email = Column(String, unique=True)
    date_of_birth = Column(Date)
    gender = Column(String)
    stream = Column(String)
    year_of_enrollment = Column(Integer)
    current_year = Column(Integer)
    current_semester = Column(Integer)
    address = Column(String)
    status = Column(String, default="active")

    enrollments = relationship("Enrollment", back_populates="student")


# PROFESSORS TABLE
class Professor(Base):
    __tablename__ = "professors"

    professor_id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    email = Column(String, unique=True)
    department = Column(String)
    years_of_experience = Column(Integer)
    specialization = Column(String)

    offerings = relationship("CourseOffering", back_populates="professor")


# COURSES TABLE
class Course(Base):
    __tablename__ = "courses"

    course_id = Column(Integer, primary_key=True, index=True)
    course_name = Column(String, nullable=False)
    course_code = Column(String, unique=True)
    credits = Column(Integer)
    department = Column(String)
    course_type = Column(String)
    description = Column(String)

    offerings = relationship("CourseOffering", back_populates="course")


# COURSE OFFERING TABLE
class CourseOffering(Base):
    __tablename__ = "course_offering"

    offering_id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.course_id"))
    professor_id = Column(Integer, ForeignKey("professors.professor_id"))
    semester = Column(Integer)
    academic_year = Column(String)
    status = Column(String, default="ongoing")

    course = relationship("Course", back_populates="offerings")
    professor = relationship("Professor", back_populates="offerings")
    enrollments = relationship("Enrollment", back_populates="offering")


class Enrollment(Base):
    __tablename__ = "enrollment"

    enrollment_id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.student_id"))
    offering_id = Column(Integer, ForeignKey("course_offering.offering_id"))
    attendance = Column(Integer)
    internal_marks = Column(Integer)
    exam_marks = Column(Integer)
    total_marks = Column(Integer)
    grade = Column(String)
    grade_points = Column(Float)
    status = Column(String, default="ongoing")
    enrollment_date = Column(Date)

    student = relationship("Student", back_populates="enrollments")
    offering = relationship("CourseOffering", back_populates="enrollments")