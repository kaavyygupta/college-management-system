from database import engine, Base
import models

# This creates all tables
Base.metadata.create_all(bind=engine)

print("All tables created successfully!")