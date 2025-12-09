import pandas as pd
import numpy as np

def preprocessing():
    course_data = pd.read_csv("courses.csv")
    course_data = course_data.dropna(how="any")
    
    course_data["helper"] = course_data["Course"].str.extract(r"MAT-(\d{3})")
    course_data["helper"] = course_data["helper"].astype(int)
    course_data["Cap"] = course_data["Cap"].astype(int)
    course_data = course_data[(course_data["helper"] >= 100) & (course_data["helper"] < 200)]
    course_data = course_data.drop(columns=["helper"])
    course_data = course_data.reset_index(drop=True)
    course_data.index = course_data.index + 1
    course_data.to_csv("preprocessed_data/courses.csv")
    
    rooms_data = pd.read_csv("rooms.csv")
    rooms_data = rooms_data[rooms_data["Classroom List"].str.startswith("Wellman Hall")]
    rooms_data = rooms_data.reset_index(drop=True)
    rooms_data.index = rooms_data.index + 1
    rooms_data.to_csv("preprocessed_data/rooms.csv")

    courses = course_data.apply(lambda row: tuple(row), axis= 1).to_numpy()
    rooms = rooms_data.apply(lambda row: tuple(row), axis=1).to_numpy()

    return courses, rooms