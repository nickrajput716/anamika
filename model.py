import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import datetime

class StudyPlannerModel:
    def __init__(self):
        self.rf_reg = None
        self.rf_clf = None
        self.X_columns = None
        self.load_and_train_model()
    
    def load_and_train_model(self):
        """Load dataset and train models"""
        df = pd.read_csv('StudentsPerformance.csv')
        
        # Data preprocessing
        df['gender'] = df['gender'].map({'female': 0, 'male': 1})
        df['race/ethnicity'] = df['race/ethnicity'].map({
            'group A': 0, 'group B': 1, 'group C': 2, 'group D': 3, 'group E': 4
        })
        df['parental level of education'] = df['parental level of education'].map({
            "some high school": 0, "high school": 1, "some college": 2,
            "associate's degree": 3, "bachelor's degree": 4, "master's degree": 5
        })
        df['lunch'] = df['lunch'].map({'standard': 0, 'free/reduced': 1})
        df['test preparation course'] = df['test preparation course'].map({'none': 0, 'completed': 1})
        
        df['average_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3
        df['performance'] = pd.cut(df['average_score'], bins=[0, 60, 80, 100], labels=['Low', 'Medium', 'High'])
        
        X = df[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']]
        self.X_columns = X.columns
        y_reg = df['average_score']
        y_clf = df['performance']
        
        # Train models
        self.rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_reg.fit(X, y_reg)
        self.rf_clf.fit(X, y_clf)
    
    def predict(self, gender, race, parent_edu, lunch, test_prep):
        """Predict score and performance level"""
        input_data = pd.DataFrame([[gender, race, parent_edu, lunch, test_prep]], 
                                 columns=self.X_columns)
        score = self.rf_reg.predict(input_data)[0]
        level = self.rf_clf.predict(input_data)[0]
        return score, level
    
    def generate_timetable(self, score):
        """Generate study timetable based on score"""
        if score < 60:
            math_h, reading_h, writing_h = 3.0, 2.0, 2.0
            total_h = 7.0
        elif score < 80:
            math_h, reading_h, writing_h = 2.0, 1.5, 1.5
            total_h = 5.0
        else:
            math_h, reading_h, writing_h = 1.5, 1.0, 1.0
            total_h = 3.5
        
        timetable = [
            {"time": "06:00 AM", "activity": "Wake Up + Morning Exercise (30 min)"},
            {"time": "06:30 AM", "activity": "Breakfast"},
            {"time": "07:00 AM", "activity": f"MATH STUDY ({math_h} hours)"},
            {"time": "09:00 AM", "activity": "BREAK (10 min)"},
            {"time": "09:10 AM", "activity": f"READING ({reading_h} hours)"},
            {"time": "11:00 AM", "activity": "LUNCH + REST"},
            {"time": "01:30 PM", "activity": f"WRITING ({writing_h} hours)"},
            {"time": "03:30 PM", "activity": "POMODORO BREAK (15 min)"},
            {"time": "03:45 PM", "activity": "REVISION + WEAK TOPICS (1 hour)"},
            {"time": "04:45 PM", "activity": "OUTDOOR ACTIVITY"},
            {"time": "06:00 PM", "activity": "MOCK TEST / PAST PAPERS (1 hour)"},
            {"time": "08:00 PM", "activity": "DINNER"},
            {"time": "09:00 PM", "activity": "LIGHT REVISION (30 min)"},
            {"time": "09:30 PM", "activity": "SLEEP (8 hours recommended)"}
        ]
        
        return {
            'timetable': timetable,
            'math_hours': math_h,
            'reading_hours': reading_h,
            'writing_hours': writing_h,
            'total_hours': total_h
        }
    
    def save_progress(self, math_h, reading_h, writing_h):
        """Save daily progress"""
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        total = math_h + reading_h + writing_h
        target = 5.0
        progress = min(100, (total / target) * 100)
        
        new_entry = pd.DataFrame([{
            "Date": today,
            "Math_Hours": math_h,
            "Reading_Hours": reading_h,
            "Writing_Hours": writing_h,
            "Total_Hours": total,
            "Progress_%": progress
        }])
        
        if os.path.exists("progress_log.csv"):
            log_df = pd.read_csv("progress_log.csv")
            log_df = pd.concat([log_df, new_entry], ignore_index=True)
        else:
            log_df = new_entry
        
        log_df.to_csv("progress_log.csv", index=False)
        return progress
    
    def get_progress_data(self):
        """Get progress data for charts"""
        if os.path.exists("progress_log.csv"):
            df = pd.read_csv("progress_log.csv")
            return df.to_dict('records')
        return []
