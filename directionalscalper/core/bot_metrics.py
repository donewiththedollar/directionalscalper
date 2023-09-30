import sqlite3
from datetime import datetime

class BotDatabase:
    def __init__(self, db_file="bot_data.db"):
        self.db_file = db_file

    def get_connection(self):
        return sqlite3.connect(self.db_file)

    def create_tables_if_not_exists(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS initial_values (
                id INTEGER PRIMARY KEY,
                initial_equity REAL,
                start_date TEXT
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_gains (
                id INTEGER PRIMARY KEY,
                avg_daily_gain REAL
            )
            ''')

            conn.commit()

    def save_initial_values(self, initial_equity, start_date):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM initial_values WHERE id = 1')  # Clear existing values
            cursor.execute('INSERT INTO initial_values (id, initial_equity, start_date) VALUES (?, ?, ?)', 
                           (1, initial_equity, start_date.isoformat()))
            
            conn.commit()

    def get_initial_values(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT initial_equity, start_date FROM initial_values WHERE id = 1')
            row = cursor.fetchone()
            
            if row:
                initial_equity, start_date_str = row
                start_date = datetime.strptime(start_date_str, '%Y-%m-%dT%H:%M:%S.%f')
                return initial_equity, start_date
            return None, None

    def get_average_daily_gain(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT avg_daily_gain FROM daily_gains WHERE id = 1')  # Assuming you saved it with an id of 1
            row = cursor.fetchone()
            
            if row:
                return row[0]
            return 0.0
