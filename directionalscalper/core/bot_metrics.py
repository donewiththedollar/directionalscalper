import sqlite3
import pytz
from datetime import datetime
from directionalscalper.core.strategies.logger import Logger

logging = Logger(logger_name="BotMetrics", filename="BotMetrics.log", stream=True)

class BotDatabase:
    def __init__(self, db_file="bot_data.db", exchange=None):
        self.db_file = db_file
        self.exchange = exchange

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
                if start_date_str:  # Ensure start_date_str is not None
                    start_date = datetime.strptime(start_date_str, '%Y-%m-%dT%H:%M:%S.%f')
                    return initial_equity, start_date
            else:
                # Insert initial values if not found
                initial_equity = 0.0  # Set your default initial equity
                start_date = datetime.now()
                self.save_initial_values(initial_equity, start_date)
                return initial_equity, start_date


    # def get_initial_values(self):
    #     with self.get_connection() as conn:
    #         cursor = conn.cursor()
            
    #         cursor.execute('SELECT initial_equity, start_date FROM initial_values WHERE id = 1')
    #         row = cursor.fetchone()
            
    #         if row:
    #             initial_equity, start_date_str = row
    #             if start_date_str:  # Ensure start_date_str is not None
    #                 start_date = datetime.strptime(start_date_str, '%Y-%m-%dT%H:%M:%S.%f')
    #                 return initial_equity, start_date
    #         return None, None


    def get_average_daily_gain(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT avg_daily_gain FROM daily_gains WHERE id = 1')  # Assuming you saved it with an id of 1
            row = cursor.fetchone()
            
            if row:
                return row[0]
            return 0.0

    def compute_average_daily_gain(self):

        quote_currency = "USDT"
        
        # Retrieve stored values
        initial_values = self.get_initial_values()
        if initial_values:
            initial_equity, start_date_str = initial_values
            if start_date_str:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%dT%H:%M:%S.%f')
            else:
                logging.error("Start date string from database is None.")
                return 0
        else:
            logging.warning("Initial values not found in database. Inserting new values.")
            initial_equity = self.exchange.get_balance_bybit(quote_currency)
            start_date = datetime.now()
            self.save_initial_values(initial_equity, start_date)

        # Safety check
        if not isinstance(start_date, datetime):
            logging.error("Start date is not a valid datetime object. Defaulting to current datetime.")
            start_date = datetime.now()

        # Compute average daily gain percentage
        current_equity = self.exchange.get_balance_bybit(quote_currency)
        days_passed = (datetime.now() - start_date).days
        avg_daily_gain = self.compute_average_daily_gain_percentage(initial_equity, current_equity, days_passed)

        # Check if current time is past 8 PM EST and reset once a day
        est_now = datetime.now(pytz.timezone('US/Eastern'))
        if est_now.hour == 20:
            initial_equity = current_equity
            start_date = datetime.now()
            self.save_initial_values(initial_equity, start_date)

        return avg_daily_gain

    @staticmethod
    def compute_average_daily_gain_percentage(initial_equity, current_equity, days_passed):
        if days_passed == 0:
            return 0
        return ((current_equity - initial_equity) / (initial_equity * days_passed)) * 100