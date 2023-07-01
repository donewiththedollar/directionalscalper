# from rich.table import Table

# class BaseTable:
#     def __init__(self):
#         self.table = Table(show_header=True, header_style="bold magenta")
#         self.table.add_column("Short Take Profit", style="dim", width=20)
#         self.table.add_column("Long Take Profit", style="dim", width=20)

#     def add_row(self, short_take_profit, long_take_profit):
#         self.table.add_row(str(short_take_profit), str(long_take_profit))
# from rich.table import Table

# class BaseTable:
#     def __init__(self):
#         self.table = Table(show_header=True, header_style="bold magenta", title="DirectionalScalper Multi")
#         self.table.add_column("Short Take Profit", style="dim", width=20)
#         self.table.add_column("Long Take Profit", style="dim", width=20)

#     def add_row(self, short_take_profit, long_take_profit):
#         self.table.add_row(str(short_take_profit), str(long_take_profit))

# from rich.table import Table

# class BaseTable:
#     def __init__(self):
#         self.table = Table(show_header=True, header_style="bold magenta", title="DirectionalScalper Multi")
#         self.table.add_column("Info", style="dim", width=20)
#         self.table.add_column("Values", style="dim", width=20)

#     def add_row(self, info, value):
#         self.table.add_row(str(info), str(value))

# base_table = BaseTable()

# base_table.add_row("Short Take Profit", short_take_profit)
# base_table.add_row("Long Take Profit", long_take_profit)

# from rich.table import Table

# class BaseTable:
#     def __init__(self):
#         self.table = Table(show_header=True, header_style="bold magenta", title="DirectionalScalper Multi")
#         self.table.add_row("Short Take Profit")
#         self.table.add_row("Long Take Profit")

# class BaseTable:
#     def __init__(self):
#         self.table = Table(show_header=False, box=None, title="DirectionalScalper Multi")
#         self.table.add_column("Condition", justify="center")
#         self.table.add_column("Config", justify="center")
#         self.table.add_column("Current", justify="center")
#         self.table.add_column("Status")

#     def add_row(self, condition, config, current, status):
#         self.table.add_row(condition, str(config), str(current), status)

# import threading
# from rich.table import Table

# class BaseTable:
#     def __init__(self):
#         self.table = Table(show_header=False, box=None, title="DirectionalScalper Multi")
#         self.table.add_column("Condition", justify="center")
#         self.table.add_column("Value", justify="center")
#         self.lock = threading.Lock()  # create a lock object

#     def add_row(self, condition, value):
#         with self.lock:  # acquire the lock
#             self.table.add_row(condition, str(value))  # convert value to string in case it's not already

from rich.table import Table

class BaseTable:
    def __init__(self):
        self.table = Table(show_header=False, box=None, title="DirectionalScalper Multi")
        self.table.add_column("Condition", justify="center")
        self.table.add_column("Value", justify="center")

    def add_row(self, condition, value):
        self.table.add_row(condition, str(value))  # convert value to string in case it's not already
