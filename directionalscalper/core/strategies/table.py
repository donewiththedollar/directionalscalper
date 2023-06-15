from rich.table import Table

class BaseTable:
    def __init__(self):
        self.table = Table(show_header=False, box=None, title="DirectionalScalper Multi")
        self.table.add_column("Condition", justify="center")
        self.table.add_column("Config", justify="center")
        self.table.add_column("Current", justify="center")
        self.table.add_column("Status")

    def add_row(self, condition, config, current, status):
        self.table.add_row(condition, str(config), str(current), status)
