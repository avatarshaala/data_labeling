import sqlite3

class LabelDB:
    def __init__(self):
        self.conn = sqlite3.connect("annotation_db.db")
        self.c = self.conn.cursor()
        self.create_db()

    def create_db(self):
        # Create program table
        self.c.execute('''CREATE TABLE IF NOT EXISTS program (
                         prog_id INTEGER PRIMARY KEY AUTOINCREMENT,
                         prog_file TEXT,
                         prog_code TEXT)''')

        # Create annotations table
        self.c.execute('''CREATE TABLE IF NOT EXISTS annotations (
                         annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                         prog_id INTEGER,
                         line_number INTEGER,
                         annotated_text TEXT,
                         relevancy INTEGER,
                         quality INTEGER,
                         sufficiency INTEGER,
                         date TEXT,
                         FOREIGN KEY(prog_id) REFERENCES program(prog_id))''')

        self.conn.commit()

    def save_annotation(self, prog_id, line_number, annotated_text, relevancy, quality, sufficiency, date):
        self.c.execute('''INSERT INTO annotations (prog_id, line_number, annotated_text, relevancy, quality, sufficiency, date)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''', (prog_id, line_number, annotated_text, relevancy, quality, sufficiency, date))
        self.conn.commit()

    def edit_annotation(self, annotation_id, prog_id, line_number, annotated_text, relevancy, quality, sufficiency, date):
        self.c.execute('''UPDATE annotations SET prog_id=?, line_number=?, annotated_text=?, relevancy=?, quality=?, sufficiency=?, date=?
                          WHERE annotation_id=?''', (prog_id, line_number, annotated_text, relevancy, quality, sufficiency, date, annotation_id))
        self.conn.commit()

    def insert_program(self, prog_file, prog_code):
        self.c.execute("INSERT INTO program (prog_file, prog_code) VALUES (?, ?)", (prog_file, prog_code))
        prog_id = self.c.lastrowid
        self.conn.commit()
        return prog_id

    def __del__(self):
        self.conn.close()
