import streamlit as st
import sqlite3
import uuid
from pygments import highlight
from pygments.lexers import JavaLexer
from pygments.formatters import HtmlFormatter


def create_db():
    conn = sqlite3.connect("annotation_db.db")
    c = conn.cursor()

    # Create program table
    c.execute('''CREATE TABLE IF NOT EXISTS program (
                 prog_id INTEGER PRIMARY KEY,
                 prog_file TEXT,
                 prog_code TEXT)''')

    # Create annotations table
    c.execute('''CREATE TABLE IF NOT EXISTS annotations (
                 annotation_id INTEGER PRIMARY KEY,
                 prog_id INTEGER,
                 line_number INTEGER,
                 statement TEXT,
                 relevancy INTEGER,
                 quality INTEGER,
                 sufficiency INTEGER,
                 FOREIGN KEY(prog_id) REFERENCES program(prog_id))''')

    conn.commit()
    conn.close()


def highlight_java_code(java_code):
    return highlight(java_code, JavaLexer(), HtmlFormatter())


def annotate_java_code(java_code):
    annotated_code = ""
    inside_comment = False
    for idx, line in enumerate(java_code.split("\n")):
        if line.strip().startswith("/**"):
            inside_comment = True
            annotated_code += "\n/**\n\n"
        elif inside_comment and line.strip().startswith("*/"):
            inside_comment = False
            annotated_code += "\n*/\n\n"
        elif inside_comment and line.strip().startswith("*") and not is_header_line(line):
            if "/*" in line or "*/" in line:
                # Skip block comment delimiters
                annotated_code += f"{line}\n"
            else:
                # Generate unique identifiers for checkboxes
                checkbox_ids = [str(uuid.uuid4()) for _ in range(3)]  # Generate unique identifiers
                # Add checkboxes and label for each comment line
                annotated_code += f'<input type="checkbox" id="{checkbox_ids[0]}" name="{checkbox_ids[0]}"> '  # First checkbox
                annotated_code += f'<input type="checkbox"> id="{checkbox_ids[1]}" name="{checkbox_ids[1]}"'  # Second checkbox
                annotated_code += f'<input type="checkbox"> id="{checkbox_ids[2]}" name="{checkbox_ids[2]}"'  # Third checkbox
                annotated_code += f'<label for="{idx}">{line.strip()}</label><br>\n'  # Label
        else:
            annotated_code += f"{line}\n"
    return annotated_code


def is_header_line(line):
    header_keywords = ["* author:", "* topics:", "* subtopics:", "* goalDescription:", "* source:", "* output:"]
    return any(line.strip().startswith(keyword) for keyword in header_keywords)


def save_annotation(prog_id, line_number, statement, relevancy, quality, sufficiency):
    conn = sqlite3.connect("annotation_db.db")
    c = conn.cursor()
    c.execute('''INSERT INTO annotations (prog_id, line_number, statement, relevancy, quality, sufficiency)
                 VALUES (?, ?, ?, ?, ?, ?)''', (prog_id, line_number, statement, relevancy, quality, sufficiency))
    conn.commit()
    conn.close()


def main():
    st.title("Java Code Annotator")

    # Create SQLite database
    create_db()

    # Move file uploader to sidebar
    uploaded_file = st.sidebar.file_uploader("Upload Java code file", type=["java"])

    if uploaded_file is not None:
        # Read uploaded Java code
        java_code = uploaded_file.getvalue().decode("utf-8")

        # Split code into header and body
        header_lines = []
        body_lines = []
        is_header = True
        for line in java_code.split("\n"):
            if line.strip().startswith("/**"):
                is_header = False
            if is_header:
                header_lines.append(line)
            else:
                body_lines.append(line)
        header_code = "\n".join(header_lines)
        body_code = "\n".join(body_lines)

        # Save program code to database
        conn = sqlite3.connect("annotation_db.db")
        c = conn.cursor()
        c.execute("INSERT INTO program (prog_file, prog_code) VALUES (?, ?)", (uploaded_file.name, body_code))
        prog_id = c.lastrowid
        conn.commit()
        conn.close()

        # Annotate comments in body
        st.subheader("Annotated Java Code")
        with st.expander("Show Annotated Code"):
            annotated_body_code = annotate_java_code(body_code)
            st.write(header_code + "\n" + annotated_body_code, unsafe_allow_html=True)

        # Save Annotation button
        if st.sidebar.button("Save Annotation"):
            # Placeholder for saving annotation logic
            pass


if __name__ == "__main__":
    main()
