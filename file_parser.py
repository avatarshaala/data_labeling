import uuid
from pygments import highlight
from pygments.lexers import JavaLexer
from pygments.formatters import HtmlFormatter

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
                annotated_code += f'<input type="checkbox" id="{checkbox_ids[0]}" name="{checkbox_ids[0]}">'  # First checkbox
                annotated_code += f'<input type="checkbox" id="{checkbox_ids[1]}" name="{checkbox_ids[1]}">'  # Second checkbox
                annotated_code += f'<input type="checkbox" id="{checkbox_ids[2]}" name="{checkbox_ids[2]}">'  # Third checkbox
                annotated_code += f'<label for="{idx}">{line.strip()}</label><br>\n'  # Label
        else:
            annotated_code += f"{line}\n"
    return annotated_code


def is_header_line(line):
    header_keywords = ["* author:", "* topics:", "* subtopics:", "* goalDescription:", "* source:", "* output:"]
    return any(line.strip().startswith(keyword) for keyword in header_keywords)


def parse(java_code):
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
    annotated_body_code = annotate_java_code(body_code)

    return header_code, annotated_body_code
