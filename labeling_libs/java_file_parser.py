import re
from annotation_to_json import JSONConverter

class JavaFileParser:

    @staticmethod
    def save_to_json(java_code, filename, date, annotations=None, user=None):

        if annotations is None:
            annotations = {"relevant": 0, "quality": 0}
        if user is None:
            user = "default"
        header_code, body_code = JavaFileParser.get_header_and_body(java_code)

        # header_code = header_code.replace("\r","\n")
        # body_code = body_code.replace("\r", "\n")

        header_code = header_code.split("\n")
        # Add two extra newlines
        header_code.append("\n")
        header_code.append("\n")

        line_num = 1

        # Save header to JSON line by line
        for header_line in header_code:
            type = "header_comment"
            if header_line.strip().startswith("/*"):
                type= "begin_header"
            elif header_line.strip().startswith("*/"):
                type = "end_header"
            elif header_line.strip()=="":
                type = ""
            JSONConverter.save_to_json(
                filename,
                line_num,
                content=header_line,
                type=type,
                user=user,
                date=date,
                annotations=annotations
            )

            line_num += 1

        body_code = body_code.split("\n")
        # Save body to JSON line by line
        for body_line in body_code:
            type = "body_comment"
            if body_line.strip().startswith("/*"):
                type = "begin_comment"
            elif body_line.strip().startswith("*/"):
                type = "end_comment"
            elif body_line.strip().startswith("*") :
                type = "comment"
            elif body_line.strip() == "":
                type = ""

            JSONConverter.save_to_json(
                filename,
                line_num,
                content=body_line,
                type=type,
                user=user,
                date=date,
                annotations=annotations
            )

            line_num += 1


    @staticmethod
    def get_header_and_body(java_code):
        header_code = ""
        body_code = ""
        is_header = True

        for line in java_code.split("\n"):
            if is_header:
                header_code += line + "\n"
                if re.match(r"\s*\*/", line):
                    is_header = False
            else:
                body_code += line + "\n"

        return header_code.strip(), body_code.strip()

    @staticmethod
    def remove_comments(java_code):
        return re.sub(r'/\*.*?\*/', '', java_code, flags=re.DOTALL)

    @staticmethod
    def is_header_line(line):
        header_keywords = ["* author:", "* topics:", "* subtopics:", "* goalDescription:", "* source:", "* output:"]
        return any(line.strip().startswith(keyword) for keyword in header_keywords)

#
#
# if __name__ == "__main__":
#     # Read the content of the "ai_car.java" file
#     with open("../experiments/15/ai_CityStats.java", "r") as file:
#         java_code = file.read()
#
#     # Parse the Java file
#
#     header_code, body_code = JavaFileParser.get_header_and_body(java_code)
#     clean_code = JavaFileParser.remove_comments(java_code)
#     print("Header Code:")
#     print(header_code)
#     print("\nBody Code:")
#     print(body_code)
#     print("\nClean Code:")
#     print(clean_code)
