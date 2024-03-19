import re
import json
import os
class JavaToJSON:

    @staticmethod
    def __is_header_line(line):
        header_keywords = ["* author:", "* topics:", "* subtopics:", "* goalDescription:", "* source:", "* output:"]
        return any(line.strip().startswith(keyword) for keyword in header_keywords)

    @staticmethod
    def __get_line_types(code_lines):

        line_types = []
        is_header = True

        for i, code_line in enumerate(code_lines):
            type = ""
            if is_header:
                type = "header_comment"
                if code_line.strip().startswith("/*"):
                    type = "begin_header"
                elif code_line.strip().startswith("*/"):
                    type = "end_header"
                elif code_line.strip() == "":
                    type = ""
                if re.match(r"\s*\*/", code_line):
                    is_header = False
            else:
                type = "code_body"
                if code_line.strip().startswith("/*"):
                    type = "begin_comment"
                elif code_line.strip().startswith("*/"):
                    type = "end_comment"
                elif code_line.strip().startswith("*"):
                    type = "comment"
                elif code_line.strip() == "":
                    type = ""

            line_types.append(type)

        return line_types

    @staticmethod
    def __convert_to_json(java_code, source_file, target_file, user, date, labels, other_info=""):
        '''

        the JSON format looks as follows
        {
        original_file_name:""
        json_file_name:""
        user:"admin"
        date:"05-03-2024"
        other_info:"annotated, more info could be added"
        lines:{
            "1":{
                content:""
                type:""
                labels:{
                    "sufficient":0
                    }
                }
            "2":{
                content:""
                type:""
                labels:{
                    "relevant":0,
                    "quality":1
                    }
                }
            }
        }
        '''

        # json_file = f"{target_dir}/{target_file}"
        # if os.path.exists(json_file):
        #     with open(json_file, 'r') as file:
        #         data = json.load(file)
        # else:
        #     data = {}


        #get code lines types
        code_lines = java_code.split("\n")
        line_types = JavaToJSON.__get_line_types(code_lines)

        data = {}
        # start building the dictionary for JSON's initial header
        data['original_file_name'] = source_file
        data['json_file_name'] = target_file
        data['user'] = user
        data['date'] = date
        data['other_info'] = other_info

        # lines of code starts from  here
        data['lines'] = {}

        for i, (code_line, type) in enumerate(zip(code_lines, line_types), start=1):

            data['lines'][f'{i}'] = {'content':code_line, 'type':type}

            if type == 'begin_comment' or type == 'comment':
                data['lines'][f'{i}']['labels'] = {}
                labs = labels['comment_label'] if type == 'comment' else labels['block_label']
                for criteria, label in labs.items():
                    data['lines'][f'{i}']['labels'][criteria] = label


        return data

    @staticmethod
    def parse_as_json(java_code, source_file, target_file, user, date, labels, other_info=""):

        data = JavaToJSON.__convert_to_json(
            java_code,
            source_file,
            target_file,
            user,
            date,
            labels,
            other_info=""
        )

        return data

    @staticmethod
    def save_java_to_json(java_code, source_file, target_file, target_dir, user, date, labels, other_info=""):

        data = JavaToJSON.__convert_to_json(
            java_code,
            source_file,
            target_file,
            user, date,
            labels,
            other_info=""
        )

        # Write the updated data to the file
        with open(f'{target_dir}/{target_file}', 'w') as file:
            json.dump(data, file, indent=4)

    @staticmethod
    def read_json(source_file, source_dir):

        json_file = f"{source_dir}/{source_file}"
        if os.path.exists(json_file):
            with open(json_file, 'r') as file:
                data = json.load(file)
        else:
            raise FileNotFoundError("File not found")

        return data
        # return json.dumps(data)

    @staticmethod
    def list_json_files(json_dir, user=None, date=None):

        json_files = []
        users = []
        dates = []
        for filename in os.listdir(json_dir):
            if filename.endswith(".json"):
                # Split filename into user, date, and target_file
                parts = filename.split(".")
                if len(parts) >= 3:
                    file_user, file_date, _,_ = parts
                    # Check if the file meets the filtering conditions
                    if (user is None or user == file_user) and (date is None or date == file_date):
                        json_files.append(filename)
                        users.append(file_user)
                        dates.append(file_date)

        return json_files, list(set(users)), list(set(dates))

    @staticmethod
    def delete_json_file(json_file, json_dir):
        pass

    @staticmethod
    def save_data_to_json(data, json_file, json_dir, user, date, create_new=None):
        # Write the updated data to the file
        with open(f'{json_dir}/{json_file}', 'w') as file:
            json.dump(data, file, indent=4)
