import json
from datetime import datetime, timedelta
import random
import os

class JSONConverter:
    @staticmethod
    def save_to_json(filename, line_number, content, type, user, date, annotations):

        line_number = f'{line_number}'
        date = date.strftime("%d-%m-%Y")
        # Check if the file exists, and load data if it does
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                data = json.load(file)
        else:
            data = {}

        # Check if the line number is already in the data, if not, create a new entry
        if line_number not in data:
            data[line_number] = {"content": content, "type": type, "annotation": {}}

        # Check if the user is already in the annotations, if not, create a new entry
        if user not in data[line_number]["annotation"]:
            data[line_number]["annotation"][user] = {}

        # Check if the annotations for the given user and date already exist, if not, create a new entry
        if date not in data[line_number]["annotation"][user]:
            data[line_number]["annotation"][user][date] = annotations
        else:
            # Append the annotations to the existing ones
            existing_annotations = data[line_number]["annotation"][user][date]
            existing_annotations.update(annotations)
            data[line_number]["annotation"][user][date] = existing_annotations

        # Write the updated data back to the file
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)

        # Return the updated JSON data
        return json.dumps(data)

    @staticmethod
    def update_json(filename, line_number, user, date, annotations):

        line_number = f'{line_number}'
        date = date.strftime("%d-%m-%Y")
        # Check if the file exists, and load data if it does
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                data = json.load(file)
        else:
            raise FileNotFoundError("File not found")

        # Check if the line number is already in the data, if not, raise an error
        if line_number not in data:
            raise KeyError(f"Line number {line_number} not found")

        # Check if the user is already in the annotations, if not, create a new entry
        if user not in data[line_number]["annotation"]:
            data[line_number]["annotation"][user] = {}

        # Check if the annotations for the given user and date already exist, if not, create a new entry
        if date not in data[line_number]["annotation"][user]:
            data[line_number]["annotation"][user][date] = annotations
        else:
            # Append the annotations to the existing ones
            existing_annotations = data[line_number]["annotation"][user][date]
            existing_annotations.update(annotations)
            data[line_number]["annotation"][user][date] = existing_annotations

        # Write the updated data back to the file
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)

        # Return the updated JSON data
        return json.dumps(data)

    @staticmethod
    def delete_annotation(filename, user, date=None):
        # Check if the file exists, and load data if it does
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                data = json.load(file)
        else:
            raise FileNotFoundError("File not found")

        # Iterate over each line number and its data
        for line_number, line_data in data.items():
            # Check if the user exists in the annotations for this line
            if user in line_data.get("annotation", {}):
                # Check if date is provided
                if date:
                    # Check if the date exists in the user's annotations
                    if date in line_data["annotation"][user]:
                        # Delete the annotation for the specified date
                        del line_data["annotation"][user][date]
                else:
                    # Delete all annotations for the user
                    del line_data["annotation"][user]

        # Write the updated data back to the file
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)

        # Return the updated JSON data
        return json.dumps(data)

    @staticmethod
    def get_recent_annotation(filename, user):
        # Check if the file exists, and load data if it does
        # print("FileNAME",filename)
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                data = json.load(file)
        else:
            raise FileNotFoundError("File not found")

        recent_annotations = {}
        recent_dates = set()

        # Iterate over each line number and its data
        for line_number, line_data in sorted(data.items(), key=lambda x: int(x[0])):
            if "annotation" in line_data and user in line_data["annotation"]:
                # Get all annotations for the user for this line
                user_annotations = line_data["annotation"][user]
                # Find the most recent annotation date
                most_recent_date = max(user_annotations.keys())
                # accumulate most recent dates
                recent_dates.add(most_recent_date)
                # Get the most recent annotation for the user
                most_recent_annotation = user_annotations[most_recent_date]
                # Store the most recent annotation for the line number
                recent_annotations[line_number] = {
                    "content": line_data["content"],
                    "type": line_data["type"],
                    "annotation": {
                        user: {
                            most_recent_date: most_recent_annotation
                        }
                    }
                }

        # return json.dumps(recent_annotations, indent=4)
        return list(recent_dates), recent_annotations


    @staticmethod
    def read_json(filename, user=None, date=None):

        # Check if the file exists, and load data if it does
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                data = json.load(file)
        else:
            raise FileNotFoundError("File not found")

        # Initialize an empty result dictionary to store filtered data
        result = {}
        # Iterate over each line number and its data
        for line_number, line_data in data.items():
            # Check filtering conditions for user and date
            if user is None and date is None:
                result[line_number] = line_data
            elif user is not None and date is None:
                result[line_number] = {}
                result[line_number]["content"] = line_data["content"]
                result[line_number]["type"] = line_data["type"]
                result[line_number]["annotation"] = {}
                if user in line_data["annotation"]:
                    result[line_number]["annotation"][user] = line_data["annotation"][user]
            elif user is None and date is not None:
                result[line_number] = {}
                result[line_number]["content"] = line_data["content"]
                result[line_number]["type"] = line_data["type"]
                result[line_number]["annotation"] = {}
                for user, user_data in line_data["annotation"].items():
                    result[line_number]["annotation"][user] ={}
                    if date.strftime("%d-%m-%Y") in user_data:
                        result[line_number]["annotation"][user][date.strftime("%d-%m-%Y")] = line_data["annotation"][user][date.strftime("%d-%m-%Y")]
            else:
                result[line_number] = {}
                result[line_number]["content"] = line_data["content"]
                result[line_number]["type"] = line_data["type"]
                result[line_number]["annotation"] = {}
                if user in line_data["annotation"]:
                    result[line_number]["annotation"][user] = {}
                    if date.strftime("%d-%m-%Y") in line_data['annotation'][user]:
                        result[line_number]["annotation"][user][date.strftime("%d-%m-%Y")] = line_data["annotation"][user][date.strftime("%d-%m-%Y")]


        # Return the filtered JSON data
        return json.dumps(result)


# if __name__ == "__main__":
#     filename = "example_annotations.json"
#
#     lines_data = [
#         {"line_number": 1, "content": "Content of line 1", "type": "Type 1"},
#         {"line_number": 2, "content": "Content of line 2", "type": "Type 2"},
#         {"line_number": 3, "content": "Content of line 3", "type": "Type 3"},
#         {"line_number": 4, "content": "Content of line 4", "type": "Type 4"}
#     ]
#
#     users = ["user1", "user2"]
#     dates = [datetime.now(), datetime.now() - timedelta(days=1)]
#
#     for line_data in lines_data:
#         for user in users:
#             for date in dates:
#                 annotations = {"relevant": random.randint(0, 1), "quality": random.randint(0, 1)}
#                 JSONConverter.save_to_json(filename, line_data["line_number"], line_data["content"], line_data["type"], user, date, annotations)
#
#     print("Annotations for all users and dates:")
#     print(JSONConverter.read_json(filename))
#
#     print("\nAnnotations for user1:")
#     print(JSONConverter.read_json(filename, user="user1"))
#
#     print("\nAnnotations for a specific date:")
#     print(JSONConverter.read_json(filename, date=dates[0]))
#
#     print("\nAnnotations for user2 on a specific date:")
#     print(JSONConverter.read_json(filename, user="user2", date=dates[1]))
#
#     print("\nAnnotations for user2 on a specific date:")
#     print(JSONConverter.read_json(filename, user="user2", date=dates[1]))
#
#     # Delete all annotations for user1
#     JSONConverter.delete_annotation(filename, "user1")
#     print("\nAfter deleting all annotations for user1:")
#     print(JSONConverter.read_json(filename))
