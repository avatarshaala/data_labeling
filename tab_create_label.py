from labeling_libs.annotation_to_json import JSONConverter
import labeling_page
from datetime import datetime

def populate_tab(st, json_file, user):

    print("HERE")
    # Retrieve the most recent annotations for the given user
    dates, recent_annotations = JSONConverter.get_recent_annotation(json_file, user)
    date = datetime.strptime(dates[0],"%d-%m-%Y").strftime("%d-%m-%Y")

    # print(recent_annotations)
    st.subheader(f"User: {user}, Date: {date}, File: {json_file}")

    # Display the Java program with checkboxes
    for line_number, line_data in recent_annotations.items():
        content = line_data.get("content", "")
        type = line_data.get("type","")
        for date, annotations in line_data["annotation"][user].items():

            # Create a columns layout with 3 columns
            cols = [1]*len(annotations)
            cols.append(10)

            cols = st.columns(cols)

            with cols[len(cols)-1]:
                st.code(content,language="java")

            if type != "begin_comment" and type != "comment":
                break


            idx = 0
            # Collect annotations for the line
            collected_annotations = {}
            for criteria, value in annotations.items():

                value = True if value==1 else False



                # print(f"checkbox_{line_number}_{idx}")
                with cols[idx]:
                    # st.write(criteria)
                    # Collect annotations using checkboxes
                    collected_annotations[criteria] = st.checkbox(" ",key=f"checkbox_{line_number}_{idx}", help=criteria,value=value)
                idx+=1

