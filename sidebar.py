import os
import os.path
from util.filename_formatter import dotjava_to_dotjson
# from labeling_db import LabelDB

# from data_labeling.labeling_libs.java_file_parser import JavaFileParser
from labeling_libs.java_to_json import JavaToJSON
class Sidebar:

    @staticmethod
    def __convert_java_to_json(st, uploaded_file):

        date_today = st.session_state.date_today
        user = st.session_state.user
        annotations = st.session_state.default_annotations
        # target_dir = st.session_state.json_dir
        # annotations = {"relevant": 0, "quality": 1}

        # Read uploaded Java code
        content = uploaded_file.getvalue().decode("utf-8")
        source_file = uploaded_file.name

        target_file = dotjava_to_dotjson(source_file,user,date_today)

        return JavaToJSON.parse_as_json(
            content,
            source_file,
            target_file,
            user,
            date_today,
            annotations,
            other_info=""
        )
        # JavaToJSON.save_java_to_json(content, source_file, target_file, target_dir, user, date_today, annotations, other_info="")
        # JavaFileParser.save_to_json(content, filename, date_today, annotations=annotations, user=user)

    @staticmethod
    def __save_annotation(st, create_new= None):

        data = st.session_state.working_labeling_data
        json_file = st.session_state.working_file
        json_dir = st.session_state.json_dir
        user = st.session_state.user
        date = st.session_state.date_today

        JavaToJSON.save_data_to_json(data, json_file, json_dir, user, date, create_new=create_new)


    @staticmethod
    def __list_json_files(st, user=None, date=None):
        # st.session_state.date_today
        # get the list of json files, creater of the files and date created
        # date = None in parameter means files created in any date by the user
        json_files, users, dates = JavaToJSON.list_json_files(
            json_dir=st.session_state.json_dir,
            user=user,
            date=date
        )

        return json_files, users, dates

    @staticmethod
    def __create_sidebar(st):

        date_today = st.session_state.date_today
        user = st.session_state.user

        st.markdown(f"""User: {user}<br>Date: {date_today}""",
                    unsafe_allow_html=True)

        st.divider()

        json_files, users, dates = st.session_state.json_files

        uploaded_file = st.file_uploader("Upload Java code file", type=["java"])

        if uploaded_file:

            filename = uploaded_file.name
            json_filename = dotjava_to_dotjson(filename, user, date_today)

            if json_filename in json_files:
                st.write("Warning: There already exist JSON file of this name")
            else:
                with st.spinner("Converting java file to JSON data"):
                    st.session_state.working_file = json_filename
                    st.session_state.working_labeling_data = Sidebar.__convert_java_to_json(st, uploaded_file)

                    Sidebar.__save_annotation(st, create_new=None)

                    # reload the JSON files
                    st.session_state.json_files = Sidebar.__list_json_files(st, user=None, date=None)
                    json_files, users, dates = st.session_state.json_files

                st.write(f"{filename} saved")



        #handle file selection option on change event
        def file_sel_change():
            # print("ONCHANGE ON CHANGE", st.session_state.sel_file)
            if st.session_state.sel_file is not None:
                file_option = st.session_state.sel_file
                st.session_state.working_labeling_data = JavaToJSON.read_json(file_option, st.session_state.json_dir)
                st.session_state.working_file = file_option


        user_option = st.selectbox("Select user:", users, index=None, key="sel_user")

        date_option = st.selectbox("Select file date:", dates, index=None, key = "sel_date")

        file_option = st.selectbox("Select file:", json_files, index=None, key="sel_file", on_change=file_sel_change)

        st.write(f"User Option: {user_option}")
        st.write(f"Date Option: {date_option}")
        st.write(f"File option: {file_option}")


    @staticmethod
    def load_sidebar(st):
        if "json_files" not in st.session_state:
            user = st.session_state.user
            st.session_state.json_files = Sidebar.__list_json_files(st, user=None, date=None)

        with st.sidebar:
            Sidebar.__create_sidebar(st)
