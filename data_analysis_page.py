import os
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
from itertools import combinations
import matplotlib.colors as mcolors

# Define the folder path
folder_path = "jsons"

# Function to load the dataframe from a JSON file
def load_dataframe(json_file):
    # Read JSON file
    with open(json_file, "r") as f:
        data = json.load(f)

    # Extract user, date, and file name from the file name
    file_name = os.path.basename(json_file)
    user, date, file = file_name.split('.', 2)

    # Initialize lists to store data
    lines = []

    # Process each line in the JSON file
    for line_number, line_data in data["lines"].items():
        content = line_data.get("content", "")
        labels = line_data.get("labels", {})

        # Check if the line has labels
        if labels:
            relevant = labels.get("relevant", pd.NA)
            quality = labels.get("quality", pd.NA)
            sufficient = labels.get("sufficient", pd.NA)

            # Append line data to the list
            lines.append([user, file, date, int(line_number), content, relevant, quality, sufficient])

    # Create a DataFrame
    df = pd.DataFrame(lines,
                      columns=["user", "file", "date", "line", "content", "relevant", "quality", "sufficient"])

    return df


# Function to load the sidebar
def load_sidebar(selected_files):
    st.sidebar.title("Selected Files")

    # Display selected files
    for i, file_name in enumerate(selected_files):
        # Display clickable filename
        if st.sidebar.button(file_name, key=f"btn_{i}"):
            selected_files.remove(file_name)  # Remove file from selected files list
            print(selected_files)
            st.rerun()
            # load_body()  # Load DataFrame for selected file


# Function to load the main body
def load_body():
    st.title("JSON File Viewer")

    # handle file selection option on change event
    def file_sel_change():
        if st.session_state.sel_data_file is not None:
            selected_file = st.session_state.sel_data_file
        if selected_file not in st.session_state.selected_files:
            st.session_state.selected_files.append(selected_file)


    # Select JSON file
    selected_file = st.selectbox("Select JSON File",
                                 [file for file in os.listdir(folder_path) if file.endswith(".json")], index=None,
                                 format_func=lambda x: os.path.basename(x), key='sel_data_file',  on_change=file_sel_change)


    print("list:", selected_file)
    print("Session: ", st.session_state.selected_files)

    if selected_file:
        # Load DataFrame for selected file
        df = load_dataframe(os.path.join(folder_path, selected_file))

        # Display DataFrame
        st.write(df)


# Function to append data from selected files
def append_data(selected_files):
    # Initialize an empty DataFrame to store appended data
    appended_df = pd.DataFrame()

    # Iterate over selected files
    for file_name in selected_files:
        # Load DataFrame for selected file
        df = load_dataframe(os.path.join(folder_path, file_name))
        # Append DataFrame to the main DataFrame
        appended_df = pd.concat([appended_df, df], ignore_index=True)

    return appended_df

# def concatenate_dataframes(appended_df):
#     # Initialize an empty DataFrame to store concatenated data
#     concatenated_df = pd.DataFrame()
#
#     # Group appended dataframe by user
#     grouped = appended_df.groupby("user")
#
#     # Iterate over groups
#     for user, group_df in grouped:
#         # Add suffix to label columns based on user name
#         suffix = f"_{user.lower()}"
#         label_columns = [col for col in group_df.columns if col not in ["user", "file", "date", "line", "content"]]
#         group_df = group_df[["file", "line", "content"] + label_columns].add_suffix(suffix)
#
#         # Concatenate dataframes
#         concatenated_df = pd.concat([concatenated_df, group_df], axis=1)
#
#     return concatenated_df

def join_dataframes(original_df):
    final_df = None  # Initialize an empty dataframe to store the final result

    # Step 1: Iterate over unique users
    users = original_df['user'].unique()

    # Step 2: Iterate over users and perform left join with the final dataframe
    for user in users:
        # Step 3: Filter dataframe for each user
        user_df = original_df[original_df['user'] == user].copy()
        # Step 4: Remove 'date' column
        user_df = user_df.drop(columns=['date', 'user'])
        user_df = user_df.rename(columns={col: f"{col}_{user}" for col in ["relevant", "quality", "sufficient"]})
        # Step 5: Perform left join with the final dataframe
        if final_df is None:
            final_df = user_df  # Initialize final_df with user_df for the first iteration
        else:
            user_df = user_df.drop(columns=['content'])
            final_df = pd.merge(final_df, user_df, on=['file', 'line'], suffixes=('', f'_{user}'), how='left')

    return final_df



# Example usage:
# joined_dataframe = join_dataframes(original_dataframe)


# Function to calculate percentage of relevant and quality
# Function to calculate percentage of relevant, quality, and sufficient
def calculate_percentage(df):
    total_lines = len(df)
    relevant_count = df['relevant'].sum()
    quality_count = df['quality'].sum()

    # Count total valid records for 'sufficient' column
    total_sufficiency_count = df['sufficient'].notnull().sum()

    # Count 'sufficient' records with value 1
    sufficient_count = (df['sufficient'] == 1).sum()

    # Calculate percentages
    relevant_percentage = (relevant_count / total_lines) * 100
    quality_percentage = (quality_count / total_lines) * 100

    # Calculate sufficient percentage based on valid records
    sufficient_percentage = (sufficient_count / total_sufficiency_count) * 100

    return relevant_percentage, quality_percentage, sufficient_percentage

# Function to generate LaTeX visualization
def latex_visualization(categories, percentages):
    # Start LaTeX code
    latex_code = (
        "\\begin{figure}[htbp]\n" +
        "\\centering\n" +
        "\\begin{tikzpicture}\n" +
        "\\begin{axis}[\n" +
        "\tybar,\n" +
        "\tymin=0,\n" +
        "\tymax=100,\n" +
        "\twidth=\\textwidth,\n" +
        "\theight=6cm,\n" +
        "\tylabel={Percentage},\n" +
        "\tsymbolic x coords={" +
        ", ".join(categories) +
        "},\n" +
        "\txtick=data,\n" +
        "\tnodes near coords,\n" +
        "\tnodes near coords align={vertical},\n" +
        "\tbar width=0.5cm,\n" +
        "]\n"
    )

    # # Define colors for bars dynamically based on the number of categories
    # num_categories = len(categories)
    # color_palette = sns.color_palette('muted', n_colors=num_categories)
    # colors = [mcolors.rgb2hex(color)[1:] for color in color_palette]

    # Define colors for bars
    colors = ['blue', 'green', 'red']

    # Append LaTeX code for each bar with its color
    for i, (category, percentage) in enumerate(zip(categories, percentages)):
        latex_code += (
            f'\\addplot[fill={colors[i]}!40,draw={colors[i]}!40] coordinates {{({category},{percentage})}};\n'
        )

    # Append remaining LaTeX code
    latex_code += (
        '\\end{axis}\n' +
        '\\end{tikzpicture}\n' +
        '\\caption{Percentage of ' +
        ', '.join(categories) +
        ' Labels}\n' +
        '\\label{fig:bar_chart}\n' +
        '\\end{figure}\n'
    )

    return latex_code


def calculate_kappa(joined_df, labels):
    data = {label: {} for label in labels}
    files_lines_used = []
    for label in labels:
        user_columns = [col for col in joined_df.columns if col.startswith(label + '_')]
        users = [col.split('_')[-1] for col in user_columns]

        for user1, user2 in combinations(users, 2):
            user1_col = f'{label}_{user1}'
            user2_col = f'{label}_{user2}'

            # Filter dataframe for user1 and user2
            subset_df = joined_df[['file', 'line', user1_col, user2_col]].dropna().copy()

            # Keep track of files and lines used in kappa and return finally
            files_lines_used = files_lines_used+ [(file, line) for file, line in zip(subset_df['file'].tolist(), subset_df['line'].tolist())]

            # Ensure that the target columns contain appropriate data types
            subset_df[user1_col] = subset_df[user1_col].astype('category')
            subset_df[user2_col] = subset_df[user2_col].astype('category')

            # Compute Cohen's kappa coefficient
            kappa = cohen_kappa_score(subset_df[user1_col], subset_df[user2_col])
            data[label][f'{user1}-{user2}'] = kappa

    # Create DataFrame with specified column names
    kappas_df = pd.DataFrame(data)
    kappas_df.index.name = 'Rators'
    return kappas_df.reset_index(), files_lines_used

# Function to visualize the percentage
def visualize_percentage(relevant_percentage, quality_percentage, sufficient_percentage):
    categories = ['Relevant', 'Quality', 'Sufficient']
    percentages = [relevant_percentage, quality_percentage, sufficient_percentage]

    # Generate LaTeX visualization
    latex_code = latex_visualization(categories, percentages)

    # Plot the bar chart using Seaborn
    plt.figure(figsize=(8, 6))
    sns.barplot(x=categories, y=percentages, palette='muted')
    plt.xlabel('Categories')
    plt.ylabel('Percentage')
    plt.title('Percentage of Relevant, Quality, and Sufficient Labels')
    plt.ylim(0, 100)
    for i, percentage in enumerate(percentages):
        plt.text(i, percentage + 2, f"{percentage:.2f}%", ha='center')
    st.pyplot(plt)


    # Display the LaTeX visualization using a custom HTML component
    st.write("LaTeX Visualization:")
    st.code(latex_code)



def main():
    # Initialize list to store selected files
    selected_files = st.session_state.get("selected_files", [])

    # Update session state with selected files
    st.session_state.selected_files = selected_files

    # Load the sidebar
    load_sidebar(selected_files)

    # Load the body
    load_body()

    # Button to append data from selected files
    if st.button("Append Data"):
        appended_df = append_data(selected_files)
        st.write("Appended Data:")
        st.write(appended_df)



        # Concatenate dataframes for each user
        joined_df = join_dataframes(appended_df)
        st.write("Joined Data:")
        st.write(joined_df)

        # Calculate percentage of relevant, quality, and sufficient
        relevant_percentage, quality_percentage, sufficient_percentage = calculate_percentage(appended_df)
        st.write(f"Percentage of Relevant Labels: {relevant_percentage:.2f}%")
        st.write(f"Percentage of Quality Labels: {quality_percentage:.2f}%")
        st.write(f"Percentage of Sufficient Labels: {sufficient_percentage:.2f}%")

        # Visualize the percentage
        visualize_percentage(relevant_percentage, quality_percentage, sufficient_percentage)

        # Calculate kappa
        labels = ['relevant', 'quality', 'sufficient']
        kappas, files_lines = calculate_kappa(joined_df, labels)
        st.write("Interrator agreements")
        st.write(kappas)
        st.write("Files Used for kappa")
        st.write(set([file[0] for file in files_lines]))
        # st.write(set([tuple(files) for files,_ in files_lines]))


if __name__ == "__main__":
    main()
