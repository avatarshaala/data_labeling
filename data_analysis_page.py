import os
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (cohen_kappa_score, classification_report, confusion_matrix)

from scipy.stats import chi2_contingency

from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
from itertools import combinations

import matplotlib.colors as mcolors
from io import BytesIO

def render_latex(formula, fontsize=12, dpi=300):
    """Renders LaTeX formula into Streamlit."""
    fig = plt.figure()
    text = fig.text(0, 0, '$%s$' % formula, fontsize=fontsize)

    fig.savefig(BytesIO(), dpi=dpi)  # triggers rendering

    bbox = text.get_window_extent()
    width, height = bbox.size / float(dpi) + 0.05
    fig.set_size_inches((width, height))

    dy = (bbox.ymin / float(dpi)) / height
    text.set_position((0, -dy))

    buffer = BytesIO()
    fig.savefig(buffer, dpi=dpi, format='jpg')
    plt.close(fig)

    st.image(buffer)

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


# # Function to load the sidebar
# def load_sidebar(selected_files):
#     st.sidebar.title("Selected Files")
#
#     # Display selected files
#     for i, file_name in enumerate(selected_files):
#         # Display clickable filename
#         if st.sidebar.button(file_name, key=f"btn_{i}"):
#             selected_files.remove(file_name)  # Remove file from selected files list
#             print(selected_files)
#             st.rerun()
#             # load_body()  # Load DataFrame for selected file


# Function to load the main body
# def load_body():
#     st.title("JSON File Viewer")
#
#     # handle file selection option on change event
#     def file_sel_change():
#         if st.session_state.sel_data_file is not None:
#             selected_file = st.session_state.sel_data_file
#         if selected_file not in st.session_state.selected_files:
#             st.session_state.selected_files.append(selected_file)
#
#     def user_sel_change():
#         if st.session_state.sel_user is not None:
#             st.write(st.session_state.sel_user)
#         pass
#
#     users = set(file.split(".")[0] for file in os.listdir(folder_path) if file.endswith((".json")))
#     # print(users)
#     selected_user = st.multiselect("Select annotators",
#                                  users, default=None,
#                                  format_func=lambda x: os.path.basename(x), key='sel_user',
#                                  on_change=user_sel_change)
#
#     # Select JSON file
#     selected_file = st.selectbox("Select JSON File",
#                                  [file for file in os.listdir(folder_path) if file.endswith(".json")], index=None,
#                                  format_func=lambda x: os.path.basename(x), key='sel_data_file',  on_change=file_sel_change)
#
#
#     print("list:", selected_file)
#     print("Session: ", st.session_state.selected_files)
#
#
#     if selected_file:
#         # Load DataFrame for selected file
#         df = load_dataframe(os.path.join(folder_path, selected_file))
#
#         # Display DataFrame
#         st.write(df)





# Define a function to calculate the majority vote for each criterion
# def get_majority_vote(group):
#   """
#   This function calculates the majority vote for each criterion ('relevant', 'quality', 'sufficient').
#
#   Args:
#       group: A pandas group object containing annotations for a specific data point.
#
#   Returns:
#       A Series containing the final annotation for each criterion (relevant, quality, sufficient).
#   """
#   criteria = ['relevant', 'quality', 'sufficient']
#   final_annotations = {}
#   for criterion in criteria:
#     # Consider only 1 and 0 values (exclude None)
#     votes = group[criterion].dropna()
#     if len(votes) == 0:
#       final_annotations[criterion] = None  # No valid votes, set to None
#     else:
#       final_annotations[criterion] = votes.mode().iloc[0]  # Majority vote
#   return pd.Series(final_annotations)

def majority_vote(x):
    mod = pd.Series.mode(x)
    # print("MODE:", mod.values)
    if len(mod.values) == 0:
        mod = None

    return mod



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
    total_relevant_count = df['relevant'].notnull().sum()
    total_quality_count = df['quality'].notnull().sum()

    relevant_count = (df['relevant']==1).sum()
    quality_count = (df['quality']==1).sum()

    # Count total valid records for 'sufficient' column
    total_sufficiency_count = df['sufficient'].notnull().sum()

    # Count 'sufficient' records with value 1
    sufficient_count = (df['sufficient']==1).sum()

    # Calculate percentages
    relevant_percentage = (relevant_count / total_relevant_count) * 100
    quality_percentage = (quality_count / total_quality_count) * 100

    # Calculate sufficient percentage based on valid records
    sufficient_percentage = (sufficient_count / total_sufficiency_count) * 100

    return (relevant_percentage, total_relevant_count), \
           (quality_percentage, total_quality_count), \
           (sufficient_percentage, total_sufficiency_count)

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
            subset_df[user1_col] = subset_df[user1_col].astype(int)
            subset_df[user2_col] = subset_df[user2_col].astype(int)
            # st.write(subset_df[user1_col])
            # st.write(subset_df[user2_col])

            # if label=='relevant' and user1=='user2' and user2 == 'user3':
            #     print(list(subset_df[user1_col].values))
            #     print("-----------")
            #     print(list(subset_df[user2_col].values))

            # Compute Cohen's kappa coefficient
            kappa = cohen_kappa_score(subset_df[user1_col].values, subset_df[user2_col].values)
            data[label][f'{user1}-{user2}'] = kappa

    # Create DataFrame with specified column names
    kappas_df = pd.DataFrame(data)
    kappas_df.index.name = 'Raters'
    return kappas_df.reset_index(), files_lines_used


def calculate_fleiss_kappa(df,labels):
    calc_kappas = {label: {} for label in labels}
    # files_lines_used = []
    for label in labels:
        user_columns = [col for col in df.columns if col.startswith(label + '_')]
        # users = [col.split('_')[-1] for col in user_columns]
        subset_data = df[user_columns].dropna().to_numpy().astype(int)

        table, cat_uni = aggregate_raters(subset_data,n_cat=2) #n_cat = 2 for binary annotation (0 or 1)
        kappa = fleiss_kappa(table, method='fleiss')

        calc_kappas[label] = kappa

    return calc_kappas



def calculate_prf_confusion(df, criteria):

    data = {crit: {} for crit in criteria}
    for crit in criteria:
        user_columns = [col for col in df.columns if col.startswith(crit + '_')]
        users = [col.split('_')[-1] for col in user_columns]
        data[crit] = {}
        for user1, user2 in combinations(users, 2):

            data[crit][f'{user1}-{user2}'] = {}
            user1_col = f'{crit}_{user1}'
            user2_col = f'{crit}_{user2}'

            y_true = df[user1_col].dropna().to_numpy().astype(int)
            y_pred = df[user2_col].dropna().to_numpy().astype(int)

            report = classification_report(y_true,y_pred, labels=[0,1], output_dict=True)

            metrices = ['precision', 'recall', 'f1-score', 'support']
            for met in metrices:
                data[crit][f'{user1}-{user2}'][met] = {
                    '0':report['0'][met],
                    '1':report['1'][met],
                    'macro avg':report['macro avg'][met],
                    'weighted avg': report['weighted avg'][met]

                }

            cm = confusion_matrix(y_true, y_pred,labels=[0, 1])
            data[crit][f'{user1}-{user2}']['confusion matrix'] = cm

            data[crit][f'{user1}-{user2}']['accuracy'] = report['accuracy']

            # st.write(data)


    return data


def chi2_test(df, labels):
    data = {label: {} for label in labels}
    files_lines_used = []
    for label in labels:
        user_columns = [col for col in df.columns if col.startswith(label + '_')]
        users = [col.split('_')[-1] for col in user_columns]

        for user1, user2 in combinations(users, 2):
            user1_col = f'{label}_{user1}'
            user2_col = f'{label}_{user2}'

            # Filter dataframe for user1 and user2
            subset_df = df[['file', 'line', user1_col, user2_col]].dropna().copy()

            # # Keep track of files and lines used in kappa and return finally
            # files_lines_used = files_lines_used + [(file, line) for file, line in
            #                                        zip(subset_df['file'].tolist(), subset_df['line'].tolist())]

            # Ensure that the target columns contain appropriate data types
            subset_df[user1_col] = subset_df[user1_col].astype(int)
            subset_df[user2_col] = subset_df[user2_col].astype(int)


            ## Create contingency table
            contingency_table = pd.crosstab(subset_df[user1_col], subset_df[user2_col])

            # Perform chi-square test
            chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
            chi = {"chi2_stat": chi2_stat, "p_val": p_val, "dof": dof, "expected": expected}
            data[label][f'{user1}-{user2}'] = chi

    # Create DataFrame with specified column names
    chi_df = pd.DataFrame(data)
    chi_df.index.name = 'Raters'
    return chi_df.reset_index()

# Function to visualize the percentage
def visualize_percentage(relevant_percentage, quality_percentage, sufficient_percentage, categories):

    percentages = [relevant_percentage, quality_percentage, sufficient_percentage]
    # Plot the bar chart using Seaborn
    plt.figure(figsize=(8, 6))
    sns.barplot(x=categories, y=percentages, palette='muted',edgecolor='azure', linewidth=0.5)
    plt.xlabel('Categories')
    plt.ylabel('Percentage')
    plt.title('Percentage of Relevant, Quality, and Sufficient Labels')
    plt.ylim(0, 100)
    for i, percentage in enumerate(percentages):
        plt.text(i, percentage + 2, f"{percentage:.2f}%", ha='center')
    st.pyplot(plt)
    plt.savefig("plots/percentages.pdf", format="pdf", bbox_inches="tight")



def visualize_percentage_agreement(classification_report, raters = []):
    plt.rc('font', size=16)  # default size
    plt.rc('axes', titlesize=22)  # title size of axes
    plt.rc('axes', labelsize=22)
    plt.rc('xtick', labelsize=22)
    # plt.rc('legend',fontsize=50)

    n_decimal = 3
    criteria = {"relevant":"Relevancy", "quality":"Quality", "sufficient":"Sufficiency"}
    rater_combinations = list(combinations(raters, 2))
    n_cols = len(rater_combinations)

    fig, axes = plt.subplots(1, n_cols, figsize=(25,5), layout='constrained', sharex=True, sharey=True)

    for i, (user1, user2) in enumerate(rater_combinations):
        agreements = {"both_0": [], "both_1": [], "disagree": []}
        for crit in criteria:
            confusion_matrix = classification_report[crit][f'{user1}-{user2}']['confusion matrix']
            total = confusion_matrix.sum()
            both_0 = 100 * confusion_matrix[0, 0] / total
            both_1 = 100* confusion_matrix[1, 1] / total

            both_0 = int(both_0*10**n_decimal)/(10**n_decimal)
            both_1 = int(both_1*10**n_decimal)/(10**n_decimal)

            disagree = 100 - both_0 - both_1


            agreements['both_0'].append(both_0)
            agreements['both_1'].append(both_1)

            agreements['disagree'].append(disagree)

        x = np.arange(len(criteria))
        width = 0.19  # the width of the bars
        multiplier = 0
        # fig, ax = plt.subplots(layout='constrained')

        for (agr, vals), col in zip(agreements.items(), ['darksalmon','tan','cornflowerblue']):
            print("VAL: ", agr, vals)
            offset = width * multiplier
            rects = axes[i].bar(
                x + offset,
                vals, width,
                label=agr,
                edgecolor='azure',
                linewidth=0.5,
                color= col
            )
            axes[i].bar_label(rects, padding=3)
            multiplier += 1

        axes[i].set_ylabel('% Values ')
        axes[i].set_title(f'{user1.replace("user","Rater")} and {user2.replace("user","Rater")}')
        axes[i].set_xticks(x + width, [label for _, label in criteria.items()])
        axes[i].legend(loc='upper right', fontsize=20, frameon=False)
        axes[i].set_ylim(0, 110)

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axes.flat:
            ax.label_outer()

    st.pyplot(plt)
    plt.savefig("plots/percentage_agreement.pdf", format="pdf", bbox_inches="tight")


def visualize_prf(classification_report, raters=[]):
    plt.rc('font', size=16)  # default size
    plt.rc('axes', titlesize=22)  # title size of axes
    plt.rc('axes', labelsize=22)
    plt.rc('xtick', labelsize=22)

    # plt.rc('legend',fontsize=50)
    def construct_plot(ax, report, metrices, title):

        classes = ["class 0", "class 1"]  # The raters rate each data as 0 or 1
        x = np.arange(len(classes))

        n_decimal = 3

        # Create a dictionary to store values of metrices from the report
        met_values = {met: [] for met in metrices}
        # Store the value of each metric from the report in n_decimal places
        for metric, values in met_values.items():
            for i, c in enumerate(classes):
                # precisions.append()
                values.append(int(report[f'{metric.lower()}'][f'{i}'] * 10 ** n_decimal) / (10 ** n_decimal))

        width = 0.17  # the width of the bars
        multiplier = 0

        for (metric, values), col in zip(met_values.items(),['darksalmon','tan','cornflowerblue']):
            offset = width * multiplier
            rects = ax.bar(x + offset, values, width, label=metric, color=col, edgecolor='azure', linewidth=0.5)
            ax.bar_label(rects, padding=3)
            multiplier += 1

            # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Values ')
        ax.set_title(f'{title}')
        ax.set_xticks(x + width, classes)
        ax.legend(loc='upper center', fontsize=20, frameon=False)
        ax.set_ylim(0, 1.07)
        # ax.bar_label(fontsize=14)

    # we want to visualize metrices of each class (0, 1) as group in bar chart,
    # hence value of each matric of two class is stored in a list of two elements
    # metrices = {"Precision": [], "Recall": [], "F1-Score": []}

    criteria = [("relevant","Relevancy"), ("quality","Quality"), ("sufficient","Sufficiency")]
    # criteria = ["Relevancy", "Quality", "Sufficiency"]
    rater_combinations = list(combinations(raters, 2))

    n_rows = len(criteria)
    n_cols = len(rater_combinations)

    fig, axes = plt.subplots(n_rows, n_cols, layout='constrained',figsize=(25,13), sharex=True, sharey=True)

    metrices = ["Precision", "Recall", "F1-Score"]
    for row in range(n_rows):
        for col in range(n_cols):
            rater1, rater2 = rater_combinations[col]
            # st.write(rater1, rater2)
            # st.write("________")
            report = classification_report[f'{criteria[row][0]}'][f"{rater1}-{rater2}"]
            title = f"{criteria[row][1]} {rater1.replace('user', 'Rater')} and {rater2.replace('user', 'Rater')}"
            ax = axes[row, col]
            # ax = axes[row, col]
            construct_plot(ax, report, metrices, title)


    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axes.flat:
        ax.label_outer()


    st.pyplot(plt)
    plt.savefig(f"plots/prf_metrices.pdf", format="pdf", bbox_inches="tight")




def handler(handle_name):
    def file_sel_change():
        if st.session_state.sel_data_files is not None:
            selected_files = st.session_state.sel_data_files
            st.session_state.selected_files = selected_files
            # st.write(st.session_state.selected_files)
        #     selected_file = st.session_state.sel_data_files
        # if selected_file not in st.session_state.selected_files:
        #     st.session_state.selected_files.append(selected_file)

    def annotator_sel_change():
        if st.session_state.sel_annotators is not None:
            selected_annotators = st.session_state.sel_annotators
            st.session_state.selected_annotators = selected_annotators
            # st.write(st.session_state.selected_annotators)

    handles = {
        'file_sel_change': file_sel_change,
        'annotator_sel_change': annotator_sel_change
    }

    return handles[handle_name]
def main():
    annotators = set(file.split(".")[0] for file in os.listdir(folder_path) if file.endswith((".json")))
    print(len(st.session_state.selected_annotators))
    files = [ file for file in os.listdir(folder_path) if file.endswith(".json") and
              (file.split(".")[0] in st.session_state.selected_annotators or len(st.session_state.selected_annotators)==0)
              ]

    default_files = st.session_state.selected_files
    if "All" in st.session_state.selected_files:
        st.session_state.selected_files = files
        default_files = "All"
        files = []


    # files = [file for file in files if file.split(".")[0] in st.session_state.selected_annotators]
    # print(users)


    annotator_select_handler = handler("annotator_sel_change")
    selected_annotator = st.multiselect("Select annotators",
                                        annotators,
                                        default=None,
                                        key='sel_annotators',
                                        on_change=annotator_select_handler)

    # Select JSON file
    file_select_handler = handler("file_sel_change")
    selected_file = st.multiselect("Select JSON File",
                                   ["All"] + files,
                                   default=default_files,
                                   format_func=lambda x: os.path.basename(x), key='sel_data_files',
                                   on_change=file_select_handler)

    st.write(st.session_state.selected_files)
    st.write(st.session_state.selected_annotators)

    # Button to append data from selected files
    if st.button("Append Data"):

        # Metric calculations
        appended_df = append_data(st.session_state.selected_files)
        # Concatenate (join) dataframes for each user
        joined_df = join_dataframes(appended_df)

        #Majority vote calculation
        # Group the DataFrame by 'file' and 'line' (unique identifier)
        vote_df = appended_df.groupby(['file', 'line']).agg(
            {"relevant":majority_vote,
             "quality":majority_vote,
             "sufficient":majority_vote
             }
        ).reset_index()

        # Calculate percentage of relevant, quality, and sufficient
        (relevant_percentage,num_rel_data), (quality_percentage, num_qual_data), (sufficient_percentage, num_suff_data) = calculate_percentage(vote_df)

        labels = ['relevant', 'quality', 'sufficient']

        # Calculate kappa
        kappas, files_lines = calculate_kappa(joined_df, labels)
        f_kappa = calculate_fleiss_kappa(joined_df, labels)

        # Calculate precision recall f1 and confusion matrix
        prf_confusion = calculate_prf_confusion(joined_df, criteria = labels)

        chi_square = chi2_test(joined_df,labels=labels)


        # Display output and visualization
        with st.expander("View dataframes"):
            st.subheader("Appended DataFrame")
            st.write(appended_df)

            st.subheader("Joined Data")
            st.write(joined_df)

            st.subheader("Majority Vote")
            st.write(vote_df)

        # Visualize the percentage
        percentages = [relevant_percentage, quality_percentage, sufficient_percentage]
        categories = ['Relevant', 'Quality', 'Sufficient']

        with st.expander("Visualizations"):
            st.write(f"Percentage of Relevant Labels: {relevant_percentage:.2f}% #Total: {num_rel_data}")
            st.write(f"Percentage of Quality Labels: {quality_percentage:.2f}% #Total: {num_qual_data}")
            st.write(f"Percentage of Sufficient Labels: {sufficient_percentage:.2f}% #Total: {num_suff_data}")

            visualize_percentage(
                relevant_percentage,
                quality_percentage,
                sufficient_percentage,
                categories = ['Relevant', 'Quality', 'Sufficient']
            )

        # Generate LaTeX visualization
        # Display the LaTeX visualization using a custom HTML component

        with st.expander("LaTeX Code for percentage visualization"):
            latex_code = latex_visualization(categories, percentages)
            st.code(latex_code)




        with st.expander("Interrater agreements"):
            st.subheader("Cohen's Kappa")
            st.write(kappas)
            st.subheader("Fleiss Kappa")
            st.write(f_kappa)

            st.subheader("Chi-Squared Test")
            st.write(chi_square)

            st.subheader("Classification report")

            visualize_prf(prf_confusion,raters = st.session_state.selected_annotators)

            visualize_percentage_agreement(prf_confusion, st.session_state.selected_annotators)

            st.json(prf_confusion)


            # st.subheader("Confusion Matrix")
            # st.json(confusion)
            st.subheader("Files Used for kappa")
            st.write({i: f for i, f in enumerate(set([file[0] for file in files_lines]))})
        # st.write(set([tuple(files) for files,_ in files_lines]))






if __name__ == "__main__":
    st.set_page_config(layout="wide")

    if "selected_files" not in st.session_state:
        st.session_state.selected_files = []
    if "selected_annotators" not in st.session_state:
        st.session_state.selected_annotators = []

    main()
