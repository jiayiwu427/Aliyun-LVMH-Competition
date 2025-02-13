import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from chatbot_app import QWENapp
import ast
import jieba
from wordcloud import WordCloud
import re
import numpy as np

# Function to count word frequencies
def count_word_freq(column):
    """
    Count keyword frequencies in a pandas DataFrame column.
    :param column: A pandas Series where each row contains a list of words.
    :return: Counter object with word frequencies.
    """
    # Flatten all lists from the column into a single list, ignoring NaN values
    all_keywords = [word for keywords in column.dropna() for word in keywords]
    
    # Count the frequency of each word using Counter
    return Counter(all_keywords)


# Function to get top N frequent keywords
def get_top_n(counter_obj, n=10):
    return counter_obj.most_common(n)
def count_col_list(df,col):
    dict_cnt = {}
    total = [j for i in df[col] for j in i if j not in (['其他','未明确提及','other'])]
    counter = Counter(total)
    for k,v in counter.items():
        dict_cnt[k] = round(v/len(total)*100,1)
    return dict_cnt
def plot_top_n(counter):
    labels, counts = zip(*counter) 
    plt.figure(figsize=(8, 4))  # Set the figure size
    plt.barh(labels, counts, color='skyblue')  # Create a horizontal bar plot
    plt.rcParams['font.sans-serif'] = ['Hei']
    # Step 5: Add labels and title
    plt.xlabel('Percentage of Total Mentions')  # Add x-axis label
    plt.ylabel(col)  # Add y-axis label
    #plt.title('Word Frequency')  # Add title

    # Step 6: Show the bar plot
    plt.gca().invert_yaxis()  # Invert y-axis to show the highest frequency on top
    st.pyplot(plt)

# Concatenate all text in the pandas Series into a single string
stopwords_path = 'cn_stopwords.txt'
with open(stopwords_path, 'r', encoding='utf-8') as f:
    stopwords = set([line.strip() for line in f])
stopwords.update(['话题','飞吻','害羞','偷笑','萌萌哒','捂脸','失望','捂脸','害羞','派对','笑哭','R','r','笑','哭','大笑','汗颜','惹','皱眉','叹气','色色','萌萌','哒'])
stopwords.update(['lv','lv村上隆','村上隆','路易威登','包包','包','路易','威登','路', '易', '威', '登','易威登','louisvuitton'])

def plot_wordcloud(df,max_words = 150,stopwords = stopwords):
    text = ' '.join(df['内容'])
    jieba.add_word("村上隆")
    jieba.add_word("lv")
    jieba.add_word("Louis Vuitton")
    regex = r"#\w+"
        # Remove all hashtags from the text
    #text = re.sub(regex, "", text).strip() 
    # Tokenize the text using jieba
    tokenized_text = jieba.lcut(text)
    text_final = []
    for word in tokenized_text:
        if re.match(r'^[a-zA-Z\s]+$', word):  # Matches English words/spaces
            word = word.lower().replace(" ", "")  # Convert to lowercase, remove spaces 
        # Keep the word only if it's not in stopwords
        if word not in stopwords:
            text_final.append(word)
    text_final = ''.join(text_final)
    # Specify the path to a Chinese-compatible font
    font_path = '/System/Library/Fonts/PingFang.ttc'  # Adjust path to your Mac's Chinese font file

    # Generate the word cloud
    wordcloud = WordCloud(
        random_state=42,
        font_path=font_path,  # Specify Chinese font
        background_color='white',  # Background color of the word cloud
        width=800,  # Width of the generated image
        height=400,  # Height of the generated image
        max_words=150,  # Maximum number of words in the word cloud
        collocations=False  # Avoid duplicate words
    ).generate(text_final)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Turn off axis
    st.pyplot(plt)
# Dummy Data: Replace this with your own dataset
def compute_word_scores(df, col):
    """
    Compute word scores (relative TF-IDF) per persona, accounting for term occurrences once per sentence.
    """
    df = df[df['persona'] != '无法确定']  # Filter rows where persona isn't "无法确定"
    word_score = {}
    persona_word_counts = {}
    persona_total_docs = {}  # Store the total number of sentences per persona

    # Define stopwords
    stopwords.update(['', '[', ']', '#', '\t', '～', '\xa0', '\ufeff', '️', '话题', '飞吻', '害羞', '偷笑',
                      '萌萌哒', '捂脸', '失望', '派对', '笑哭', 'R', 'r', '笑', '哭', '大笑', '汗颜', '惹', '皱眉', '叹气',
                      '色色', '萌萌', '哒', 'lv', 'lv村上隆', '村上隆', '路易威登', '包包', '包', '路易', '威登', 
                      '路', '易', '威', '登', '易威登', 'louisvuitton','🌟'])

    # Add custom words to jieba's dictionary
    jieba.add_word("carryall")
    jieba.add_word("Carryall")
    jieba.add_word("快时尚")
    jieba.add_word("穿搭")
    jieba.add_word("新款")
    jieba.add_word("太古里")
    jieba.add_word("读书")

    # Step 1: Compute document frequencies (count unique words per sentence for a persona)
    for persona in df['persona'].unique():
        text_data = df[df['persona'] == persona][col]  # All sentences/documents for this persona
        persona_total_docs[persona] = len(text_data)  # Count total number of documents (sentences)

        word_counts = Counter()  # Count words (unique per sentence)
        for sentence in text_data:
            # Preprocessing: Remove hashtags
            regex = r"#\w+"
            #sentence = re.sub(regex, "", sentence).strip()

            # Tokenize sentence
            tokens = jieba.lcut(sentence)
            filtered_tokens = []

            for token in tokens:
                # Lowercase and clean English words
                if re.match(r'^[a-zA-Z\s]+$', token):  # Matches English words/spaces
                    token = token.lower().replace(" ", "")
                if token not in stopwords:  # Filter out stopwords
                    filtered_tokens.append(token)

            # Convert filtered tokens to set (unique words per sentence)
            unique_words = set(filtered_tokens)
            word_counts.update(unique_words)  # Add unique words from each sentence

        # Store word counts for this persona
        persona_word_counts[persona] = word_counts

    # Step 2: Compute document frequency (DF) across all personas
    word_in_personas = Counter()  # Count how many personas each word appears in
    for word_counts in persona_word_counts.values():
        for word in word_counts:
            word_in_personas[word] += 1  # Increment count for each persona where word appears

    # Step 3: Compute TF-IDF scores
    for persona, word_counts in persona_word_counts.items():
        scores = {}
        for word, count_in_docs in word_counts.items():
            # Term Frequency (TF): Fraction of documents where the word appears
            tf = count_in_docs / persona_total_docs[persona]
            # Inverse Document Frequency (IDF): Reduce weight for common words across all personas
            idf = np.log(len(persona_word_counts) / (1 + word_in_personas[word]))
            # Final TF-IDF score
            scores[word] = tf * idf

        word_score[persona] = scores

    return word_score


def plot_word_clouds(word_scores,persona):
    """
    Plot a word cloud for each persona using the word scores.
    """
    font_path = '/System/Library/Fonts/STHeiti Medium.ttc'  # Path to a font that supports Chinese characters
    scores = word_scores[persona]
    wc = WordCloud(
        random_state = 42,
        font_path=font_path,
        background_color="white",
        max_words=100,
        width=400,
        height=400
    ).generate_from_frequencies(scores)
    
    # Plot the word cloud
    plt.figure(figsize=(5, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud for {persona}")
    st.pyplot(plt)

# Streamlit App



def display_column_data(column_name, top_n):
    keyword_counter = count_word_freq(df[column_name])
    top_n_keywords = keyword_counter.most_common(top_n)
    
    st.subheader(f"Top {top_n} Keywords for `{column_name}`")

    # Plot results
    if top_n_keywords:
        keywords, freq = zip(*top_n_keywords)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=freq, y=keywords, orient='h', palette='viridis')
        plt.title(f"Top {top_n} Keywords - `{column_name}`", fontsize=16)
        plt.xlabel("Frequency", fontsize=12)
        plt.ylabel("Keywords", fontsize=12)
        st.pyplot(plt)



df = pd.read_excel('processed_data/04_segmentation.xlsx')
col_list = ['Style_Category', 'Functionality_Category', 'Element_Category',
       'Product_Category', 'Competitor_Category','Product_Mention','Competitor','Keywords_Design_Elements', 'Keywords_Design_Style',
       'Keywords_Functionality']
for col in col_list:

# Convert string representations of lists to actual Python lists
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
persona_df = pd.read_csv('persona_descfription.csv')

def main():
    #st.set_page_config(page_title="Sidebar and Tabs Integration", layout="wide")
    if "selected_page" not in st.session_state:
        st.session_state["selected_page"] = "during engagement"

    # Function to update session state when navigation changes
    def change_page_by_sidebar(page):
        st.session_state["selected_page"] = page

    def change_page_by_tabs():
        st.session_state["selected_page"] = st.session_state.tab_selection
    # Sidebar Navigation
    with st.sidebar:
        st.header("Navigation")
        sidebar_selection = st.radio(
            "Go to",
            options=["before engagement", "during engagement", "after engagement"],
            index=["before engagement", "during engagement", "after engagement"].index(st.session_state["selected_page"]),
            on_change=lambda: change_page_by_sidebar(sidebar_selection)
        )


    # Render Page Based on Selected Page
    if st.session_state["selected_page"] == "before engagement":
        #st.title("Welcome to the Home Page")
        #st.write("This is the home page of your application!")

    #page = st.sidebar.radio('Choose:',('before engagement','after engagement'))
    #if page == 'before engagement':
        # Main application
        st.title("Product Insights")
        st.subheader("SOCIAL LISTENING")
        with st.container(border = True):
            st.write('What are people talking about on social media?')
            plot_wordcloud(df)
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border = True):
                st.write('What are the hottiest products?')
                temp = pd.DataFrame(get_top_n(count_word_freq(df[(df['Product_Category'] != '其他') & (df['Product_Category'] != '未明确提及')]['Product_Category'])))
                temp.columns = ['product', 'counts']
                temp = temp.sort_values(by = 'counts', ascending = False)
                st.bar_chart(temp,x='product',y='counts',horizontal = True)
        with col2:
            with st.container(border = True):
                st.write('Who are interested the most?')
                st.dataframe(pd.read_csv('persona_descfription.csv').reset_index(drop = True).iloc[:,:2])
        st.write('---')
        personas = ['潮流先锋', '实用主义者', '高端需求消费者', '复古怀旧派']

        # Step 2: Create a selectbox for selecting a persona
        selected_persona = st.selectbox("Select a persona", personas)

        # Step 3: Display content based on the selected persona
        if selected_persona == personas[0]:
            with st.container(border = True):
                st.write(persona_df[persona_df['segmentation'] == personas[0]]['description'].values[0])
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write('Gender Distribution')
            with col2:
                st.write('Age Distribution')
            with col3:
                st.write('City Tier Distribution')
            with col4:
                st.write('Word Cloud')
                plot_word_clouds(compute_word_scores(df, col="内容"),personas[0])
            with st.expander("See posts"):
                col1, col2 = st.columns(2)

                # Left Column: Original post text
                with col1:
                    st.subheader("Original Post 1")
                    st.write(df.iloc[2045,0])

                # Right Column: JSON Tags
                with col2:
                    st.subheader("Original Post 2")
                    st.write(df.iloc[1788,0])
        elif selected_persona == personas[1]:
            with st.container(border = True):
                st.write(persona_df[persona_df['segmentation'] == personas[1]]['description'].values[0])           
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write('Gender Distribution')
            with col2:
                st.write('Age Distribution')
            with col3:
                st.write('City Tier Distribution')
            with col4:
                st.write('Word Cloud')
                plot_word_clouds(compute_word_scores(df, col="内容"),personas[1])
            with st.expander("See posts"):
                col1, col2 = st.columns(2)

                # Left Column: Original post text
                with col1:
                    st.subheader("Original Post 1")
                    st.write(df.iloc[2045,0])

                # Right Column: JSON Tags
                with col2:
                    st.subheader("Original Post 2")
                    st.write(df.iloc[1788,0])
        elif selected_persona == personas[2]:
            st.write(persona_df[persona_df['segmentation'] == personas[2]]['description'].values[0])
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write('Gender Distribution')
            with col2:
                st.write('Age Distribution')
            with col3:
                st.write('City Tier Distribution')
            with col4:
                plot_word_clouds(compute_word_scores(df, col="内容"),personas[2])
        elif selected_persona == personas[3]:
            st.write(persona_df[persona_df['segmentation'] == personas[3]]['description'].values[0])
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write('Gender Distribution')
            with col2:
                st.write('Age Distribution')
            with col3:
                st.write('City Tier Distribution')
            with col4:
                plot_word_clouds(compute_word_scores(df, col="内容"),personas[3])
            with st.expander("See posts"):
                col1, col2 = st.columns(2)

                # Left Column: Original post text
                with col1:
                    st.subheader("Original Post 1")
                    st.write(df.iloc[2045,0])

                # Right Column: JSON Tags
                with col2:
                    st.subheader("Original Post 2")
                    st.write(df.iloc[1788,0])

            # Tabs for each column
        tab1, tab2, tab3, tab4, tab5 = st.tabs(['Keywords_Design_Elements', 'Keywords_Design_Style',
        'Keywords_Functionality', 'Keywords_Other','Product_Mention'])

        with tab1:
            st.header("Keywords_Design_Elements")
            n_values = list(range(5, 51, 5))  # Dropdown options: 5, 10, 15, ..., 50
            top_n = st.selectbox("Select Top N Keywords to Display", options=n_values, index=0,key = 'key1')
            display_column_data("Keywords_Design_Elements", top_n)

        with tab2:
            st.header("Keywords_Design_Style")
            n_values = list(range(5, 51, 5))  # Dropdown options: 5, 10, 15, ..., 50
            top_n = st.selectbox("Select Top N Keywords to Display", options=n_values, index=0,key = 'key2')
            display_column_data("Keywords_Design_Style", top_n)

        with tab3:
            st.header("Keywords_Functionality")
            n_values = list(range(5, 51, 5))  # Dropdown options: 5, 10, 15, ..., 50
            top_n = st.selectbox("Select Top N Keywords to Display", options=n_values, index=0,key = 'key3')
            display_column_data("Keywords_Functionality", top_n)
        with tab4:
            st.header("Keywords_Other")
            n_values = list(range(5, 51, 5))  # Dropdown options: 5, 10, 15, ..., 50
            top_n = st.selectbox("Select Top N Keywords to Display", options=n_values, index=0,key = 'key4')
            display_column_data("Keywords_Other", top_n)
        with tab5:
            st.header("Product_Mention")
            n_values = list(range(5, 51, 5))  # Dropdown options: 5, 10, 15, ..., 50
            top_n = st.selectbox("Select Top N Keywords to Display", options=n_values, index=0,key = 'key5')
            display_column_data("Product_Mention", top_n)
    elif st.session_state["selected_page"] == "during engagement":
        st.title("Page 2")
        st.write("Welcome to Page 2!")
        st.write("This is some content for Page 1.")
        # Initialize chatbot (set a system message)
        if "chatbot" not in st.session_state:
            # System message defines the chatbot's behavior
            st.session_state.chatbot = QWENapp(
                system_message="You are a helpful assistant. Respond succinctly to user queries."
            )

        # Initialize conversation history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []  # Store the conversation history
        if user_input := st.chat_input(placeholder="any question?"):

        # Chat interface
            # Append the user message to the conversation history
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Get response from the chatbot
            response = st.session_state.chatbot.ask(user_input)

            # Append the bot response to the conversation history
            st.session_state.messages.append({"role": "assistant", "content": response})
    # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.chat_message("user").write(message['content'])
            elif message["role"] == "assistant":
                st.chat_message("assistant").write(message['content'])
    elif st.session_state["selected_page"] == "after engagement":
        st.write('tbd')






        # Plot the frequencies

if __name__ == "__main__":
    main()