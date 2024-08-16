import PyPDF2
import requests
import openai
import os
import streamlit as st
import json
import random
import re
import requests

from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv(os.path.expanduser('~/.env'))

os.environ["OPENAI_API_KEY"]= st.secrets["OPENAI_API_KEY"]
## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Exam paper Generator"



def extract_json_from_response(response_text):
    # Regular expression to find JSON-like structure in the response text
    json_pattern = r'\{.*\}|\[.*\]'
    response_text = response_text.replace("```json", "").replace("```", "").strip()
    if imageQuestions:
        response_text = response_text.split("**Note**")[0].strip()
    
    # Search for the JSON block in the response text
    match = re.search(json_pattern, response_text, re.DOTALL)
    
    if match:
        json_str = match.group(0)
        return json_str
    else:
        return None

def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text(separator=" ", strip=True)

questions_type_contest = "Contest Trivia"
questions_type_survey = "Survey"

st.title("Contest Survey Question Generator")
## Sidebar for settings
st.sidebar.title("Settings")

## Select the OpenAI model
engine=st.sidebar.selectbox("Select Open AI model",["gpt-4o-mini","gpt-4o","gpt-4-turbo","gpt-4"])

## Adjust response parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
questionType = st.sidebar.radio('Choose an option:',[questions_type_contest, questions_type_survey])
question_count=st.sidebar.slider("Number of Questions",min_value=1,max_value=10,value=5)
imageQuestions = st.sidebar.checkbox('Image questions')


pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
url_input = st.text_input("Enter a web URL")




def get_image_urls_for_options(text):
    # Create the prompt to get image URLs for each option
    prompt = f"""
    For the following question, provide an image URL for each option:

    Question: {text}

    Options:
    {text}
    """

    try:
        # Make the API call
        response = openai.chat.completions.create(
            model=engine,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        )
        
        # Extract the response
        image_urls_text = response.choices[0].message.content.strip()
        
        # Parse the response to get image URLs
        image_urls = {}
        lines = image_urls_text.split('\n')
        for line in lines:
            if line.strip():
                option, url = line.split(":", 1)
                image_urls[option.strip()] = url.strip()
        
        return image_urls
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def generate_questions_answers(content):
    # Define possible options
     options_labels = ["A", "B", "C", "D"]
    
    # Shuffle the options labels to randomize the correct option's position
     shuffled_labels = random.sample(options_labels, len(options_labels))
     prompt = ""

     if imageQuestions:
          prompt = (
                    f"Generate {question_count} question(s) with four objective-type options (A, B, C, D) and each option is represented by an image URL."
                    "The images should be related to the options and found from the internet. Each question should have one correct answers and the remaining options should be incorrect. The correct answers should be randomly assigned to any of the options (A, B, C, or D). Format the output as JSON with the question, the four options, and the correct options. Ensure the options are in random order and the correct answers are not always in the same positions.\n\n"
                    "Content:\n"
                    f"{content}\n\n"
                    "Example JSON format:\n"
                    "[\n"
                    "  {\n"
                    '    "question": "Which of the following are characteristics of a smart home system?",\n'
                    '    "options": {\n'
                    '      "A": "https://example.com/image1.jpg",\n' # Image representing the correct answer
                    '      "B": "https://example.com/image2.jpg",\n' # Image of an incorrect answer
                    '      "C": "https://example.com/image3.jpg",\n' # Image of an incorrect answer
                    '      "D": "https://example.com/image4.jpg"\n' # Image of an incorrect answer
                    "    },\n"
                    '    "correct_options": "A",\n'  # The labels of the correct answers
                    "  }\n"
                    "]"
                    )
     elif questionType == questions_type_contest:
        prompt = (
                        f"Generate {question_count} question(s) with four objective-type options (A, B, C, D) based on the following content. "
                    "Each question should have one correct answers and the remaining options should be incorrect. The correct answers should be randomly assigned to any of the options (A, B, C, or D). Format the output as JSON with the question, the four options, and the correct options. Ensure the options are in random order and the correct answers are not always in the same positions.\n\n"
                    "Content:\n"
                    f"{content}\n\n"
                    "Example JSON format:\n"
                    "[\n"
                    "  {\n"
                    '    "question": "Which of the following are characteristics of a smart home system?",\n'
                    '    "options": {\n'
                    '      "A": "Automated lighting",\n'   # Correct answer
                    '      "B": "Manual heating control",\n' # Incorrect answer
                    '      "C": "Remote security monitoring",\n' # Correct answer
                    '      "D": "Traditional thermostat"\n' # Incorrect answer
                    "    },\n"
                    '    "correct_options": "A",\n'  # The labels of the correct answers
                    "  }\n"
                    "]"
                )
     else:
        prompt = (
                        f"Generate {question_count} question(s) with four objective-type options (A, B, C, D) based on the following content. "
                    "Each question may have one or more correct answers and the remaining options should be incorrect. The correct answers should be randomly assigned to any of the options (A, B, C, or D). Format the output as JSON with the question, the four options, and the correct options. Ensure the options are in random order and the correct answers are not always in the same positions.\n\n"
                    "Content:\n"
                    f"{content}\n\n"
                    "Example JSON format:\n"
                    "[\n"
                    "  {\n"
                    '    "question": "Which of the following are characteristics of a smart home system?",\n'
                    '    "options": {\n'
                    '      "A": "Automated lighting",\n'   # Correct answer
                    '      "B": "Manual heating control",\n' # Incorrect answer
                    '      "C": "Remote security monitoring",\n' # Correct answer
                    '      "D": "Traditional thermostat"\n' # Incorrect answer
                    "    },\n"
                    '    "correct_options": ["A", "C"]\n'  # The labels of the correct answers
                    "  }\n"
                    "]"
                )
          
     try:
        response = openai.chat.completions.create(
            model=engine,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            # max_tokens=50000  # Increase this if the content is too long
        )
        message_content = response.choices[0].message.content.strip()
        if not message_content:
            st.error("The response from the model was empty. Try refining your prompt.")
            return None
        return message_content
     
     except Exception as e:
        st.error(f"An error occurred: {e}")
        return None
     
     

if st.button("Generate Questions"):
    if pdf_files or url_input:
        combined_text = ""

        if pdf_files:
            pdf_text = extract_text_from_pdfs(pdf_files)
            combined_text += pdf_text

        if url_input:
            url_text = extract_text_from_url(url_input)
            combined_text += url_text

        if combined_text:
            qa_pairs = generate_questions_answers(combined_text)
            qa_pairs = extract_json_from_response(qa_pairs)

        # Format into JSON
            try:
                st.json(qa_pairs)
            except Exception as e:
                st.write(qa_pairs)
            
            # urls = get_image_urls_for_options(qa_pairs)
            # print("Image URLs:{urls}")
         
            # Option to download the JSON
            st.download_button("Download JSON", qa_pairs, "questions_answers.json", "application/json")
        else:
            st.error("No content to generate questions from.")
    else:
        st.error("Please upload a PDF or provide a URL.")


def getImageFromDalle(text):
    response = openai.images.generate(
                model="dall-e-3",
                prompt="a white siamese cat",
                size="1024x1024",
                quality="standard",
                n=1,
                )

    image_url = response.data[0].url
