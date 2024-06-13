import streamlit as st
import openai
from openai import OpenAI
import requests,concurrent.futures,math
import json
import base64, subprocess,pickle
from io import BytesIO
import tempfile
import os, io, time,ast
import fitz
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from extract_ques import PDFQuestionExtractor
from Teachers_Answers import PDFTeacherAnswerExtractor
from functools import partial
import logging
logging.basicConfig(level=logging.ERROR)
from logger import logger

# api_key = os.environ['openai_api_key']
# azure_endpoint = os.environ['azure_endpoint']
# azure_key = os.environ['azure_key']


api_key = st.secrets['openai_api_key']
azure_endpoint = st.secrets['azure_endpoint']
azure_key = st.secrets['azure_key']



def process_question(analyze_function, headers, question, teacher_answer, temp_pdf_path, temp_dir):
    response = analyze_function(headers=headers, question=question, answer=teacher_answer, answers_pdf=temp_pdf_path, output_folder=temp_dir)
    fb = json.loads(response)
    print("fb-------------->",fb)
    return fb

def process_chunk(questions_chunk, answers_chunk, analyze_function, headers, temp_pdf_path, temp_dir):
    try:
        results = []
        print("Questions chunk given to model----->",questions_chunk)
        print("-------------------------")
        print("answers chunk given to model---------->",answers_chunk)

        #process_fn = partial(process_question, analyze_function, headers, temp_pdf_path=temp_pdf_path, temp_dir=temp_dir)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            
            
            futures = []
            for question, teacher_answer in zip(questions_chunk, answers_chunk):
                futures.append(executor.submit(process_question, analyze_function, headers, question, teacher_answer, temp_pdf_path, temp_dir,))
            #futures = executor.map(process_fn, questions_chunk, answers_chunk)

            feedback =[]
            for future in concurrent.futures.as_completed(futures):
                feedback.append(future.result())
            
            
            # for j in feedback:
            #     results.append(j)
            
                
        return feedback
    except Exception as e:
         st.write(logger.error("Error in process_chunk function"))
         logger.exception("Exception occurred: %s", e)
         print("Got error in process_chunk function")
         st.write("got error in process_chunk function")
         st.write(e)

def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

def extract_images_from_pdf(pdf_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = []
    pdf_document = fitz.open(pdf_path)
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        image_list = page.get_images(full=True)
        for image_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_extension = base_image["ext"]
            image_filename = f"{output_folder}/page{page_number+1}.{image_extension}"
            with open(image_filename, "wb") as image_file:
                image_file.write(image_bytes)
            image_paths.append(image_filename)
    return image_paths

def evaluation( feedback, topic, Question_Number):
        return json.dumps({"feedback": feedback, "topic": topic,"Question_Number": Question_Number})

@retry(wait=wait_random_exponential(min=10, max=60), stop=stop_after_attempt(10))
def send_request(headers, payload):
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response

def analyze_answers(headers, question, answer, answers_pdf, output_folder):
        image_paths = extract_images_from_pdf(answers_pdf, output_folder)
        base64_images = [encode_image(image_path) for image_path in image_paths]

        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": """ You are going to evaluate a student's handwritten answers against an answer sheet based on a provided question paper. Follow these steps to accurately complete the evaluation and provide constructive feedback:
             
    Carefully read through the Question_Paper to understand what is being asked in each question.
    Review the Answer_Sheet to understand the correct and expected answers for each question in the Question_Paper.
    Analyze each of the Student_Answers in images scene especially the MSQS are written in roman numbers, evaluate them carefully by comparing them to the corresponding Teachers Answers .
    If the question requires the diagram then carefully look at the students answers in images scene and evaluate.
                 
👉 (Remember): after taking the question number from the given question do not remove it. Give the complete question given to you.    
👉 (Remember) You should give the same question with question number at the starting of the question. (which contains question number, MaximumMarks and DiagramNeeded) as given to you. Because you don't have the right to give maximummarks to a question or anything.
             
    As a teacher Consider the following guidelines while consdering and evaluating the given question:
             
> Ignore Spelling Mistakes.
> Types of questions to consider:

   1. Compound Questions: If a question contains two or more parts, extract all of them as a single question.

    Example: "19. How is China rose indicator prepared? What colour change should you expect when this indicator is added to the sodium hydroxide solution?. (MaximumMarks:2, Diagram_Needed:No)" should be extracted as:
    "19. How is China rose indicator prepared? What colour change should you expect when this indicator is added to the sodium hydroxide solution?. (MaximumMarks:2, Diagram_Needed:No)"

    2. Multiple questions in a single question:
        example:"31. Rob started running towards the north, starting from his home. He covered 20 metres in 2 minutes. Next, he increased his speed and ran back 20 metres in 1 minute. Then, he turned north again and ran for 2 minutes with the starting speed, covering 20 metres in this direction. a) Draw a distance-time graph for Rob. b) What is the average velocity of Rob? ('MaximumMarks':4, 'Diagram_Needed':Yes)
                 When questions like above are given, you have to clearly serach for the question in it that needs diagram and you have to evaluate other parts with answer provided. 
        remember: you must give response.
                 
     
    Give the output that should contain Question,overall score, and following metrics:

    1)Accuracy: Determine the correctness of the information provided in the response. Are the facts and concepts presented accurate?

    2)Relevance: Assess whether the student's response directly addresses the question asked. Is the information provided relevant to the topic being discussed?

    3)Depth of Understanding: Evaluate the depth of the student's understanding of the topic. Does the response demonstrate a deep understanding of the concepts, theories, and principles involved?

    4)Completeness: Consider whether the response covers all aspects of the question comprehensively. Are all relevant points addressed, or are there significant omissions?

    5)Clarity of Expression: Evaluate the clarity and coherence of the student's writing. Is the response well-organized and easy to follow? Are ideas expressed clearly and concisely?

    6)Use of Examples: Assess whether the student provides relevant examples to support their arguments or explanations. Do the examples enhance understanding and illustrate key points effectively?

    7)Critical Thinking: Analyze the student's ability to critically evaluate and analyze information. Does the response demonstrate critical thinking skills, such as the ability to analyze, evaluate, and synthesize information?

    8)Originality: Consider whether the student offers original insights or perspectives on the topic. Does the response demonstrate creativity or independent thinking?

    9)Integration of Sources: If applicable, evaluate the student's ability to integrate information from multiple sources (e.g., textbooks, articles, lectures) into their response. Are sources properly cited and integrated into the discussion?

    10)Argumentation and Logic: Assess the strength of the student's arguments and the logical coherence of their reasoning. Are arguments well-supported with evidence and logical reasoning?
         
    11)Diagram(If needed for question) : Provide detailed feedback on student's handwritten diagram, explaining why it is correct or incorrect along with errors or improvements needed. 
         
    12)Overall Quality: Provide an overall assessment of the quality of the student's response. Consider all of the above metrics in conjunction to determine the overall effectiveness of the response.

    Make sure to provide all metrics.If not applicable ignore the specific metric by saying N/A. ensuring to include a newline character at the end to properly format your result.Display both Question and Answer written.

    Example response:
    "1. Question: What is the Role of roughages in the Alimentary tract. (MaximumMarks:2, Diagram_Needed:No)?
    Teacher_Answer: The main functions of roughages are: They help in movement of food through the alimentary canal as well as in proper bowel movement, Helps to prevent constipation and to get rid of undigested food, It also helps in retaining water in the body.       
    Student_Answer: Promoting Digestive Health, Aiding in weight management, Regulating Blood Sugar level, lowering cholesterol levels, preventing colon Cancer

    1) Accuracy:  High
    2) Relevance:  High
    3) Depth of Understanding:  Satisfactory
    5) Clarity of Expression: High
    6) Use of Examples: N/A
    7) Critical Thinking: N/A
    8) Originality: N/A
    9) Integration of Sources: N/A
    10) Argumentation and Logic: N/A
    11) Grammar and Mechanics: Satisfactory
    12) Diagram: Not Applicable.
    13) Feedback: Great job! Your answer accurately covers the roles of roughage but could benefit from more detail on how each role is performed.

    14) Overall Quality:  High
             
    Overall Score: 1.5/2
                    
 Note1:  For the value of topic you do not take the question. Topic name is the list of concepts and sub concepts from which the question is curated. And YOU MUST PROVIDE ALL THE METRIC VALUES and any string should not be unterminated!!!.
 Note2: You MUST USE TOOLS!. If you don't find the answer and diagram(if needed) it means student has not written the answer, then move to next question by saying the student has not written the answer. Also remember not to create new or new sentences from the students answers, extract them completely and evaluate it. If the question needs a diagram, do evaluate it and give feedback on the diagram as well.Don't use tools in parallel.  
                  
              """},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"YOU MUST ALWAYS CALL TOOLS!. Find the relevant answer for the question: {question} in the given image and give response accurately by taking referance of teacher answer:{answer['Teacher_Answer']}.In the response any string value of metrics should be terminated and provide all the delimiters required in the response. You should provide the complete question given to you with question number at the starting."},
                        *[
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            for base64_image in base64_images
                        ]
                    ]
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "evaluation",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "topic": {"type": "string"},
                                "Question_Number": {"type":"number"},
                                "feedback": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "question": {"type": "string"},
                                            "Teacher_Answer": {"type": "string"},
                                            "Student_Answer": {"type": "string"},
                                            "Accuracy": {"type": "string"},
                                            "Relevance": {"type": "string"},
                                            "Completeness": {"type": "string"},
                                            "Depth of Understanding": {"type": "string"},
                                            "Clarity of Expression": {"type": "string"},
                                            "Use of Examples": {"type": "string"},
                                            "Overall Quality": {"type": "string"},
                                            "No of words used": {"type": "number"},
                                            "feedback": {"type": "string"},
                                            "Diagram": {"type": "string"},
                                            "Overall Score": {"type": "number"},
                                        },
                                        "required": ["question", "Teacher_Answer", "Student_Answer", "Accuracy", "Relevance", "Completeness", "Depth of Understanding", "Clarity of Expression", "Use of Examples", "Diagram", "Overall Quality", "feedback", "No of words used", "Overall Score"]
                                    }
                                }
                            },
                            "required": ["topic", "feedback","Question_Number"]
                        }
                    }
                }
            ],
            "max_tokens": 1500,
            "temperature": 0,
            "top_p": 0.000000000000000001,
            "seed": 102,
            "tool_choice": "required"
        }

        response = send_request(headers=headers, payload=payload)
        response_json = response.json()
        response_message = response_json["choices"][0]["message"]
        tool_calls = response_message.get('tool_calls', [])

        if tool_calls:
            available_functions = {
                "evaluation": evaluation,
            }
            function_name = tool_calls[0]['function']['name']
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_calls[0]['function']['arguments'])
            return function_to_call(
                feedback=function_args.get("feedback"),
                topic=function_args.get("topic"),
                Question_Number=function_args.get("Question_Number")
            )
        else:
            return "No tool calls found in the response"
        
headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }


# class AI_Evaluation:
#     def __init__(self):
#         self.api_key = api_key
#         self.azure_endpoint = azure_endpoint
#         self.azure_key = azure_key
#         self.headers = {
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {self.api_key}"
#         }
#         self.static_directory = 'static'
#         self.ensure_static_directory_exists()

#     def ensure_static_directory_exists(self):
#         if not os.path.exists(self.static_directory):
#             os.makedirs(self.static_directory)

#     def encode_image(self, image_path):
#         with open(image_path, "rb") as image_file:
#             return base64.b64encode(image_file.read()).decode('utf-8')

#     def extract_images_from_pdf(self, pdf_path, output_folder):
#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder)

#         image_paths = []
#         pdf_document = fitz.open(pdf_path)
#         for page_number in range(len(pdf_document)):
#             page = pdf_document.load_page(page_number)
#             image_list = page.get_images(full=True)
#             for image_index, img in enumerate(image_list):
#                 xref = img[0]
#                 base_image = pdf_document.extract_image(xref)
#                 image_bytes = base_image["image"]
#                 image_extension = base_image["ext"]
#                 image_filename = f"{output_folder}/page{page_number+1}.{image_extension}"
#                 with open(image_filename, "wb") as image_file:
#                     image_file.write(image_bytes)
#                 image_paths.append(image_filename)
#         return image_paths

#     @retry(wait=wait_random_exponential(min=10, max=60), stop=stop_after_attempt(10))
#     def send_request(self, headers, payload):
#         response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
#         response.raise_for_status()
#         return response

#     def evaluation(self, feedback, topic):
#         return json.dumps({"feedback": feedback, "topic": topic})

#     def save_questions_to_file(self, questions, file_path):
#         with open(file_path, "w", encoding='utf-8') as file:
#             for question in questions:
#                 file.write(f"{question}\n")

#     def read_document_from_url(self,url):
#         response = requests.get(url)
#         response.raise_for_status()  # Ensure we raise an error for bad responses
#         return response
    
#     def chunkify(self, lst, n):
#         """Divide list into chunks of size n."""
#         return [lst[i:i + n] for i in range(0, len(lst), n)]
    

#     def ai_evaluate(self, questions_url, teachers_url, answers_url):
#         try:
#             temp_dir = tempfile.mkdtemp(prefix='output_Images_folder')

#             extractor = PDFQuestionExtractor(
#                 openai_api_key=self.api_key,
#                 azure_endpoint=self.azure_endpoint,
#                 azure_key=self.azure_key
#             )
#             listofquestions = extractor.analyze_read(questions_url)

#             extractor = PDFTeacherAnswerExtractor(openai_api_key=self.api_key, azure_endpoint=self.azure_endpoint, azure_key=self.azure_key)
#             listofanswers = extractor.analyze_read(teachers_url)

#             start = time.time()
#          #   answers_pdf = self.read_document_from_url(answers_url)

#             with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
#                 temp_pdf.write(answers_url.read())
#                 temp_pdf_path = temp_pdf.name

#             chunk_size = 8
#             question_chunks = self.chunkify(listofquestions, chunk_size)
#             print("Length of Questions_Chunks------------->",len(question_chunks))
#             answer_chunks = self.chunkify(listofanswers, chunk_size)
#             print("Lenth of Teachers_Answer Chunks----------->",len(answer_chunks))
            
#             # with concurrent.futures.ProcessPoolExecutor() as executor:
#             #     futures = []
#             #     for questions_chunk, answers_chunk in zip(question_chunks, answer_chunks):
#             #         futures.append(executor.submit(self.process_chunk, questions_chunk, answers_chunk, self.analyze_answers, self.headers, temp_pdf_path, temp_dir))
                
#             #     evaluation_sheet = []
#             #     for future in concurrent.futures.as_completed(futures):
#             #         evaluation_sheet.extend(future.result())

#             #     # evaluation_sheet now contains all feedback items
#             #     st.write(evaluation_sheet)

#             evaluation_sheet = []
#             for questions_chunk, answers_chunk in zip(question_chunks, answer_chunks):
#                 results = process_chunk(questions_chunk, answers_chunk, analyze_answers, self.headers, temp_pdf_path, temp_dir)
#                 print("Results----------->",results)
#                 st.write(results)
#                 evaluation_sheet.extend(results)

#            # results = process_chunk(listofquestions,listofanswers,analyze_answers,self.headers,temp_pdf_path, temp_dir)

#          #   evaluation_sheet now contains all feedback items
#             st.write("------------------------")
#          #   st.write(evaluation_sheet)

#             end = time.time()
#             st.write(f"Time taken to evaluate all the questions with vision model: {end-start} seconds")
#         except Exception as e:
#             st.error(f"An error occurred: {e}")
#             return e
        
def chunkify(lst, n):
        """Divide list into chunks of size n."""
        return [lst[i:i + n] for i in range(0, len(lst), n,)]       

def save_questions_to_file(questions, file_path):
        with open(file_path, "w", encoding='utf-8') as file:
            for question in questions:
                file.write(f"{str(question)}\n")

def sort_questions(questions):
     try:    
        # questions = questions.split('\n')
        # question_dicts = [ast.literal_eval(question) for question in questions]
        question_dicts = questions
        sorted_questions = sorted(question_dicts,key=lambda x:x['Question_Number'])
        for question in sorted_questions:
            if "Question_Number" in question:
                del question['Question_Number']
        logger.info("Sucessfully sorted the questions")
     except Exception as e:
          st.error(f"Error Occured in Sort function:{e}")
          logger.exception(f"Error Occured in Sort function:{e}")
          print(e)
     return sorted_questions
     
def convert_string_to_dict(list_of_strings):
    try:
        # Convert each string in the list to a dictionary
        list_of_dicts = [ast.literal_eval(item) for item in list_of_strings]
        logger.info("Successfully Convert the string to dict")
    except Exception as e:
        st.write(e)
        logger.exception(f"Error Occurred in conversion from string to dict: {e}")
        list_of_dicts = []  # Return an empty list in case of an error
    return list_of_dicts

def load_textfile(path):
     with open(path,'r') as file:
                      content = file.read()
     return content

def reset_app_state(list_of_questions,listofanswers):
    st.cache_data.clear()
    if os.path.exists(list_of_questions):
        os.remove(list_of_questions)
    if os.path.exists(listofanswers):
         os.remove(listofanswers)
    st.success("Cache cleared and temporary files removed!")  

def main():
    st.title("Paper Evaluation with GPT-4o")
    st.sidebar.title("Upload Files")
    questions_pdf = st.sidebar.file_uploader("**Upload Questions**", type=['pdf'])
    teachers_answers_pdf = st.sidebar.file_uploader("**Upload Teacher's Answers**", type=['pdf'])
    students_answers_pdf = st.sidebar.file_uploader("**Upload Student's Answers**", type=['pdf'])

    questions_file = "listofquestions.txt"
    answers_file = "listofanswers.txt"

    checkbox = st.sidebar.checkbox("Check Me to Evaluate Different Question Paper")

    if checkbox:
        reset_app_state(list_of_questions=questions_file,listofanswers=answers_file)
        
    if st.button("Evaluate"):
        if questions_pdf and teachers_answers_pdf and students_answers_pdf:
            try:
                temp_dir = tempfile.mkdtemp(prefix='output_Images_folder')

                if os.path.exists(questions_file) and os.path.exists(answers_file):
                     try:
                        listofquestions = load_textfile(path=questions_file)
                        listofanswers = load_textfile(path=answers_file)
                        listofquestions = listofquestions.strip().split('\n')
                        listofanswers = listofanswers.strip().split('\n')
                        listofanswers = convert_string_to_dict(listofanswers)
                        logger.info("List of questions and answers found in path")
                     except Exception as e:
                          st.write(e)
                          logger.error("Occured error while loading the questions")
                          logger.exception(e)
                    
                else:
                    extractor = PDFQuestionExtractor(
                        openai_api_key= api_key,
                        azure_endpoint=azure_endpoint,
                        azure_key=azure_key
                    )
                    listofquestions = extractor.analyze_read(questions_pdf)
                    extractor = PDFTeacherAnswerExtractor(openai_api_key=api_key, azure_endpoint=azure_endpoint, azure_key=azure_key)
                    listofanswers = extractor.analyze_read(teachers_answers_pdf)
                 
                
                st.write("Now Paper Evaluation is Started. You will get to see Marks Very soon. Thanks for the Patience...")
                start = time.time()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                    temp_pdf.write(students_answers_pdf.read())
                    temp_pdf_path = temp_pdf.name
                
                 

                print("List of questions------->",listofquestions)
                print("List of answers--------->",listofanswers)
                chunk_size = 8
                question_chunks = chunkify(listofquestions, chunk_size)
                print("Length of Questions_Chunks------------->",len(question_chunks))
                answer_chunks = chunkify(listofanswers, chunk_size)
                print("Lenth of Teachers_Answer Chunks----------->",len(answer_chunks))
                
                
                filename = 'Evluation_sheet2_with_parallelprocessing.txt'
                evaluation_sheet = []
                for questions_chunk, answers_chunk in zip(question_chunks, answer_chunks):
                    results = process_chunk(questions_chunk, answers_chunk, analyze_answers, headers, temp_pdf_path, temp_dir)
                    print("Results----------->",results)
                    un_sorted = sort_questions(results)
                    for result in un_sorted:  
                        st.write(result)
                        evaluation_sheet.append(result)
                    save_questions_to_file(evaluation_sheet,file_path=filename)
                    logger.info(f"Questions saved to {filename}")                                                                                         
 
                st.write("Skip the Checkbox at sidebar, if you want to evalute another student answers for the same question paper")
                 
               # st.write(f"Time taken to evaluate all the questions with vision model: {end-start} seconds")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.exception(e)
                return e

        else:
            st.warning("Please upload all the required files.")

if __name__ == "__main__":
    main()
