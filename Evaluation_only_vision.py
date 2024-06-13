import base64,os,fitz,json,logging,requests,tempfile,openai
from tenacity import retry,stop_after_attempt,wait_random_exponential
logging.basicConfig(level=logging.WARNING)
import streamlit as st
from extract_questions_for_vision import PDFQuestionExtractor
import concurrent.futures
from logger import logger

# api_key = os.environ['openai_api_key']
# azure_endpoint = os.environ['azure_endpoint']
# azure_key = os.environ['azure_key']

api_key = st.secrets['openai_api_key']
azure_endpoint = st.secrets['azure_endpoint']
azure_key = st.secrets['azure_key']

class AI_Evaluate:
    def __init__(self,api_key,azure_key,azure_endpoint):
        self.headers =  {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"}
        self.api_key = api_key
        self.azure_key = azure_key
        self.azure_endpoint=azure_endpoint
        
    def encode_image(self,image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    def extract_images_from_pdf(self,pdf_path, output_folder):
        # Create output folder if it does not exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # List to store paths of extracted images
        image_paths = []

        # Open the PDF file
        pdf_document = fitz.open(pdf_path)

        # Iterate through each page of the PDF
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)

            # Get the images on the page
            image_list = page.get_images(full=True)

            # Extract and save each image
            for image_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_extension = base_image["ext"]
                image_filename = f"{output_folder}/page{page_number+1}.{image_extension}"

                with open(image_filename, "wb") as image_file:
                    image_file.write(image_bytes)
                
                # Append the image path to the list
                image_paths.append(image_filename)
        
        # Return the list of image paths
        return image_paths
    
    def evaluation(self,feedback, topic):
        return json.dumps({"feedback": feedback, "topic": topic})
    
    

    def ai_evaluate(self,questions_pdf,answers_pdf):
        try:
            temp_dir = tempfile.mkdtemp(prefix='output_Images_folder')

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            Questions_extractor = PDFQuestionExtractor(
                    openai_api_key=self.api_key,
                    azure_endpoint=self.azure_endpoint,
                    azure_key=self.azure_key
                )
            list_of_questions = Questions_extractor.analyze_read(questions_pdf)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(answers_pdf.read())
                temp_pdf_path = temp_pdf.name

            def save_questions_to_file(questions, file_path):
                with open(file_path, "w",encoding='utf-8') as file:
                    for question in questions:
                        file.write(f"{question}\n")

            questions_chunks = chunkify(list_of_questions,chunk_size=8)
            for chunk in questions_chunks:
                results = process_chunk(analyze_function=Analyize_answers,question_chunk=chunk,temp_pdf_path=temp_pdf_path,temp_dir=temp_dir)
                sorted = sort_questions(results)
                for question in sorted:
                    st.write(question) 

            evaluation_sheet = []
            st.write("👉 Evaluation has started! you will get to see your marks soon...")
            for question in list_of_questions:
                question = question.strip()
                print("Question_strip---------->",question)
                response = self.Analyize_answers( question=question, answers_pdf=temp_pdf_path, OUTPUT_FOLDER=temp_dir)
                fb = json.loads(response)
                for correction in fb['feedback']:
                    evaluation_sheet.append(correction)
                    st.write(correction)
                    
                file_path = 'evaluation_sheet7_with_vision.txt'
                save_questions_to_file(evaluation_sheet, file_path)
                print(f"Questions saved to {file_path}")
            st.write("Paper Evaluation is done! ✅")
            #return evaluation_sheet
                   
             
        except Exception as e:
            logging.error("Error in main: %s", e)
            st.error(f"An error occurred: {e}")

headers =  {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"}

def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


def extract_images_from_pdf(pdf_path, output_folder):
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # List to store paths of extracted images
    image_paths = []

    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Iterate through each page of the PDF
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)

        # Get the images on the page
        image_list = page.get_images(full=True)

        # Extract and save each image
        for image_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_extension = base_image["ext"]
            image_filename = f"{output_folder}/page{page_number+1}.{image_extension}"

            with open(image_filename, "wb") as image_file:
                image_file.write(image_bytes)
            
            # Append the image path to the list
            image_paths.append(image_filename)
    
    # Return the list of image paths
    return image_paths
    
def evaluation(feedback, topic, Question_Number):
    return json.dumps({"feedback": feedback, "topic": topic, "Question_Number": Question_Number})

def Analyize_answers(question, answers_pdf, OUTPUT_FOLDER):
        logging.info("Starting analysis for question: %s", question)

        # Convert PDF to images
        image_paths = extract_images_from_pdf(answers_pdf, output_folder=OUTPUT_FOLDER)
        logging.debug("Extracted images: %s", image_paths)

        # Encode images to base64
        base64_images = [encode_image(image_path) for image_path in image_paths]
        logging.debug("Encoded images to base64")

        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": """You are a teacher who evaluates students answers and give marks based on the MaximumMarks provided for the question and Feedback on the given data.Ignore Spelling Mistakes.

        👉 As a teacher, your primary responsibility is to evaluate the students' work. You don't need to solve the problems yourself; simply review and assess the answers they have provided.
        👉 (Remember): after taking the question number from the given question do not remove it. Give the complete question given to you.    
                    
        As a teacher Consider the following guidelines while consdering and evaluating the given question:

    1. Compound Questions: If a question contains two or more parts, extract all of them as a single question.

        Example: "19. How is China rose indicator prepared? What colour change should you expect when this indicator is added to the sodium hydroxide solution?. (MaximumMarks:2, Diagram_Needed:No)" should be extracted as:
        "19. How is China rose indicator prepared? What colour change should you expect when this indicator is added to the sodium hydroxide solution?. (MaximumMarks:2, Diagram_Needed:No)"

        2. Multiple questions in a single question:
            example:"31. Rob started running towards the north, starting from his home. He covered 20 metres in 2 minutes. Next, he increased his speed and ran back 20 metres in 1 minute. Then, he turned north again and ran for 2 minutes with the starting speed, covering 20 metres in this direction. a) Draw a distance-time graph for Rob. b) What is the average velocity of Rob? ('MaximumMarks':4, 'Diagram_Needed':Yes)
                    When questions like above are given, you have to clearly serach for the question in it that needs diagram and you have to evaluate other parts with answer provided. 
            remember: you must give response.
                    
        👉 You should give the same question (which contains question number, MaximumMarks and DiagramNeeded) as given to you. Because you don't have the right to give maximummarks to a question or anything.
            
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

        11)Grammar and Mechanics: Consider the student's use of grammar, punctuation, and spelling. Are there any grammatical errors or typos that affect the clarity of the response?
            
        12)Diagram(If needed for question) : Provide detailed feedback on student's handwritten diagram, explaining why it is correct or incorrect along with errors or improvements needed.
            
        13)Overall Quality: Provide an overall assessment of the quality of the student's response. Consider all of the above metrics in conjunction to determine the overall effectiveness of the response.

        Make sure to provide all metrics.If not applicable ignore the specific metric by saying N/A. ensuring to include a newline character at the end to properly format your result.Display both Question and Answer written.

        Example response:
        "1. Question: What is the Role of roughages in the Alimentary tract. (MaximumMarks:2, Diagram_Needed:No)?
        Answer: Promoting Digestive Health, Aiding in weight management, Regulating Blood Sugar level, lowering cholesterol levels, preventing colon Cancer

        Overall Score: 1.5/2

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
                 
    👉  Additionally, remember: for two different questions, there can't be same answer in the images provided. If you evaluated one question initially with one answer, then this answer will not be the answer to any question given next. And also do not provide overall score as 0.2,0.7,2.2,3.8 you should always provide in terms of o.5 or full marks like 1.5,0.5,0,2.5 like this.
                 
    Note1:  For the value of topic you do not take the question. topic will be the concept from which the question is curated. And YOU MUST PROVIDE ALL THE METRIC VALUES and any string should not be unterminated!!!.
    Note2: You MUST USE TOOLS!. If you don't find the answer and diagram(if needed) it means student has not written the answer, then move to next question by saying the student has not written the answer. Also remember not to create new or new sentences from the students answers, extract them completely and evaluate it. If the question needs a diagram, do evaluate it and give feedback on the diagram as well.Don't use tools in parallel.   
                """},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"YOU MUST ALWAYS CALL TOOLS!. Find the relevant answer for the question: {question} in the given image and give response accurately.sometimes student writes the answer in next image in continution so do not consider that complete answer present in one image only, consider the next image if answer is there or not in continution.And also you must provide all the metric values specially: overall score for every question given to you!. Additionally, remember: for two different questions, there can't be same answer in the images provided. If you evaluated one question initially with one answer, then this answer will not be the answer to any question given next. And also do not provide overall score as 0.2,0.7,2.2,3.8 you should always provide in terms of o.5 or full marks like 1.5,0.5,0,2.5 like this."},
                        *[
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}","detail": "high"}}
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
                    "topic": {
                        "type": "string"
                    },
                    "Question_Number": {
                        "type": "number"
                    },
                    "feedback": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string"
                                },
                                "answer": {
                                    "type": "string"
                                },
				"Accuracy": {
                                    "type": "string"
                                },
				"Relevance": {
                                    "type": "string"
                                },
				"Completeness": {
                                    "type": "string"
                                },
				"Depth of Understanding": {
                                    "type": "string"
                                },
				"Clarity of Expression": {
                                    "type": "string"
                                },
				"Use of Examples": {
                                    "type": "string"
                                },
				"Grammar and Mechanics": {
                                    "type": "string"
                                },
				"Overall Quality": {
                                    "type": "string"
                                },
				"No of words used": {
                                    "type": "number"
                                },
                      "feedback" : {
                                    "type":"string"
                                },
                        "Diagram":{
                                    "type":"string"
                        },
                                "Overal Score": {
                                    "type": "number",
                 
                                },                            },
                            "required": ["question", "answer","Accuracy","Relevance","Completeness","Depth of Understanding","Clarity of Expression","Use of Examples","Grammar and Mechanics","Diagram","Overall Quality","feedback","No of words used","Overal Score"]
                        }
                    }
                },
                "required": ["topic", "Question_Number" , "feedback"]
            }
        }
        }
    ],
            "max_tokens": 1500,
            "temperature":0,
            "top_p":0.000000000000000001,
            "seed":92,
            "tool_choice":"required"
        }
        
        
        @retry(wait=wait_random_exponential(min=10, max=60), stop=stop_after_attempt(10))
        def exponential_backoff(headers, payload):
            logging.info("Sending request to OpenAI API")
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()  # Ensure we raise an error for bad responses
            logging.info("Received response from OpenAI API")
            return response

        try:
            response = exponential_backoff(headers=headers, payload=payload)
        except requests.RequestException as e:
            logging.error("Request failed: %s", e)
            raise e

        logging.info("Processing API response")
        response_json = response.json()
        logging.debug("API response JSON: %s", response_json)
        response_message = response_json["choices"][0]["message"]
        tool_calls = response_message.get('tool_calls', [])

        if tool_calls:
            logging.info("Tool calls found in response")
            # Step 3: call the function
            available_functions = {
                "evaluation": evaluation,
            }
            function_name = tool_calls[0]['function']['name']
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_calls[0]['function']['arguments'])
            function_response = function_to_call(
                feedback=function_args.get("feedback"),
                topic=function_args.get("topic"),
                Question_Number = function_args.get("Question_Number")
            )
            logging.info("Function call result: %s", function_response)
            return function_response
        else:
            logging.warning("No tool calls found in the response")

def chunkify(lst,chunk_size:int):
    chunks =[]
    for i in range(0,len(lst),chunk_size):
        chunks.append(lst[i:i+chunk_size])

    return chunks

def process_question(Analyze_function,question,temp_pdf_path,temp_dir,):
    response = Analyze_function( question=question, answers_pdf=temp_pdf_path, OUTPUT_FOLDER=temp_dir)
    fb = json.loads(response)
    print("fb-------------->",fb)
    return fb

def process_chunk(analyze_function,question_chunk,temp_pdf_path,temp_dir):
    print("Given Questions chunk --->",question_chunk)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures =[]
        for question in question_chunk:
            futures.append(executor.submit(process_question,analyze_function,question,temp_pdf_path,temp_dir))

        feedback =[]
        for future in concurrent.futures.as_completed(futures):
            feedback.append(future.result())
        
        return feedback

def sort_questions(questions):
     try:    
        # questions = questions.split('\n')
        # question_dicts = [ast.literal_eval(question) for question in questions]
        question_dicts = questions
        sorted_questions = sorted(question_dicts,key=lambda x:x['Question_Number'])
        # for question in sorted_questions:
        #     if "Question_Number" in question:
        #         del question['Question_Number']
        logger.info("Sucessfully sorted the questions")

     except Exception as e:
          st.error(f"Error Occured in Sort function:{e}")
          logger.exception(f"Error Occured in Sort function:{e}")
          print(e)
     return sorted_questions

def load_textfile(path):
     with open(path,'r') as file:
                      content = file.read()
     return content

def reset_app_state(list_of_questions):
    st.cache_data.clear()
    if os.path.exists(list_of_questions):
        os.remove(list_of_questions)
    st.success("Cache cleared and temporary files removed!")

if __name__ == "__main__":
    st.title("Paper Evaluation with GPT-4o")
    st.sidebar.title("Upload Files")
    questions_pdf = st.sidebar.file_uploader("**Upload Questions**", type=['pdf'])
    answers_pdf = st.sidebar.file_uploader("**Upload Answers**", type=['pdf'])

    checkbox = st.sidebar.checkbox("Check Me to Evaluate Different Question Paper")
    list_of_questions = "list_of_questions_for_vision.txt"

    if checkbox:
        reset_app_state(list_of_questions=list_of_questions)
     

    # Button to trigger evaluation
     

    if st.button("Evaluate"):
        if questions_pdf and answers_pdf:
            # evaluate = AI_Evaluate(api_key=api_key,azure_endpoint=azure_endpoint,azure_key=azure_key)
            # evaluate.ai_evaluate(questions_pdf=questions_pdf,answers_pdf=answers_pdf)

            try:
                temp_dir = tempfile.mkdtemp(prefix='output_Images_folder')

                list_of_questions = "list_of_questions_for_vision.txt"


                if os.path.exists(list_of_questions):
                    list_of_questions = load_textfile(list_of_questions)
                    list_of_questions = list_of_questions.strip().split('\n')
                    logger.info("list of questions found in path")
                
                else:

                    Questions_extractor = PDFQuestionExtractor(
                            openai_api_key=api_key,
                            azure_endpoint=azure_endpoint,
                            azure_key=azure_key
                        )
                    list_of_questions = Questions_extractor.analyze_read(questions_pdf)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                    temp_pdf.write(answers_pdf.read())
                    temp_pdf_path = temp_pdf.name

                def save_evaluated_answers(questions,file_path):
                    with open(file=file_path,mode='w',encoding='utf-8') as file:
                        for question in questions:
                            file.write(f"{str(question)}\n")


                st.write("Now Paper Evaluation is Started. You will get to see Marks Very soon. Thanks for the Patience...")
                questions_chunks = chunkify(list_of_questions,chunk_size=8)
                evaluated_sheet =[]
                for chunk in questions_chunks:
                    results = process_chunk(analyze_function=Analyize_answers,question_chunk=chunk,temp_pdf_path=temp_pdf_path,temp_dir=temp_dir)
                    print("results------>",results)
                    sorted_ques = sort_questions(results)
                    for question in sorted_ques:
                        st.write(question) 
                        evaluated_sheet.append(question)
                    file_name = 'Evaluation_sheet_without_teacher_solutions_using_GPT4o.txt'
                    save_evaluated_answers(file_path=file_name,questions=evaluated_sheet)
                    logger.info(f"Saved evaluated sheet to:{file_name}")
                st.write("Paper Evaluation is done ✅")
                st.write("Skip the Checkbox at sidebar, if you want to evalute another student answers for the same question paper")
                 

            except Exception as e:
                st.write(e)
                logger.exception(e)
           
        else:
            st.warning("Please upload both questions and answers files.")
