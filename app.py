import os
import fitz  # PyMuPDF
from openai import AzureOpenAI
import json
from flask_cors import CORS
from flask import Flask, request, Response, jsonify
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import pytesseract
from PIL import Image
from werkzeug.utils import secure_filename
import re
import pycountry
# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Handle CORS
# Azure OpenAI Client Initialization
client = AzureOpenAI(
    api_key="3c3384effe084ff3b56101ab0c1d14df",
    api_version="2024-05-01-preview",
    azure_endpoint="https://confitech.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-02-15-preview"
)

deployment_name = 'gpt-4'

# Configurations for file upload
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size of 16 MB
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text += page.get_text()
    return text




def allFieldsExtractor(content):
    print('Sending request to extract invoice details')

    chat_prompt = [{
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": """You are an AI assistant that helps extract invoice details. Given an invoice data converted from a PDF, extraxt all the details from the Invoice and return the output in valid JSON format.
                The Invoice data can be English or Arabic. So return accordingly.
                The output should consists of all the important values from the PDF extatacted and also return the line items from the invoice.
                DO NOT GENERATE anything Outside the Context.Give Only The values from the PDF content.

                Make sure to format your response as a JSON object.
                JUST RETURN THE JSON DATA"""
            }
        ]
    }]

    messages = chat_prompt + [{
        "role": "user",
        "content": content
    }]
    
    # API call to OpenAI GPT-4 model
    try:
        completion = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            max_tokens=3000,
            temperature=0.3,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        return None





# Function to extract invoice details using OpenAI
def extractInvoiceDetailsOpenAI(content):
    print('Sending request to extract invoice details')

    chat_prompt = [{
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": """You are an AI assistant that helps extract invoice details. Given an invoice data converted from a PDF, extract the following details and return the output in valid JSON format. 
                FIELDS TO EXTRACT ARE : 
                1. Invoice Number
                2. Invoice Dt
                3. Claim No
                4. Name
                5. AK no
                6. Address
                7. Consultant
                8. Gender
                9. Policy number
                10. Entitlement
                11. Organization
                12. List of all the invoice line items with their Date of Service, Procedure, Description, Provider, Days/Units, Rate, Amount.
                Make sure to format your response as a JSON object.
                JUST RETURN THE JSON DATA"""
            }
        ]
    }]

    messages = chat_prompt + [{
        "role": "user",
        "content": content
    }]
    
    # API call to OpenAI GPT-4 model
    try:
        completion = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            max_tokens=3000,
            temperature=0.1,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        return None



def extractInvoiceHighlightsOpenAI(content):
    print('Sending request to extract invoice Highlights')

    # Modify the system message to guide the model on what to do
    chat_prompt = [{
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": """You are an AI assistant that helps extract invoice details. Given an invoice data converted from a PDF, extract the following insights and return the output in Crisp Bullet Points. 
                Insights to be Extracted.
                1. Product having highest quantities from the Invoice.
                2. Most Valuable product in the Invoice.
                3. Total Bill Value is more than 5000
                
                

                Make sure to format your response as a Bullet Points Below.
                Example :
                Most Selling Product :
                Most Valuable Product :
                Bill value more than 5000 : True/ False 

                JUST RETURN ONLY IF YOU FIND THE DATA, DO NOT MAKE UP YOUR ANSWER , DO NOT ANSWER ANYTHING OUT OF CONTEXT}"""
            }
        ]
    }]
    
    # Add the invoice content to the user message
    messages = chat_prompt + [{
        "role": "user",
        "content": content
    }]
    
    # Make the API call to OpenAI GPT-4 model
    completion = client.chat.completions.create(  
        model=deployment_name,
        messages=messages,
        max_tokens=3000,  
        temperature=0.1,  
        top_p=0.95,  
        frequency_penalty=0,  
        presence_penalty=0,
        stop=None,  
        stream=False
    )
    return (completion.choices[0].message.content)
    #Extract response and print as JSON
    # try:
    #     # Accessing the message content from the response
    #     json_response = completion.choices[0].message.content
        
    #     # Parse the response as JSON
    #     import json
    #     parsed_json = json.loads(json_response)
    #     return parsed_json
    # except Exception as e:
    #     print(f"Error parsing response: {e}")
    #     return None


# Route to handle PDF or image upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF or image files are allowed'}), 400

    # Ensure upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # Save file securely
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # Check file type
        if filename.lower().endswith('.pdf'):
            extracted_text = extract_text_from_pdf(file_path)
        else:
            image = Image.open(file_path)
            extracted_text = pytesseract.image_to_string(image)
    except Exception as e:
        return jsonify({'error': f'Error extracting text: {str(e)}'}), 500

    # Send text to OpenAI
    try:
        data = extractInvoiceDetailsOpenAI(extracted_text)
        if data:
            json_str = data.strip().replace("```json\n", "").replace("\n```", "")
            parsed_json = json.loads(json_str)
            return jsonify(parsed_json)
        else:
            return jsonify({'error': 'No valid response from OpenAI'}), 500
    except Exception as e:
        return jsonify({'error': f'Error processing data: {str(e)}'}), 500



# Function to extract claim details using OpenAI
def extractClaimDetailsOpenAI(content):
    print('Sending request to extract Claim details')

    chat_prompt = [{
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": """You are an AI assistant that helps extract Insurance Claim details. Given an Claim data converted from a PDF, extract the following details and return the output in valid JSON format. 
                FIELDS TO EXTRACT ARE : 
                1. Member Name
                2. Name
                3. Policy Number
                4. Membership No
                5. Age
                6. Relation
                7. Enrolment Date
                8. Enrolment From Date
                9. Enrolment To Date
                10. Treatment Department
                11. Receive Date
                12. Service Date
                13. Contact Number 1
                14. Contact Number 2
                JUST RETURN THE JSON DATA"""
            }
        ]
    }]

    messages = chat_prompt + [{
        "role": "user",
        "content": content
    }]
    
    # API call to OpenAI GPT-4 model
    try:
        completion = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            max_tokens=3000,
            temperature=0.1,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        return None



@app.route('/claimupload', methods=['POST'])
def claim_upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF or image files are allowed'}), 400

    # Ensure upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # Save file securely
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # Check file type
        if filename.lower().endswith('.pdf'):
            extracted_text = extract_text_from_pdf(file_path)
        else:
            image = Image.open(file_path)
            extracted_text = pytesseract.image_to_string(image)
    except Exception as e:
        return jsonify({'error': f'Error extracting text: {str(e)}'}), 500

    # Send text to OpenAI
    try:
        data = extractClaimDetailsOpenAI(extracted_text)
        if data:
            json_str = data.strip().replace("```json\n", "").replace("\n```", "")
            parsed_json = json.loads(json_str)
            return jsonify(parsed_json)
        else:
            return jsonify({'error': 'No valid response from OpenAI'}), 500
    except Exception as e:
        return jsonify({'error': f'Error processing data: {str(e)}'}), 500



# # Route to handle PDF upload and query OpenAI API
# @app.route('/upload', methods=['POST'])
# def upload_pdf():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     # Ensure the file is a PDF
#     if not file.filename.lower().endswith('.pdf'):
#         return jsonify({'error': 'Only PDF files are allowed'}), 400
    
#     # Create an uploads folder if it doesn't exist
#     if not os.path.exists(app.config['UPLOAD_FOLDER']):
#         os.makedirs(app.config['UPLOAD_FOLDER'])

#     # Save the file temporarily
#     pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     file.save(pdf_path)
    
#     # Extract text from the PDF
#     try:
#         pdf_text = extract_text_from_pdf(pdf_path)
#     except Exception as e:
#         return jsonify({'error': f"Error extracting text from PDF: {str(e)}"}), 500
    
#     # Send the extracted text to OpenAI API to fetch details
#     try:
#         data = extractInvoiceDetailsOpenAI(pdf_text)
#         if data:
#             json_str = data.strip().replace("```json\n", "").replace("\n```", "")
#             parsed_json = json.loads(json_str)
#             return jsonify(parsed_json)
#         else:
#             return jsonify({'error': 'No valid response from OpenAI'}), 500
#     except Exception as e:
#         return jsonify({'error': f"Error processing data: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def greet():
    return("Hello Welcome to Confitech AI")

@app.route('/success', methods = ['POST'])   
def success():   
    if request.method == 'POST':   
        f = request.files['file'] 
        return(f.filename)

@app.route('/getinsights', methods=['POST'])
def getinsights():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the file temporarily
    pdf_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(pdf_path)
    
    # Extract text from the PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Send the extracted text to OpenAI API to fetch details (e.g., summary)
    try:
        data=extractInvoiceHighlightsOpenAI(pdf_text)
        # json_str = data.strip().replace("```json\n", "").replace("\n```", "")
        # parsed_json = json.loads(json_str)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/masterfields', methods=['POST'])
def getmasterfields():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Ensure the file is a PDF
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    # Create an uploads folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # Save the file temporarily
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(pdf_path)
    
    # Extract text from the PDF
    try:
        pdf_text = extract_text_from_pdf(pdf_path)
    except Exception as e:
        return jsonify({'error': f"Error extracting text from PDF: {str(e)}"}), 500
    
    # Send the extracted text to OpenAI API to fetch details
    try:
        data = allFieldsExtractor(pdf_text)
        if data:
            json_str = data.strip().replace("```json\n", "").replace("\n```", "")
            parsed_json = json.loads(json_str)
            return jsonify(parsed_json)
        else:
            return jsonify({'error': 'No valid response from OpenAI'}), 500
    except Exception as e:
        return jsonify({'error': f"Error processing data: {str(e)}"}), 500






# ======================= Configuration ==========================
AZURE_SEARCH_ENDPOINT = "https://confitechaisearch.search.windows.net"
AZURE_SEARCH_KEY = "f0UZU9eHkSnK5MW3rtuQS8qK8BQboHtfgPPXv5CHiNAzSeDSzCGg"
AZURE_SEARCH_INDEX = "confitechindex"
leftbrain_AZURE_SEARCH_INDEX="leftbrainwmindex"
AZURE_OPENAI_ENDPOINT = "https://sauga-m9v6tumo-eastus2.cognitiveservices.azure.com/"
AZURE_OPENAI_KEY = "32obHSutgHYfCst8XyDi2vKUv0VcWnV7wfznAGMQIl3njYHU1wJIJQQJ99BDACHYHv6XJ3w3AAAAACOGAimG"
AZURE_OPENAI_DEPLOYMENT = "gpt-35-turbo"
AZURE_API_VERSION = "2024-12-01-preview"
# ======================= Initialization ==========================
def init_openai_client():
    return AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

def init_search_client():
    return SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )





# ======================= Utilities ==========================
def search_documents(query, top_k=5):
    search_client = init_search_client()
    results = search_client.search(query, top=top_k)
    return "\n\n".join([doc["content"] for doc in results])
# =======================  API Endpoint  ==========================

def build_prompt(context, question):
    return [
        {
            "role": "system",
            "content": (
                "You are a friendly and professional assistant for Confitech Solutions, "
                "a company specializing in AI, consulting, and IT services. Answer queries related to "
                "Confitech's services like AI solutions, data analytics, cloud consulting, etc. "
                "Use the context from the documents in the Azure Search index to answer the user's query. "
                "DO NOT ANSWER OUTSIDE OF THE PROVIDED CONTEXT. KEEP ANSWERS INFORMATIVE AND PRECISE."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}"
        }
    ]

def stream_chat_response(messages):
    try:
        openai_client = init_openai_client()
        response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            max_tokens=3000,
            temperature=0.3,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stream=True
        )
        for chunk in response:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
    except Exception as e:
        yield f"[ERROR]: {str(e)}"

@app.route("/chat", methods=["POST"])
def chat_stream():
    data = request.get_json()
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "Query not provided"}), 400

    context = search_documents(user_query)
    messages = build_prompt(context, user_query)

    return Response(stream_chat_response(messages), mimetype="text/plain")



def leftbrain_init_search_client():
    return SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=leftbrain_AZURE_SEARCH_INDEX,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )
# ======================= Utilities ==========================
def leftbrain_search_documents(query, top_k=5):
    search_client = leftbrain_init_search_client()
    results = search_client.search(query, top=top_k)
    return "\n\n".join([doc["content"] for doc in results])


def leftbrain_build_prompt(context, question):
    return [
        {
            "role": "system",
            "content": (
                "You are a friendly and professional assistant for Left Brain Wealth Management, "
                "a company specializing in customized portfolios, active portfolio management, "
                "securities evaluation, tax planning, and retirement planning. "
                "You help clients by providing clear, informative, and precise answers "
                "based on the given context from the Azure Search index.\n\n"

                "SPECIAL INSTRUCTIONS:\n"
                "1. If the user’s question is specifically about Noland’s Notes, "
                "ALWAYS include this PDF link at the end of your answer:\n"
                "   https://leftbrainwm.com/pdf/Nolands_Notes_2025%20(V3).pdf\n"
                "2. For all other questions, DO NOT include any links in your answer.\n"
                "3. Do not hallucinate. If the answer is not found in the context, politely say so.\n"
                "4. Left Brain Wealth Management contact number is 8009300378.\n"
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}"
        }
    ]


@app.route("/leftbrainchat", methods=["POST"])
def leftbrain_chat_stream():
    data = request.get_json()
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "Query not provided"}), 400

    context = leftbrain_search_documents(user_query)
    messages = leftbrain_build_prompt(context, user_query)

    return Response(stream_chat_response(messages), mimetype="text/plain")

#AddressParser
# Azure OpenAI Configuration
client_address = AzureOpenAI(
    api_key="3c3384effe084ff3b56101ab0c1d14df",  # ⚠️ Replace with environment variable in production
    api_version="2024-05-01-preview",
    azure_endpoint="https://confitech.openai.azure.com/"
)
deployment_name_for_address = "gpt-4"
# Max length for each field
MAX_LENGTHS = {
    "Dept": 70,
    "SubDept": 70,
    "StrtNm": 70,
    "BldgNb": 16,
    "BldgNm": 35,
    "Flr": 70,
    "PstBx": 16,
    "Room": 70,
    "PstCd": 16,
    "TwnNm": 35,
    "TwnLctnNm": 35,
    "DstrctNm": 35,
    "CtrySubDvsn": 35,
    "Ctry": 2  # ISO country code
}

# Required fields for scoring
REQUIRED_FIELDS = ["TwnNm", "Ctry"]

def validate_country(value):
    country = None
    try:
        country = pycountry.countries.lookup(value)
    except:
        return None
    return country.alpha_2  # Return the 2-letter ISO code

def clean_and_score(structured_data):
    issues = []
    score = 1.0

    cleaned = {}
    for field, max_len in MAX_LENGTHS.items():
        value = structured_data.get(field)

        if value is None:
            cleaned[field] = None
        else:
            value = str(value).strip()

            # Country field special case
            if field == "Ctry":
                iso_code = validate_country(value)
                if iso_code:
                    cleaned[field] = iso_code
                else:
                    cleaned[field] = None
                    issues.append(f"Invalid country value: '{value}'")
            else:
                cleaned[field] = value[:max_len] if len(value) > max_len else value

    for field in REQUIRED_FIELDS:
        if not cleaned.get(field):
            score -= 0.5
            issues.append(f"Missing or invalid required field: {field}")

    score = max(score, 0.0)
    return cleaned, score, issues
    
@app.route('/parse-address', methods=['POST'])
def parse_address():
    data = request.get_json()
    unstructured_address = data.get("address")

    if not unstructured_address:
        return jsonify({"error": "Address is required"}), 400

    # OpenAI prompt
    prompt = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": """
You are a helpful assistant that extracts address components from unstructured input and formats them using the ISO 20022 standard.

Respond ONLY with a valid JSON object, nothing else.

Return a JSON with the following fields:
- Dept
- SubDept
- StrtNm
- BldgNb
- BldgNm
- Flr
- PstBx
- Room
- PstCd
- TwnNm
- TwnLctnNm
- DstrctNm
- CtrySubDvsn
- Ctry


If data is missing, set the value to null. Do not include explanations or any text outside JSON.
"""
            }]
        },
        {
            "role": "user",
            "content": unstructured_address
        }
    ]

    try:
        response = client_address.chat.completions.create(
            model=deployment_name_for_address,
            messages=prompt,
            max_tokens=1000,
            temperature=0.1
        )
        result = response.choices[0].message.content.strip()

        match = re.search(r'\{.*\}', result, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in the response.")

        structured_data = json.loads(match.group())
        cleaned_data, score, issues = clean_and_score(structured_data)

        return jsonify({
            "structured_address": cleaned_data,
            "score": score,
            "issues": issues
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500




# =============== Portfolio-Specific Client ===============
def init_portfolio_openai_client():
    return AzureOpenAI(
        api_key="3c3384effe084ff3b56101ab0c1d14df",  # 🔒 Replace with env var in production
        api_version="2024-05-01-preview",
        azure_endpoint="https://confitech.openai.azure.com/"
    )

PORTFOLIO_OPENAI_DEPLOYMENT = "gpt-4"

# =============== Streaming Response Generator ===============
def stream_portfolio_chat_response(portfolio_messages):
    try:
        client = init_portfolio_openai_client()
        response = client.chat.completions.create(
            model=PORTFOLIO_OPENAI_DEPLOYMENT,
            messages=portfolio_messages,
            max_tokens=1000,
            temperature=0.3,
            stream=True
        )

        for chunk in response:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
    except Exception as e:
        yield f"[ERROR]: {str(e)}"

# =============== Prompt Builder ============================
def build_portfolio_prompt_for_chat_natural(table_data, question):
    return [
        {
            "role": "system",
            "content": """
You are a financial data assistant that answers questions strictly based on the provided JSON table of stocks or bonds.

You can:
- Read and interpret tabular data in JSON format
- Perform basic calculations like average, sum, min, max, and filtering
- Compare values across rows
- Parse semicolon-separated fields like "EPS Next Year Estimate Trends"
- Provide concise, clear, and insightful answers in a natural, conversational style, like a helpful chatbot.

Use **only** the provided data — do NOT add any external info or make assumptions.

Keep answers crisp, engaging, and story-like but factual.

If the data is insufficient to answer, reply:
"I'm sorry, but the data provided doesn't have enough information to answer that."

Avoid long explanations or irrelevant commentary.
"""
        },
        {
            "role": "user",
            "content": f"Here is the data:\n{json.dumps(table_data)}\n\nQuestion:\n{question}"
        }
    ]


# =============== Route: Portfolio Query Streaming ================
@app.route("/portfolio/query-stream", methods=["POST"])
def stream_portfolio_query_response():
    data = request.get_json()
    table_data = data.get("table_data")
    question = data.get("question")

    if not table_data or not question:
        return jsonify({"error": "Both 'table_data' and 'question' are required."}), 400

    messages = build_portfolio_prompt_for_chat_natural(table_data, question)
    return Response(stream_portfolio_chat_response(messages), mimetype="text/plain")



#=================Route: DMS Chatbot===========
DMS_AZURE_SEARCH_INDEX = "dmsindex"

def dms_init_search_client():
    return SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=DMS_AZURE_SEARCH_INDEX,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )

def dms_search_documents(query, top_k=5):
    search_client = dms_init_search_client()
    results = search_client.search(query, top=top_k)
    return "\n\n".join([doc["content"] for doc in results])

def dms_build_prompt(context, question):
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful and professional assistant for DMS (Document Management System). "
                "You answer user queries related to document search, metadata, retrieval, and insights strictly "
                "based on the indexed data in Azure AI Search. "
                "Do not hallucinate. Stay concise, informative, and within the provided context."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}"
        }
    ]

@app.route("/dmschat", methods=["POST"])
def dms_chat_stream():
    data = request.get_json()
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "Query not provided"}), 400

    context = dms_search_documents(user_query)
    messages = dms_build_prompt(context, user_query)

    return Response(stream_chat_response(messages), mimetype="text/plain")




# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
