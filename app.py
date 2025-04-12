import os
import fitz  # PyMuPDF
from openai import AzureOpenAI
from flask import Flask, request, jsonify
import json
from flask_cors import CORS
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
                The output should consists of all the important values from the PDF extatacted and also return the line items from the invoice.
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




# Route to handle PDF upload and query OpenAI API
@app.route('/upload', methods=['POST'])
def upload_pdf():
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
        data = extractInvoiceDetailsOpenAI(pdf_text)
        if data:
            json_str = data.strip().replace("```json\n", "").replace("\n```", "")
            parsed_json = json.loads(json_str)
            return jsonify(parsed_json)
        else:
            return jsonify({'error': 'No valid response from OpenAI'}), 500
    except Exception as e:
        return jsonify({'error': f"Error processing data: {str(e)}"}), 500

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



# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
