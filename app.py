import traceback
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import io
from deepface import DeepFace
import numpy as np
from PIL import Image
from pymongo import MongoClient
import os
import ast
import random
import nltk
from nltk.corpus import wordnet

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)
CORS(app)

# MongoDB Configuration
MONGO_URI = "mongodb+srv://sdehire:1111@cluster0.pft5g.mongodb.net/"
DB_NAME = "sdehire"
COLLECTION_NAME = "codeanalysis"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Set the allowed file extensions
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route to handle image uploads and emotion analysis
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'image' not in request.files:
            return jsonify({'message': 'No file part'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'message': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            # Read the image in memory using PIL and convert to OpenCV format
            image_data = file.read()
            img = Image.open(io.BytesIO(image_data))

            # Convert to OpenCV format (DeepFace expects RGB images)
            img_rgb = np.array(img.convert('RGB'))

            # Perform emotion analysis using the image in RGB format with enforce_detection=False
            try:
                # Pass the image to DeepFace for analysis
                analysis = DeepFace.analyze(img_rgb, actions=['emotion'], enforce_detection=False)

                # Log the analysis result for debugging
                print("DeepFace analysis result:", analysis)

                if not analysis or len(analysis) == 0:
                    raise ValueError("No emotion analysis results found.")

                # Since DeepFace returns a list of dictionaries, access the first element
                analysis_result = analysis[0]

                # Convert np.float32 values to regular float before inserting into MongoDB
                emotion_data = {emotion: float(value) for emotion, value in analysis_result['emotion'].items()}
                dominant_emotion = analysis_result['dominant_emotion']
                confidence = emotion_data[dominant_emotion]

                # Prepare the data to be stored in MongoDB
                analysis_data = {
                    "dominant_emotion": dominant_emotion,
                    "confidence": confidence,
                    "all_emotions": emotion_data
                }

                # Insert the analysis result into MongoDB
                collection.insert_one(analysis_data)

                return jsonify({
                    'message': 'File uploaded and emotion analysis successful',
                    'dominant_emotion': dominant_emotion,
                    'confidence': confidence,
                    'all_emotions': emotion_data
                }), 200

            except Exception as e:
                error_message = f"Error during emotion analysis: {str(e)}"
                print(error_message)
                return jsonify({'message': error_message}), 500

    except Exception as e:
        error_message = f"General error: {str(e)}"
        print(error_message)
        return jsonify({'message': error_message}), 500

# Function to analyze basic code structure
def analyze_code_structure(code_snippet):
    lines = code_snippet.split('\n')
    num_lines = len(lines)
    num_functions = len([line for line in lines if line.strip().startswith('def ')])
    num_classes = len([line for line in lines if line.strip().startswith('class ')])
    num_loops = len([line for line in lines if line.strip().startswith(('for', 'while'))])
    num_conditionals = len([line for line in lines if line.strip().startswith(('if', 'elif', 'else'))])
    
    # Time complexity estimation
    if num_loops > 1:
        time_complexity = f"O(n^{num_loops})"
    elif num_loops == 1:
        time_complexity = "O(n)"
    else:
        time_complexity = "O(1)"
    
    # Space complexity estimation
    space_complexity = "O(1)"
    if any(x in code_snippet for x in ['list', 'dict', 'set']):
        space_complexity = "O(n)"
    
    return {
        "num_lines": num_lines,
        "num_functions": num_functions,
        "num_classes": num_classes,
        "num_loops": num_loops,
        "num_conditionals": num_conditionals,
        "time_complexity": time_complexity,
        "space_complexity": space_complexity
    }

# Main function to analyze code
@app.route('/analyze_code', methods=['POST'])
def analyze_code():
    try:
        data = request.json
        code_snippet = data.get('code_snippet')
        
        if not isinstance(code_snippet, str) or not code_snippet.strip():
            return jsonify({"error": "Invalid or empty code snippet provided"}), 400

        analysis = analyze_code_structure(code_snippet)
        return jsonify({
            "code_snippet": code_snippet,
            "analysis": analysis,
            
        })
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

class EnhancedQuestionGenerator:
    def __init__(self):
        """
        Initialize the question generator with enhanced templates.
        """
        self.question_templates = {
            "function": "What does the function '{name}' at line {lineno} do?",
            "function_args": "What are the purposes of the arguments in the function '{name}'?",
            "variable": "What is the role of the variable '{name}' at line {lineno}?",
            "loop": "What is the significance of the loop starting at line {lineno}?",
            "conditional": "Under what conditions does the code block at line {lineno} execute?",
            "import": "Why is the module '{name}' imported, and how is it used?",
            "docstring": "What does the following docstring explain: '{doc}'?"
        }

    def analyze_code(self, code_snippet):
        """
        Analyze the code to extract structural information.
        """
        try:
            tree = ast.parse(code_snippet)
            analysis = {
                "functions": [
                    {
                        "name": node.name,
                        "lineno": node.lineno,
                        "args": [arg.arg for arg in node.args.args]
                    }
                    for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                ],
                "variables": [
                    {"name": node.id, "lineno": node.lineno}
                    for node in ast.walk(tree) if isinstance(node, ast.Name)
                ],
                "loops": [
                    {"lineno": node.lineno}
                    for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))
                ],
                "conditionals": [
                    {"lineno": node.lineno}
                    for node in ast.walk(tree) if isinstance(node, ast.If)
                ],
                "imports": [
                    {"name": node.names[0].name, "lineno": node.lineno}
                    for node in ast.walk(tree) if isinstance(node, ast.Import)
                ],
                "docstrings": self.extract_docstrings(tree)
            }
            return analysis
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def extract_docstrings(tree):
        """
        Extract docstrings from the module and functions.
        """
        docstrings = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef)):
                docstring = ast.get_docstring(node)
                if docstring:
                    docstrings.append(docstring.strip())
        return docstrings

    def generate_questions(self, code_snippet):
        """
        Generate questions based on the analysis and enhanced templates.
        """
        analysis = self.analyze_code(code_snippet)
        if "error" in analysis:
            return [f"Error analyzing code: {analysis['error']}"]

        questions = []

        # Generate questions for functions and arguments
        for func in analysis["functions"]:
            questions.append(
                self.question_templates["function"].format(name=func["name"], lineno=func["lineno"])
            )
            if func["args"]:
                questions.append(
                    self.question_templates["function_args"].format(name=func["name"])
                )

        # Generate questions for variables
        for var in analysis["variables"]:
            questions.append(
                self.question_templates["variable"].format(name=var["name"], lineno=var["lineno"])
            )

        # Generate questions for loops
        for loop in analysis["loops"]:
            questions.append(
                self.question_templates["loop"].format(lineno=loop["lineno"])
            )

        # Generate questions for conditionals
        for cond in analysis["conditionals"]:
            questions.append(
                self.question_templates["conditional"].format(lineno=cond["lineno"])
            )

        # Generate questions for imports
        for imp in analysis["imports"]:
            questions.append(
                self.question_templates["import"].format(name=imp["name"])
            )

        # Generate questions for docstrings
        for doc in analysis["docstrings"]:
            questions.append(
                self.question_templates["docstring"].format(doc=doc)
            )

        # Deduplicate questions while preserving order
        unique_questions = list(dict.fromkeys(questions))
        
        # Select and return a random question
        return random.choice(unique_questions)

@app.route('/generate_question', methods=['POST'])
def generate_question():
    try:
        code_snippet = request.json.get("code_snippet")
        if not code_snippet:
            return jsonify({"error": "No code snippet provided"}), 400

        generator = EnhancedQuestionGenerator()
        question = generator.generate_questions(code_snippet)
        return jsonify({"question": question})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_synonyms(word):
    """Fetch synonyms of a word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())  # Add the synonym to the set
    return synonyms

def paraphrase_text(input_text):
    """A basic paraphrase function using synonym replacement."""
    # Check if input_text is a list, if so, join it into a single string
    if isinstance(input_text, list):
        input_text = ' '.join(input_text)

    words = input_text.split()
    paraphrased_words = []

    for word in words:
        synonyms = get_synonyms(word)
        if synonyms:
            # Replace the word with a random synonym (if available)
            paraphrased_words.append(list(synonyms)[0])  # Take the first synonym for simplicity
        else:
            paraphrased_words.append(word)  # Keep the word if no synonym is found

    paraphrased_text = " ".join(paraphrased_words)
    return paraphrased_text

def calculate_score(original, paraphrased):
    """A simple scoring function based on word overlap (Jaccard similarity)."""
    
    # Ensure original and paraphrased are strings (if they are lists, join them into a string)
    if isinstance(original, list):
        original = ' '.join(original)
    if isinstance(paraphrased, list):
        paraphrased = ' '.join(paraphrased)

    original_words = set(original.split())
    paraphrased_words = set(paraphrased.split())

    # Handle case where both original and paraphrased texts are empty
    if len(original_words) == 0 or len(paraphrased_words) == 0:
        return 0  # Return a score of 0 if either text is empty

    intersection = original_words.intersection(paraphrased_words)
    score = len(intersection) / len(original_words.union(paraphrased_words))  # Jaccard similarity
    return score

def paraphrase_and_score(input_text):
    """Paraphrase the text and calculate the score between original and paraphrased text."""
    # Paraphrase the input text
    paraphrased_text = paraphrase_text(input_text)

    # Calculate the similarity score (Jaccard similarity)
    score = calculate_score(input_text, paraphrased_text)

    return paraphrased_text, score

@app.route('/api/saveTranscript', methods=['POST'])
def save_transcript():
    # Get the JSON data from the request
    data = request.json
    ai_transcripts = data.get("aiTranscripts")
    user_transcripts = data.get("userTranscripts")
    
    # Print data to console
    print("Received AI Transcripts:", ai_transcripts)
    print("Received User Transcripts:", user_transcripts)
    
    # Process both AI and User transcripts
    ai_paraphrased_text, ai_score = paraphrase_and_score(ai_transcripts)
    user_paraphrased_text, user_score = paraphrase_and_score(user_transcripts)
    
    # Print results to console
    print("AI Paraphrased Text:", ai_paraphrased_text)
    print("User Paraphrased Text:", user_paraphrased_text)
    print("AI Score:", ai_score)
    print("User Score:", user_score)
    
    # Prepare the response
    response = {
        "aiTranscripts": ai_transcripts,
        "userTranscripts": user_transcripts,
        "aiParaphrasedText": ai_paraphrased_text,
        "userParaphrasedText": user_paraphrased_text,
        "aiScore": ai_score,
        "userScore": user_score
    }
    print(response)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

