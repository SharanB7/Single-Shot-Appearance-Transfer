from flask import Flask, request, jsonify, abort
from openai import OpenAI
import json
import os
import logging

# Initialize Flask app
app = Flask(__name__)
client = OpenAI()
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "default-token")

@app.before_request
def verify_token():
    """
    Middleware to verify the token before processing any request.
    """
    token = request.headers.get("Authorization")
    # logger.debug("verification started")
    if not token or token != AUTH_TOKEN:
        abort(403, description="Forbidden: Invalid or missing token")
    # logger.debug("token verified sucessfully")


# Define a GET endpoint
@app.route("/get-response", methods=["GET"])
def get_gpt_response():
    # Get the prompt from query parameters
    prompt = request.args.get("prompt")

    if not prompt:
        return jsonify({"error": "Please provide a prompt"}), 400

    try:
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    # "content": "You are a helpful assistant.",
                    "content": """
                    You are provided with:
                    A target image description (text).
                    Your tasks are:
                    Decompose the target image description into three components:
                    a0 (Text phrases to eliminate): Identify phrases that correspond to elements whose texture or pattern we want to capture from the provided image features, but we do not want these elements to appear directly in the final image.
                    a1 (Secondary text phrases): Identify phrases of the objects on which the texture has to be transferred.
                    a2 (Main role): Identify the primary subject or main element that should appear in the final image along with the secondary text phrase and the connecting verb.
                    Assign tasks using the following functions:
                    clip_encode_text(text): returns text_features
                    compute(text_features_1, text_features_2): returns text_features
                    text2image(text_features): returns image
                    ipadapter_text2image(text, image=None): returns image
                    Note: The image features V0 have already been computed using clip_encode_image(image) and are stored in V0. Use V0 in your computations where necessary.
                    Assign tasks as follows(name of all tasks should be explicit):
                    Task1:
                    function: \"clip_encode_text\"
                    input_text: [Phrases in a0]
                    output_variable: \"T0\"
                    Task2:
                    function: \"compute\"
                    action: \"subtract\"
                    input_variables: [\"V0\", \"T0\"]
                    output_variable: \"I0\"
                    Task3:
                    function: \"clip_encode_text\"
                    input_text: [Phrases in a1]
                    output_variable: \"T1\"
                    Task4:
                    function: \"compute\"
                    action: \"add\"
                    input_variables: [\"I0\", \"T1\"]
                    output_variable: \"I2\"
                    Task5:
                    function: \"text2image\"
                    input_variables: [\"I2\"]
                    output_variable: \"Generated_Image\"
                    Task6:
                    function: \"ipadapter_text2image\"
                    input_text: [Phrases in a2]
                    input_image: \"Generated_Image\"
                    output_variable: \"FinalImage\"
                    Constraints:
                    Use only the function names and keys as specified.
                    Include only the relevant keys for each task.
                    Replace [Phrases in a0], [Phrases in a1], and [Phrases in a2] with the actual phrases from your decomposition.
                    Do not include any explanations, descriptions, or additional text.
                    Your entire output should be the JSON array only.
                    Example Input:
                    (Do not include this example in your output.)
                    Target image description: \"A cat playing with a toy that has patterns similar to the fabric\"
                    Example Decomposition:
                    a0: [\"fabric\"] (We want the pattern of the fabric but not the fabric itself)
                    a1: [\"toy with a pattern of\"] (Phrases for pattern integration)
                    a2: [\"A cat playing with a toy\"] (Other phrases)
                    (Again, do not include this example in your output.)
                    """,
                },
                {"role": "user", "content": prompt},
            ],
        )
        # Extract the content from the response
        raw_response = response.choices[0].message.content

        # Step 1: Remove the code block markers
        cleaned_response = raw_response.strip("```json").strip("```").strip()

        # Step 2: Parse the JSON string into a Python object
        parsed_json = json.loads(cleaned_response)

        # Step 3: Return or use the parsed JSON
        return jsonify(parsed_json)  # Flask JSON response

    except json.JSONDecodeError as e:
        return (
            jsonify(
                {"error": "Invalid JSON format in the response", "details": str(e)}
            ),
            500,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the server
if __name__ == "__main__":
    app.run(debug=True)
