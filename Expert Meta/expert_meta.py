import os
import re
import time
import logging
import base64
import json
from pathlib import Path
import markdown
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("executor.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables (API key)
load_dotenv()

class SequentialPromptExecutor:
    def __init__(self, system_prompt, model_name="gemini-2.0-flash-thinking-exp-01-21", max_iterations=25, 
                 temperature=1, top_p=0.95, top_k=40, max_output_tokens=8192, 
                 enable_code_execution=True, attachments_folder="Attachments"):
        """
        Initialize the executor with the system prompt and model configuration.
        
        Args:
            system_prompt (str): The system prompt that guides the LLM
            model_name (str): The LLM model to use
            max_iterations (int): Maximum number of iterations to prevent infinite loops
            temperature (float): Controls randomness (higher = more random)
            top_p (float): Nucleus sampling parameter
            top_k (int): Number of highest probability tokens to consider
            max_output_tokens (int): Maximum length of generated response
            enable_code_execution (bool): Whether to enable code execution tools
            attachments_folder (str): Folder path containing files to attach to the prompt
        """
        # Configure the API
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        self.system_prompt = system_prompt
        self.model_name = model_name
        self.conversation_history = []
        self.chat_history = []  # For maintaining the chat context with the API
        self.execution_count = 0
        self.max_iterations = max_iterations
        self.attachments_folder = attachments_folder
        self.file_references = []  # To store file references after upload

        # Create output directory for rich content
        self.output_dir = "Final Solution Output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Define safety settings - set all categories to BLOCK_NONE
        safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

        # Create configuration parameters dictionary
        config_params = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_output_tokens,
            "safety_settings": safety_settings,
            "system_instruction": system_prompt
        }

        # Conditionally add tools
        if enable_code_execution:
            config_params["tools"] = [types.Tool(code_execution=types.ToolCodeExecution)]

        # Create the generation config with all parameters
        self.generation_config = types.GenerateContentConfig(**config_params)

        # Initialize with system prompt
        self._add_to_history("system", system_prompt)

        # Process any files in the Attachments folder
        if os.path.exists(self.attachments_folder):
            self._process_attachment_files()

    def _process_attachment_files(self):
        """
        Process all files in the Attachments folder and upload them to the API.
        Stores the file references for later use in prompts.
        """
        if not os.path.exists(self.attachments_folder):
            logging.warning("Attachments folder '%s' does not exist.", self.attachments_folder)
            return
            
        # Get all files in the Attachments folder
        all_files = []
        for file in os.listdir(self.attachments_folder):
            file_path = os.path.join(self.attachments_folder, file)
            if os.path.isfile(file_path):  # Make sure it's a file, not a subdirectory
                all_files.append(file_path)
        
        if not all_files:
            logging.info("No attachable files found in '%s'.", self.attachments_folder)
            return
            
        # Upload each file using the File API
        for file_path in all_files:
            try:
                file_name = os.path.basename(file_path)
                logging.info("Uploading file: %s", file_name)
                
                # Upload the file and store the file reference
                file_ref = self.client.files.upload(file=file_path)

                self.file_references.append(file_ref)
                logging.info("Successfully uploaded %s: %s", file_name, file_ref)
            except Exception as e:
                logging.error("Error uploading file %s: %s", file_path, str(e))
        
        logging.info("Uploaded %d files for use with prompts.", len(self.file_references))

    def _add_to_history(self, role, content):
        """
        Add a message to the conversation history.
        
        Args:
            role (str): The role of the message sender (system, user, assistant)
            content (str): The content of the message
        """
        # Initialize a flag for the initial user prompt
        is_initial_user_prompt = (role == "user" and 
                                  not any(msg["role"] == "user" for msg in self.conversation_history))
        
        if role == "system" or is_initial_user_prompt or role == "assistant":
            self.conversation_history.append({"role": role, "content": content})

        # For the API chat_history, only add user messages
        if role == "user":
            # For user messages, we need to check if it's the initial prompt
            # If it's the initial prompt and we have file references, we need to include them
            if not self.chat_history or len(self.chat_history) == 0:  # This is the initial prompt
                parts = [types.Part.from_text(text=content)]
                
                # Add any file references to the initial prompt
                if hasattr(self, 'file_references') and self.file_references:
                    for file_ref in self.file_references:
                        # Create a Part object with the file_data
                        file_part = types.Part(
                            file_data=types.FileData(
                                file_uri=file_ref.uri,
                                mime_type=file_ref.mime_type
                            )
                        )
                        parts.append(file_part)
                    
                self.chat_history.append(
                    types.Content(
                        role="user",
                        parts=parts
                    )
                )
            else:
                # For subsequent user messages, just add the text
                self.chat_history.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=content)]
                    )
                )
        #elif role == "assistant":
            #self.chat_history.append(
                #types.Content(
                    #role="model",
                    #parts=[types.Part.from_text(text=content)]
                #)
            #)

    def has_final_solution(self, response):
        """
        Check if the response contains a final solution.
        
        Args:
            response (str): The response to check
            
        Returns:
            bool: True if a final solution is found, False otherwise
        """
        # Check for the final solution marker
        return "FINAL SOLUTION" in response.upper()

    def extract_final_solution(self, response_text):
        """
        Extract the final solution from the response text.
        
        Args:
            response_text (str): The response text to extract from
            
        Returns:
            str or None: The entire response text when "FINAL SOLUTION" is present, or None if not found
        """
        # If "FINAL SOLUTION" is present, return the entire response_text
        if "FINAL SOLUTION" in response_text.upper():
            return response_text.strip()

        return None

    def process_rich_content(self, solution_text):
        """
        Process the solution text to extract and save rich content like code, images, graphs.
        
        Args:
            solution_text (str): The solution text to process
            
        Returns:
            dict: Dictionary containing the processed solution with metadata
        """
        # Ensure we have a valid solution text
        if solution_text is None:
            solution_text = ""

        # Create a structured result
        result = {
            "original_text": solution_text,
            "markdown": solution_text,  # By default, we treat the whole text as markdown
            "has_code": False,
            "code_blocks": [],
            "has_images": False,
            "image_paths": [],
            "has_charts": False,
            "chart_data": []
        }
        
        try:
            # Extract code blocks
            code_pattern = r"```(\w*)\n([\s\S]*?)```"
            code_matches = re.findall(code_pattern, solution_text)
            
            if code_matches:
                result["has_code"] = True
                
                # Process each code block
                for i, (lang, code) in enumerate(code_matches):
                    try:
                        code_info = {
                            "language": lang if lang else "text",
                            "code": code,
                            "index": i
                        }
                        
                        # Save the code to a file
                        filename = f"code_block_{i+1}.{lang if lang else 'txt'}"
                        file_path = os.path.join(self.output_dir, filename)
                        
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(code)
                        
                        code_info["file_path"] = file_path
                        result["code_blocks"].append(code_info)

                    except Exception as e:
                        logging.error("Error processing code block %d: %s", i, str(e))
            
            # Extract any base64 encoded images
            img_pattern = r"data:image\/(\w+);base64,([^\"')\s]+)"
            img_matches = re.findall(img_pattern, solution_text)
            
            if img_matches:
                result["has_images"] = True
                
                # Process each image
                for i, (img_type, img_data) in enumerate(img_matches):
                    try:
                        # Decode the base64 data
                        decoded_data = base64.b64decode(img_data)
                        
                        # Save the image to a file
                        filename = f"image_{i+1}.{img_type}"
                        file_path = os.path.join(self.output_dir, filename)
                        
                        with open(file_path, "wb") as f:
                            f.write(decoded_data)
                        
                        # Add to the image paths
                        result["image_paths"].append({
                            "index": i,
                            "type": img_type,
                            "file_path": file_path
                        })
                        
                        # Replace the base64 data with a link to the saved file in the markdown
                        result["markdown"] = result["markdown"].replace(
                            f"data:image/{img_type};base64,{img_data}",
                            f"file://{file_path}"
                        )
                        
                    except Exception as e:
                        logging.error("Error processing image %d: %s", i, str(e))
            
            # Look for chart data (typically JSON objects)
            chart_pattern = r"chartData\s*=\s*(\[[\s\S]*?\]);?"
            chart_matches = re.findall(chart_pattern, solution_text)
            
            if chart_matches:
                result["has_charts"] = True
                
                # Process each chart data block
                for i, chart_data_str in enumerate(chart_matches):
                    try:
                        # Try to parse the JSON data
                        chart_data = json.loads(chart_data_str)
                        
                        # Save to a JSON file
                        filename = f"chart_data_{i+1}.json"
                        file_path = os.path.join(self.output_dir, filename)
                        
                        with open(file_path, "w", encoding="utf-8") as f:
                            json.dump(chart_data, f, indent=2)
                        
                        # Add to chart data
                        result["chart_data"].append({
                            "index": i,
                            "data": chart_data,
                            "file_path": file_path
                        })
                        
                    except Exception as e:
                        logging.warning("Failed to parse chart data %d: %s", i, str(e))
            
            # Save the processed markdown to a file
            try:
                markdown_path = os.path.join(self.output_dir, "final_solution.md")
                with open(markdown_path, "w", encoding="utf-8") as f:
                    f.write(result["markdown"])
            
                result["markdown_path"] = markdown_path
            except Exception as e:
                logging.error("Failed to write markdown output to a file: %s", e)
            
            # Generate an HTML version
            try:
                html_content = self.generate_html_from_markdown(result["markdown"])
                html_path = os.path.join(self.output_dir, "final_solution.html")

                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
            
                result["html_path"] = html_path
            
            except Exception as e:
                logging.error("Failed to write HTML output: %s", e)

        except Exception as e:
        # If any unexpected error occurs during the main processing, log it
            logging.error("Error in process_rich_content: %s", str(e))

        return result

    def generate_html_from_markdown(self, markdown_content, title="Final Solution Output"):
        """
        Generate HTML content from markdown with proper code formatting.
        
        Args:
            markdown_content (str): The markdown content to convert to HTML
            title (str): The title for the HTML document
            
        Returns:
            str: The generated HTML content
        """
        # Use Python-Markdown with the CodeHilite and Fenced Code Blocks extensions      
        # Convert markdown to HTML with extensions for better code handling
        html_body = markdown.markdown(
            markdown_content,
            extensions=[
                'markdown.extensions.codehilite',   # Adds syntax highlighting
                'markdown.extensions.fenced_code',  # Handles ```code``` blocks
                'markdown.extensions.tables'        # Adds table support
            ]
        )
        
        # Create the full HTML document with additional CSS for code highlighting
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                
                /* Improved code block styling */
                pre {{ 
                    background-color: #f5f5f5; 
                    padding: 16px; 
                    border-radius: 4px; 
                    overflow-x: auto;
                    border: 1px solid #ddd;
                }}
                
                /* Inline code styling */
                code {{ 
                    font-family: 'Courier New', monospace;
                    background-color: #f5f5f5;
                    padding: 2px 4px;
                    border-radius: 3px;
                }}
                
                /* Reset background for code within pre blocks to avoid double background */
                pre code {{
                    padding: 0;
                    background-color: transparent;
                }}
                
                /* Syntax highlighting classes */
                .codehilite .hll {{ background-color: #ffffcc }}
                .codehilite .c {{ color: #408080; font-style: italic }} /* Comment */
                .codehilite .k {{ color: #008000; font-weight: bold }} /* Keyword */
                .codehilite .o {{ color: #666666 }} /* Operator */
                .codehilite .cm {{ color: #408080; font-style: italic }} /* Comment.Multiline */
                .codehilite .cp {{ color: #BC7A00 }} /* Comment.Preproc */
                .codehilite .c1 {{ color: #408080; font-style: italic }} /* Comment.Single */
                .codehilite .cs {{ color: #408080; font-style: italic }} /* Comment.Special */
                .codehilite .gd {{ color: #A00000 }} /* Generic.Deleted */
                .codehilite .ge {{ font-style: italic }} /* Generic.Emph */
                .codehilite .gr {{ color: #FF0000 }} /* Generic.Error */
                .codehilite .gh {{ color: #000080; font-weight: bold }} /* Generic.Heading */
                .codehilite .gi {{ color: #00A000 }} /* Generic.Inserted */
                .codehilite .go {{ color: #888888 }} /* Generic.Output */
                .codehilite .gp {{ color: #000080; font-weight: bold }} /* Generic.Prompt */
                .codehilite .gs {{ font-weight: bold }} /* Generic.Strong */
                .codehilite .gu {{ color: #800080; font-weight: bold }} /* Generic.Subheading */
                .codehilite .gt {{ color: #0044DD }} /* Generic.Traceback */
                .codehilite .kc {{ color: #008000; font-weight: bold }} /* Keyword.Constant */
                .codehilite .kd {{ color: #008000; font-weight: bold }} /* Keyword.Declaration */
                .codehilite .kn {{ color: #008000; font-weight: bold }} /* Keyword.Namespace */
                .codehilite .kp {{ color: #008000 }} /* Keyword.Pseudo */
                .codehilite .kr {{ color: #008000; font-weight: bold }} /* Keyword.Reserved */
                .codehilite .kt {{ color: #B00040 }} /* Keyword.Type */
                .codehilite .m {{ color: #666666 }} /* Literal.Number */
                .codehilite .s {{ color: #BA2121 }} /* Literal.String */
                .codehilite .na {{ color: #7D9029 }} /* Name.Attribute */
                .codehilite .nb {{ color: #008000 }} /* Name.Builtin */
                .codehilite .nc {{ color: #0000FF; font-weight: bold }} /* Name.Class */
                .codehilite .no {{ color: #880000 }} /* Name.Constant */
                .codehilite .nd {{ color: #AA22FF }} /* Name.Decorator */
                .codehilite .ni {{ color: #999999; font-weight: bold }} /* Name.Entity */
                .codehilite .ne {{ color: #D2413A; font-weight: bold }} /* Name.Exception */
                .codehilite .nf {{ color: #0000FF }} /* Name.Function */
                .codehilite .nl {{ color: #A0A000 }} /* Name.Label */
                .codehilite .nn {{ color: #0000FF; font-weight: bold }} /* Name.Namespace */
                .codehilite .nt {{ color: #008000; font-weight: bold }} /* Name.Tag */
                .codehilite .nv {{ color: #19177C }} /* Name.Variable */
                .codehilite .ow {{ color: #AA22FF; font-weight: bold }} /* Operator.Word */
                .codehilite .w {{ color: #bbbbbb }} /* Text.Whitespace */
                .codehilite .mb {{ color: #666666 }} /* Literal.Number.Bin */
                .codehilite .mf {{ color: #666666 }} /* Literal.Number.Float */
                .codehilite .mh {{ color: #666666 }} /* Literal.Number.Hex */
                .codehilite .mi {{ color: #666666 }} /* Literal.Number.Integer */
                .codehilite .mo {{ color: #666666 }} /* Literal.Number.Oct */
                .codehilite .sa {{ color: #BA2121 }} /* Literal.String.Affix */
                .codehilite .sb {{ color: #BA2121 }} /* Literal.String.Backtick */
                .codehilite .sc {{ color: #BA2121 }} /* Literal.String.Char */
                .codehilite .dl {{ color: #BA2121 }} /* Literal.String.Delimiter */
                .codehilite .sd {{ color: #BA2121; font-style: italic }} /* Literal.String.Doc */
                .codehilite .s2 {{ color: #BA2121 }} /* Literal.String.Double */
                .codehilite .se {{ color: #BB6622; font-weight: bold }} /* Literal.String.Escape */
                .codehilite .sh {{ color: #BA2121 }} /* Literal.String.Heredoc */
                .codehilite .si {{ color: #BB6688; font-weight: bold }} /* Literal.String.Interpol */
                .codehilite .sx {{ color: #008000 }} /* Literal.String.Other */
                .codehilite .sr {{ color: #BB6688 }} /* Literal.String.Regex */
                .codehilite .s1 {{ color: #BA2121 }} /* Literal.String.Single */
                .codehilite .ss {{ color: #19177C }} /* Literal.String.Symbol */
                .codehilite .bp {{ color: #008000 }} /* Name.Builtin.Pseudo */
                .codehilite .fm {{ color: #0000FF }} /* Name.Function.Magic */
                .codehilite .vc {{ color: #19177C }} /* Name.Variable.Class */
                .codehilite .vg {{ color: #19177C }} /* Name.Variable.Global */
                .codehilite .vi {{ color: #19177C }} /* Name.Variable.Instance */
                .codehilite .vm {{ color: #19177C }} /* Name.Variable.Magic */
                .codehilite .il {{ color: #666666 }} /* Literal.Number.Integer.Long */
                
                img {{ max-width: 100%; height: auto; }}
                h1, h2, h3 {{ color: #333; }}
                .container {{ max-width: 800px; margin: 0 auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                {html_body}
            </div>
        </body>
        </html>
        """
        
        return html_content

    def _process_full_response(self, response):
        """
        Process the full response from the model, handling all content types.
        
        Args:
            response: The full response object from the model
            
        Returns:
            str: A string representation of the full response, including rich content
        """
        result = []
        
        # Check if we have a valid response
        if not response or not response.candidates or not response.candidates[0].content:
            return ""
        
        # Access the content parts
        content = response.candidates[0].content
        
        for part in content.parts:
            # Skip empty parts where all attributes are None or empty strings
            is_empty_part = True
            for attr in vars(part):
                if attr.startswith('_'):
                    continue
                value = getattr(part, attr)
                if value is not None and value != '':
                    is_empty_part = False
                    break
                    
            if is_empty_part:
                continue
                
            # Handle text parts
            if hasattr(part, 'text') and part.text:
                result.append(part.text)
            
            # Handle code execution parts
            elif hasattr(part, 'function_call') and part.function_call:
                # Extract function call information
                function_name = part.function_call.name
                function_args = part.function_call.args
                result.append(f"```\n{function_name}({json.dumps(function_args, indent=2)})\n```")
            
            # Handle code execution results
            elif hasattr(part, 'function_response') and part.function_response:
                # Format the response
                response_data = part.function_response.response
                result.append(f"```\nFunction Response:\n{json.dumps(response_data, indent=2)}\n```")
            
            # Handle executable code
            elif hasattr(part, 'executable_code') and part.executable_code:
                # Get the code and language
                code = part.executable_code.code
                language = part.executable_code.language
                
                # Format the code with the appropriate syntax highlighting
                language_str = str(language).replace('Language.', '').lower() if hasattr(language, 'replace') else 'python'
                result.append(f"```{language_str}\n{code}\n```")
            
            # Handle code execution results (success or failure)
            elif hasattr(part, 'code_execution_result') and part.code_execution_result:
                # Get the execution outcome and output
                outcome = part.code_execution_result.outcome
                output = part.code_execution_result.output
                
                # Format based on success or failure
                if outcome == "OUTCOME_FAILED":
                    result.append(f"```\nCode Execution Failed:\n{output}\n```")
                else:
                    result.append(f"```\nCode Execution Result ({outcome}):\n{output}\n```")
            
            # Handle image parts - if the model returns images
            elif hasattr(part, 'inline_data') and part.inline_data:
                try:
                    # Get the MIME type and data
                    mime_type = part.inline_data.mime_type
                    
                    # Instead of embedding base64 data directly, just indicate an image was received
                    if 'image' in mime_type:
                        result.append("[Image received]")
                    else:
                        # For other data types, just note that we received it
                        result.append(f"[Received data of type: {mime_type}]") 
                  
                except Exception as e:
                    logging.error("Error processing inline data: %s", str(e))
                    result.append("[Error processing inline data]")
            
            # Handle chart data (typically JSON or HTML)
            elif hasattr(part, 'html') and part.html:
                # For HTML content (could be charts/visualizations)
                result.append(f"```html\n{part.html}\n```")
            
            # Handle thought parts (if the model provides reasoning)
            elif hasattr(part, 'thought') and part.thought:
                result.append(f"**Model Reasoning:**\n{part.thought}")
            
            # We shouldn't reach this point if our empty part detection is working correctly
            else:
                # Log what we found for debugging purposes only
                logging.debug("Skipping part with no meaningful content: %s", type(part).__name__)
        
        return "\n".join(result)

    def execute_chain(self, user_prompt):
        """
        Execute the entire chain of prompts automatically until a final solution is reached.
        The Assistant response should include embedded instructions that, when provided as input to 
        the model in a subsequent interaction, will guide the generation of the next Assistant response.
        
        Args:
            user_prompt (str): The initial user prompt
            
        Returns:
            list: The full conversation history
            dict: The processed final solution with metadata
        """
        try:
            # Start with the initial prompt
            self._add_to_history("user", user_prompt)

            # Send the initial user prompt to the model
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=self.chat_history,  # At this point, chat_history only has the initial user prompt
                    config=self.generation_config
                )
                
                response_text = self._process_full_response(response)
                self._add_to_history("assistant", response_text)
                self.execution_count += 1
                
                logging.info("Execution #%d completed.", self.execution_count)
                
                # Continue execution until a final solution is found or max iterations reached
                final_solution = None
                processed_solution = None

                while not self.has_final_solution(response_text) and self.execution_count < self.max_iterations:
                    # First, add the previous response as a user message to chat_history 
                    # to maintain the alternating pattern (but not conversation_history)
                    self._add_to_history("user", response_text)
                    
                    # Get next response - include the chat history for context
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=self.chat_history,  # Send the full chat history for context
                        config=self.generation_config
                    )
                    
                    response_text = self._process_full_response(response)
                    self._add_to_history("assistant", response_text)
                    self.execution_count += 1
                    
                    logging.info("Execution #%d completed.", self.execution_count)
                    
                    # If we found a final solution, extract it
                    if self.has_final_solution(response_text):
                        final_solution = self.extract_final_solution(response_text)

                        # Process the final solution for rich content
                        if final_solution:
                            processed_solution = self.process_rich_content(final_solution)
            
                if self.execution_count >= self.max_iterations and not final_solution:
                    logging.warning("Warning: Reached maximum iterations (%d) without finding a final solution.", self.max_iterations)
                
                return self.conversation_history, processed_solution or {"original_text": final_solution} if final_solution else None
            
            except Exception as api_error:
                logging.error("API error during execution: %s", str(api_error))
                return self.conversation_history, None
            
        except Exception as e:
            logging.error("Error during execution chain: %s", str(e))
            return self.conversation_history, None
    
    def get_full_conversation_text(self):
        """
        Format the entire conversation history as a single text output.

        The i//2 pairs user and assistant messages into numbered turns.
        
        Returns:
            str: Formatted conversation history
        """
        output = []
        assistant_count = 0
        continuation_user_count = 0

        for i, message in enumerate(self.conversation_history):
            if message["role"] == "system":
                output.append(f"[SYSTEM PROMPT]\n{message['content']}\n")
            elif message["role"] == "user" and i == 1:  # The initial user prompt
                # Add information about attachments if any were used
                if self.file_references:
                    file_list = ', '.join([os.path.basename(str(ref)) for ref in self.file_references])
                    output.append(f"[INITIAL USER PROMPT WITH ATTACHMENTS: {file_list}]\n{message['content']}\n")
                else:
                    output.append(f"[INITIAL USER PROMPT]\n{message['content']}\n")
            elif message["role"] == "user" and i > 1:  # The continuation user prompts
                output.append(f"[CONTINUATION USER PROMPT #{continuation_user_count}]\n{message['content']}\n")
                continuation_user_count += 1
            elif message["role"] == "assistant":
                assistant_count += 1
                output.append(f"[ASSISTANT #{assistant_count}]\n{message['content']}\n")
                
        return "\n".join(output)
    
    def generate_conversation_html(self, title="Conversation History"):
        """
        Format the entire conversation history as HTML output.
        
        Args:
            title (str): The title for the HTML document
            
        Returns:
            str: Formatted HTML conversation history
        """
        # Start the HTML document
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                pre {{ background-color: #f5f5f5; padding: 16px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; }}
                code {{ font-family: 'Courier New', monospace; }}
                .message {{ margin-bottom: 20px; padding-bottom: 20px; border-bottom: 1px solid #eee; }}
                .system {{ background-color: #f8f9fa; padding: 15px; border-left: 5px solid #6c757d; }}
                .user {{ background-color: #e9f5ff; padding: 15px; border-left: 5px solid #007bff; }}
                .assistant {{ background-color: #f0fff4; padding: 15px; border-left: 5px solid #28a745; }}
                .message-header {{ font-weight: bold; margin-bottom: 10px; color: #495057; }}
                .container {{ max-width: 900px; margin: 0 auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
        """
        
        # Add each message
        assistant_count = 0
        continuation_user_count = 0

        for i, message in enumerate(self.conversation_history):
            if message["role"] == "system":
                html_content += f"""
                <div class="message system">
                    <div class="message-header">[SYSTEM PROMPT]</div>
                    <pre>{message['content']}</pre>
                </div>
                """
            elif message["role"] == "user" and i == 1:  # The initial user prompt
                # Add information about attachments if any were used
                if self.file_references:
                    file_list = ', '.join([os.path.basename(str(ref)) for ref in self.file_references])
                    html_content += f"""
                    <div class="message user">
                        <div class="message-header">[INITIAL USER PROMPT WITH ATTACHMENTS: {file_list}]</div>
                        <pre>{message['content']}</pre>
                    </div>
                    """
                else:
                    html_content += f"""
                    <div class="message user">
                        <div class="message-header">[INITIAL USER PROMPT]</div>
                        <pre>{message['content']}</pre>
                    </div>
                    """
            elif message["role"] == "user" and i > 1:  # The continuation user prompts
                continuation_user_count += 1
                html_content += f"""
                <div class="message user">
                    <div class="message-header">[CONTINUATION USER PROMPT #{continuation_user_count}]</div>
                    <pre>{message['content']}</pre>
                </div>
                """
            elif message["role"] == "assistant":
                assistant_count += 1
                html_content += f"""
                <div class="message assistant">
                    <div class="message-header">[ASSISTANT #{assistant_count}]</div>
                    <pre>{message['content']}</pre>
                </div>
                """
        
        # Close the HTML document
        html_content += """
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def reset_conversation(self):
        """Reset the conversation while keeping the system prompt."""
        self.conversation_history = [{"role": "system", "content": self.system_prompt}]
        self.chat_history = []
        self.execution_count = 0


def main():
    """Main function to run the sequential prompt executor."""
    try:
        # Check if API key is available
        if not os.getenv("GEMINI_API_KEY"):
            logging.error("GEMINI_API_KEY not found in environment variables.")
            print("Error: GEMINI_API_KEY not found in environment variables.")
            print("Please create a .env file with your API key or set it in your environment.")
            return
        
        # Load the system prompt from system_prompt.txt
        try:
            with open("system_prompt.txt", "r", encoding="utf-8") as f:
                system_prompt = f.read()
            logging.info("System prompt loaded from system_prompt.txt")
            print("System prompt loaded from system_prompt.txt")
        except (FileNotFoundError, IOError) as e:
            logging.error("Error loading system prompt: %s", str(e))
            print(f"Error: Could not load system_prompt.txt - {str(e)}")
            return
        
        # Create the Attachments folder if it doesn't exist
        attachments_folder = "Attachments"
        if not os.path.exists(attachments_folder):
            os.makedirs(attachments_folder)
            print(f"Created attachments folder: {attachments_folder}")
            print("Place any files you want to include with your prompt in this folder.")
        else:
            # Count files in the Attachments folder
            file_count = len([f for f in os.listdir(attachments_folder) 
                             if os.path.isfile(os.path.join(attachments_folder, f))])
            if file_count > 0:
                print(f"Found {file_count} files in the Attachments folder.")
            else:
                print("No files found in the Attachments folder.")

        # Create the executor with default values
        model_name = "gemini-2.0-flash-thinking-exp-01-21"
        max_iterations = 25
        executor = SequentialPromptExecutor(
            system_prompt, 
            model_name=model_name, 
            max_iterations=max_iterations,
            attachments_folder=attachments_folder
        )
        
        # Get the initial user prompt from initial_user_prompt.txt
        try:
            with open("initial_user_prompt.txt", "r", encoding="utf-8") as f:
                user_prompt = f.read()
            logging.info("Initial user prompt loaded from initial_user_prompt.txt")
            print("Initial user prompt loaded from initial_user_prompt.txt")
        except (FileNotFoundError, IOError) as e:
            logging.error("Error loading initial user prompt: %s", str(e))
            print(f"Error: Could not load initial_user_prompt.txt - {str(e)}")
            return
        
        # Execute the chain
        print("\nExecuting prompt chain. This may take a while...")
        logging.info("Starting prompt chain execution")
        start_time = time.time()
        _, final_solution = executor.execute_chain(user_prompt)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Display execution stats
        logging.info("Execution complete in %.2f seconds with %d steps", execution_time, executor.execution_count)
        print(f"\nExecution complete! Took {execution_time:.2f} seconds.")
        print(f"Completed {executor.execution_count} execution steps.")
        
        # Save the full conversation to conversation_history.txt
        try:
            with open("conversation_history.txt", "w", encoding="utf-8") as f:
                f.write(executor.get_full_conversation_text())
            logging.info("Conversation saved to conversation_history.txt")
            print("Conversation saved to conversation_history.txt")
        except IOError as e:
            logging.error("Error saving conversation history: %s", str(e))
            print(f"Error saving conversation history: {str(e)}")

        try:
            html_conversation = executor.generate_conversation_html()
            with open("conversation_history.html", "w", encoding="utf-8") as f:
                f.write(html_conversation)
            logging.info("HTML conversation saved to conversation_history.html")
            print("HTML conversation saved to conversation_history.html")
        except IOError as e:
            logging.error("Error saving HTML conversation history: %s", str(e))
            print(f"Error saving HTML conversation history: {str(e)}")

        # Handle the solution results
        try:
            if final_solution is not None:
                print("\nFinal solution found and processed!")
                
                # Save the original text to final_solution.txt
                try:
                    with open("final_solution.txt", "w", encoding="utf-8") as f:
                        f.write(final_solution.get("original_text", "No final solution text found."))
                    logging.info("Final solution text saved to final_solution.txt")
                except IOError as e:
                    logging.error("Error saving final solution text: %s", str(e))
                    print(f"Error saving final solution text: {str(e)}")
                
                # Print a summary of the rich content found
                print("\nRich content summary:")
                print(f"- Saved to directory: {Path(executor.output_dir).absolute()}")
                
                # Markdown and HTML output
                if "markdown_path" in final_solution:
                    print(f"- Markdown solution: {Path(final_solution['markdown_path']).absolute()}")
                
                if "html_path" in final_solution:
                    print(f"- HTML solution: {Path(final_solution['html_path']).absolute()}")
                    print("  Open this file in your browser to view the formatted solution.")
                
                # Code blocks
                if final_solution.get("has_code", False):
                    print(f"- Found {len(final_solution['code_blocks'])} code blocks:")
                    for code_block in final_solution["code_blocks"]:
                        print(f"  * {code_block['language']} code: {Path(code_block['file_path']).absolute()}")
                
                # Images
                if final_solution.get("has_images", False):
                    print(f"- Found {len(final_solution['image_paths'])} images:")
                    for img in final_solution["image_paths"]:
                        print(f"  * Image {img['index']+1}: {Path(img['file_path']).absolute()}")
                
                # Charts
                if final_solution.get("has_charts", False):
                    print(f"- Found {len(final_solution['chart_data'])} chart data sets:")
                    for chart in final_solution["chart_data"]:
                        print(f"  * Chart data {chart['index']+1}: {Path(chart['file_path']).absolute()}")
                        
                print("\nFinal solution text and all rich content have been saved.")
                print("You can view the solution by opening the HTML file in your browser.")
                
            else:
                print("\nNo final solution was found in the responses.")
                try:
                    with open("final_solution.txt", "w", encoding="utf-8") as f:
                        f.write("No final solution was found after processing.")
                        logging.info("Empty final solution notification saved to final_solution.txt")
                except IOError as e:
                    logging.error("Error saving empty final solution notification: %s", str(e))
                    print(f"Error saving empty final solution notification: {str(e)}")

        except Exception as e:
            logging.error("Error processing final solution: %s", str(e))
            print(f"Error processing final solution: {str(e)}")
            # Attempt to save any error information
            try:
                with open("Solution_processing_error.txt", "w", encoding="utf-8") as f:
                    f.write(f"Error occurred while processing the solution: {str(e)}")
                print("Error details saved to solution_processing_error.txt")
            except IOError as io_error:
                print(f"Could not save error details to file: {str(io_error)}")
            
    except Exception as e:
        logging.error("An error occurred in main: %s", str(e))
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()