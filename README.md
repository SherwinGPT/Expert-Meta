
# Sequential Prompt Executor for Gemini API

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project provides a Python-based framework for interacting with Google's Gemini API. It enables sequential execution of prompts, allowing for iterative problem-solving and multi-step reasoning tasks. The framework supports code execution, rich content processing (images, charts, code), conversation history management, and robust error handling. It's designed to automate complex tasks using an agentic workflow where the output of one prompt becomes the input for the next.


## Table of Contents

1.  [Features](#features)
2.  [Project Structure](#project-structure)
3.  [Getting Started](#getting-started)
4.  [Configuration](#configuration)
5.  [Important Notes and Considerations](#important-notes-and-considerations)
6.  [License](#license)
7.  [Contributing](#contributing)


## Features

*   **Sequential Prompt Execution:**  Runs a series of prompts, where the output of each prompt is used as input for the next, enabling iterative problem-solving.
*   **System Prompt:**  Uses a system prompt (loaded from `system_prompt.txt`) to guide the overall behavior of the LLM.
*   **User Prompt:**  Starts with an initial user prompt (loaded from `initial_user_prompt.txt`).
*   **File Attachments:**  Supports attaching files from an `Attachments` folder to the initial prompt, leveraging the Gemini File API.
*   **Code Execution:**  Includes optional code execution tools (enabled by default).  The Gemini model can generate code and, if enabled, the tool will execute it.
*   **Rich Content Processing:**  Extracts and saves code blocks, images (base64 encoded), and chart data (JSON) from the LLM's responses.  
*   **Output Formats:**  Generates text file (`final_solution.txt`), Markdown (`final_solution.md`), and HTML (`final_solution.html`) versions of the final solution.
*   **Conversation History:**  Maintains a complete conversation history, saved to `conversation_history.txt` and `conversation_history.html`.
*   **Error Handling:**  Includes robust error handling and logging to `executor.log`.
*   **Configurable Model Parameters:** Allows customization of model parameters like `temperature`, `top_p`, `top_k`, and `max_output_tokens`.
*   **Safety Settings:**  Uses safety settings to filter responses (currently all set to `BLOCK_NONE`, but configurable).
*   **HTML Output**: Produces formatted HTML reports for both the final solution and the entire conversation history for easy viewing in a web browser.


## Project Structure
```
.
├── Attachments/ <- Place files to be attached to the initial prompt here.
├── Solution_Output/ <- Output directory for final solution and extracted content.
│ ├── code_block_.ext <- Extracted code blocks.
│ ├── image_.ext <- Extracted images.
│ ├── chart_data_*.json <- Extracted chart data.
│ ├── final_solution.md <- Markdown version of the final solution.
│ └── final_solution.html <- HTML version of the final solution.
├── .env <- (Not included in repository) Store your GEMINI_API_KEY here.
├── conversation_history.txt <- Full conversation history (text format).
├── conversation_history.html <- Full conversation history (HTML format).
├── executor.log <- Log file for debugging.
├── final_solution.txt <- Final solution text.
├── initial_user_prompt.txt <- Initial user prompt.
├── system_prompt.txt <- System prompt to guide the LLM.
├── expert_meta.py <- Main script.
└── README.md <- This file.
```


## Getting Started

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/SherwinGPT/Expert-Meta.git
    cd Expert-Meta
    ```

2.  **Create the Conda Environment:**

    ```bash
    conda env create -f environment.yml
    ```

    **Alternative: Using pip**

    For users who prefer pip over conda, you can install the required packages using:
    
    ```bash
    pip install -r requirements.txt
    ```

3.  **Activate the Conda Environment:**

    ```bash
    conda activate expert_meta-env
    ```

4.  **API Key Setup:**

    *   Obtain an API key from Google's Gemini API.
    *   Create a file named `.env` in the root directory of the project.
    *   Add your API key to the `.env` file:

        ```
        GEMINI_API_KEY=your_api_key_here
        ```

5.  **Prepare Prompts and Attachments:**

    *   Modify `system_prompt.txt` to define the system prompt. This prompt sets the overall context and instructions for the Gemini model. You must indicate in this prompt that the model is to provide instructions at the end of every response for the next agent to complete their task until a final output is reached. You must add the below statement to the end of your system prompt for the program to know when to stop the iteration by finding the words "FINAL SOLUTION" in the final output: 

    Final Output Format: You MUST present your final output in the following format:  
    FINAL SOLUTION:
    """
    [Final Solution]
    """
    *   Modify `initial_user_prompt.txt` to provide the initial user prompt for the task you wish the model to complete. The more detailed the better.
    *   Place any files you want to attach to the initial prompt in the `Attachments` folder.  These will be automatically uploaded and referenced in the prompt.

6.  **Run the Script:**

    ```bash
    python expert_meta.py
    ```

7. **View the Results**
    * The Solution_Output folder will contain the final solution, including any extracted content.
    * Open the `Solution_Output/final_solution.html` and `conversation_history.html` in your browser.


## Configuration

The `SequentialPromptExecutor` class constructor accepts several parameters for configuration:

*   `system_prompt` (str):  The system prompt.  Required.
*   `model_name` (str):  The Gemini model to use.  Defaults to `"gemini-2.0-flash-thinking-exp-01-21"`.
*   `max_iterations` (int):  The maximum number of sequential prompt executions.  Defaults to 20.  This prevents infinite loops.
*   `temperature` (float):  Controls the randomness of the output.  Defaults to 1.
*   `top_p` (float):  Nucleus sampling parameter.  Defaults to 0.95.
*   `top_k` (int):  Top-k sampling parameter.  Defaults to 40.
*   `max_output_tokens` (int):  Maximum number of tokens in the generated response. Defaults to 8192.
*   `enable_code_execution` (bool): Whether to enable the code execution tool.  Defaults to `True`.
*   `attachments_folder` (str) - Folder path containing attachments. Defaults to `Attachments`

You can modify these parameters when creating an instance of `SequentialPromptExecutor` in `expert_meta.py`.


## Important Notes and Considerations

*   **API Key Security:**  Never commit your `.env` file or your API key to a public repository.  Add `.env` to your `.gitignore` file.
*   **Rate Limits:**  Be aware of the Gemini API rate limits.  You may need to implement error handling and retries if you encounter rate limit errors.
*   **Cost:**  Using the Gemini API may incur costs.  Monitor your usage and billing.
*   **Code Execution:** The `enable_code_execution` flag controls whether the tool is *available* to the model. The *model* decides whether to use it.  Carefully review the generated code before executing it, especially if interacting with external systems or sensitive data.  The provided code does *not* have any sandboxing or security measures.
*   **Prompt Engineering:**  The quality of the results depends heavily on the quality of the system prompt and initial user prompt.  Experiment with different prompts to achieve the desired behavior.  The prompts should guide the model to include the "FINAL SOLUTION" text when it has finished.
*   **Model Limitations:**  The Gemini models are powerful, but they are not perfect.  They may sometimes produce incorrect, nonsensical, or unexpected results.
*   **File Upload Limits**: There are limits to the number and size of files that can be uploaded.
*   **Output Directory:** The `Solution_Output` directory is created if it doesn't exist.  The script will overwrite any existing files with the same name in that directory.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Contributing

While we are not currently accepting code contributions, we welcome bug reports and feature suggestions via GitHub Issues.
