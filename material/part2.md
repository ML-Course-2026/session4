# Summary: Exploring the Gemini API (Part 1/2)

This material is directly related to the mini project. 

As a recap, the project consists of three phases:

- **Phase 1**: Implement a mock GUI that returns hard-coded responses.
- **Phase 2**: Connect the system to a Large Language Model (LLM) to generate responses dynamically. This can be done by:
  - Using open-source LLMs from Hugging Face  
  - Calling an LLM API

We will use both approaches. Since using an API is more straightforward, this session will focus on how to implement it using practical examples and in-class activities. After class, you are expected to adapt your mini project to integrate with an LLM API.

<!-- Next session will cover more theory about how LLMs work internally, which will better prepare you to use open-source models from [Hugging Face](https://huggingface.co/models). -->

**Workflow in brief:**

There are several options for calling an LLM API. In this course, we will focus on methods that allow you to experiment with powerful models at no cost:

- [GitHub Models](https://github.com/marketplace/models) — provides access to APIs like OpenAI, Mistral, and others  
- [Gemini models](https://ai.google.dev/gemini-api/docs/models) from Google — provides access to the Gemini family of LLMs. 

> [!NOTE]  
> Here's the recommended Workflow:
> 
> 1. Generate an API key  
> 2. Test with basic examples from the API documentation to verify functionality  
> 3. Test the LLM API with prompts relevant to your project — expect unstructured output  
> 4. Refine your prompt to generate structured output (examples will be provided)  
> 5. Update your Gradio GUI to display the structured output from the LLM API


> [!TIP]
> This material includes multiple use cases for interacting with multimodal LLMs from the Gemini family. Not all examples may be directly relevant to your project, but it's beneficial to try them out in case they become useful in the future.


---

## Introduction and Rationale

We begin our exploration of Large Language Models (LLMs) by interacting with them through an Application Programming Interface (API). This approach allows us to leverage powerful, pre-trained models without managing the underlying infrastructure.

### Why Start with an API?

Using an API like Google's Gemini API provides immediate access to sophisticated AI capabilities. It allows us to focus on *how to use* the model effectively for tasks like text generation, summarization, or structured data extraction, and integrate these capabilities into applications (such as the Gradio UIs we are building). This practical experience is valuable before delving into the model's internal workings. The underlying mechanisms, such as **tokenization**, **embeddings**, and the **model pipeline**, will be covered in the next session to provide a deeper understanding necessary for customization and advanced use cases.

### Why Use the Gemini API Specifically?

For this stage, the Gemini API serves as a practical example due to its:
-   **Advanced Capabilities:** Access to models demonstrating strong performance in various tasks.
-   **Multimodality:** Ability to process text, images, audio, and video inputs.
-   **Developer Experience:** A well-documented Python SDK (`google-genai`) simplifies interaction.
-   **Structured Output:** Features for generating formatted JSON output, useful for application integration.
-   **Accessibility:** includes a free tier suitable for learning and experimentation.

### Why Explore Hugging Face / Local Models Later?

While APIs are convenient, direct interaction with models (often sourced from platforms like Hugging Face, covered in Part 3) offers distinct advantages:
-   **Flexibility/Choice:** Access to thousands of specialized models.
-   **Privacy/Control:** Data remains within your environment when run locally.
-   **Customization:** Enables fine-tuning on specific datasets.
-   **Offline Use:** Models can run without internet connectivity.
-   **Cost Efficiency:** Potentially lower cost at high scale compared to API calls.

Understanding both API-based and direct model usage allows for informed decisions based on project requirements.


---
## Setup and Initialization

The first steps involve setting up the environment. Gemini uses API keys for authentication. here's a walk through how to create an API key, and using it in colab

### Create an API key

You can [create](https://aistudio.google.com/app/apikey) your API key using Google AI Studio with a single click.  

Remember to treat your API key like a password. Do not accidentally save it in a notebook or source file you later commit to GitHub. 
In Google Colab, it is recommended to store your key in Colab Secrets. here's how to

### Add your key to Colab Secrets

Add your API key to the Colab Secrets manager to securely store it.

1. Open your Google Colab notebook and click on the 🔑 **Secrets** tab in the left panel.
   
   <img src="https://storage.googleapis.com/generativeai-downloads/images/secrets.jpg" alt="The Secrets tab is found on the left panel." width=50%>

2. Create a new secret with the name `GOOGLE_API_KEY`.
3. Copy/paste your API key into the `Value` input box of `GOOGLE_API_KEY`.
4. Toggle the button on the left to allow notebook access to the secret.


### Setup your API Key

You create a client using your API key, but instead of pasting your key into the notebook, you'll read it from Colab Secrets.

```python
from google.colab import userdata
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
```

*   **Explanation:** Accessing the Gemini API requires authentication. An API key is a unique secret credential that identifies your project or account to Google Cloud. This code retrieves the key securely stored as a Colab Secret named `GOOGLE_API_KEY`. Storing keys as secrets is crucial for security, preventing them from being exposed directly in the notebook code. You need to generate your own API key from Google AI Studio or Google Cloud Console and store it in Colab secrets for this code to work.

### Install SDK

```
%pip install google-genai -q
```

*   **Explanation:** This command installs or updates the necessary Python library, `google-genai`. This library, provided by Google, contains the functions and classes needed to interact with the Gemini API easily from Python code.  The `-q` makes the installation process quiet (less output)

<!-- The `-U` flag ensures you get the latest version, and `-q` makes the installation process quiet (less output). -->

### Initialize SDK client

```python
from google import genai
from google.genai import types # types is used for specific configurations later

# Initialize the client with the API key
client = genai.Client(api_key=GOOGLE_API_KEY)
```

*   **Explanation:** Here, we import the installed library (`genai`). The core of the interaction is the `Client` object. We create an instance of this client, passing our `GOOGLE_API_KEY` for authentication. This `client` object will be used for all subsequent calls to the API (e.g., generating content, managing files).

### Choose a model

Now choose a model. The Gemini API offers different models that are optimized for specific use cases, for more information check [Gemini models](https://ai.google.dev/gemini-api/docs/models)

```python
MODEL_ID = "gemini-2.5-flash" # @param ["gemini-2.5-flash-lite", "gemini-3.1-flash-lite-preview"] {"allow-input":true, isTemplate: true}
```

*   **Explanation:** The Gemini family includes several models optimized for different tasks, performance levels, and input modalities. This line selects which specific model variant we want to use for our requests. `gemini-2.5-flash` is chosen here as a generally capable and efficient model. Other options like `gemini-3.1-flash-lite-preview` offer different features or performance characteristics. The model ID is stored in the `MODEL_ID` variable for easy reference in later API calls. The comment `# @param ...` enables an interactive dropdown menu in Colab for selecting the model.

## Send Text Prompts

The most basic interaction involves sending a text prompt and receiving a text response.

### Basic Text Generation

```python
from IPython.display import Markdown # Used for nice formatting of output

# Make the API call
response = client.models.generate_content(
    model=MODEL_ID,
    contents="What's the largest planet in our solar system?"
)

# Display the response text
Markdown(response.text)
```

-   **Explanation:** This code demonstrates a simple text-in, text-out request.
    -   `client.models.generate_content()`: This is the primary method for sending prompts to the selected model.
    -   `model=MODEL_ID`: Specifies which Gemini model to use (the one selected earlier).
    -   `contents=...`: This argument holds the input prompt. Here, it's a simple string.
    -   The API call returns a `response` object. The generated text is typically accessed via `response.text`.
    -   `Markdown(response.text)` displays the output using Markdown formatting for better readability in environments like Colab or Jupyter.

### Text Generation with Gradio Interface


```
%pip install gradio -q # Install Gradio if not already installed
```


```python
import gradio as gr
# from IPython.display import Markdown # Already imported

# Define the function that calls the Gemini API
def ask_model(prompt):
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt
    )
    # Return the text part of the response
    # Gradio's Markdown component will render this
    return response.text

# Create the Gradio interface
gr.Interface(
    fn=ask_model, # The function to call when the user interacts
    inputs=gr.Textbox(lines=2, placeholder="Ask me something...", label="Prompt"), # Input component
    outputs=gr.Markdown(label="Response"), # Output component (displays Markdown)
    title="Gemini Model Q&A",
    description="Ask the Gemini model a question and see its response!"
).launch() # Launch the web UI
```

-   **Explanation:** This section wraps the basic text generation functionality in a simple web interface using the Gradio library.
    -   `import gradio as gr`: Imports the Gradio library.
    -   `ask_model(prompt)`: This function takes a `prompt` string (from the Gradio textbox) as input, calls the `client.models.generate_content` method just like before, and returns the `response.text`.
    -   `gr.Interface(...)`: This creates the user interface.
        -   `fn=ask_model`: Specifies the Python function to execute.
        -   `inputs=gr.Textbox(...)`: Defines the input field as a multi-line textbox.
        -   `outputs=gr.Markdown(...)`: Defines the output area, specifying that the returned text should be rendered as Markdown.
        -   `title`, `description`: Set the UI titles.
    -   `.launch()`: Starts the interactive Gradio web server and displays the UI. This allows users to interact with the Gemini model through a simple form instead of just running code cells.

## Send Multimodal Prompts

Gemini models can understand prompts containing multiple types of input, such as images and text together.

### Multimodal Generation (Image + Text)

```python
import requests # To download the image
import pathlib # To handle file paths
from PIL import Image # To work with the image object

# Download an image
IMG_URL = "https://storage.googleapis.com/generativeai-downloads/data/jetpack.png"
img_bytes = requests.get(IMG_URL).content
img_path = pathlib.Path('jetpack.png')
img_path.write_bytes(img_bytes)

# Open the image using PIL
image = Image.open(img_path)
image.thumbnail([512,512]) # Resize for display convenience

# Display the image in the notebook (optional)
from IPython.display import display
display(image)

# Send image and text prompt together
response = client.models.generate_content(
    model=MODEL_ID, # Ensure model supports multimodal, e.g., gemini-2.5-flash
    contents=[
        image, # Pass the PIL Image object directly
        "Write a short and engaging blog post based on this picture." # Text part
    ]
)

# Display the text response
Markdown(response.text)
```

-   **Explanation:** This demonstrates sending both an image and text in a single prompt.
    -   The code first downloads an image from a URL and saves it locally.
    -   It opens the image using the Python Imaging Library (PIL).
    -   The key part is the `contents` argument in `generate_content`. It's now a *list* containing multiple parts: the `image` object (PIL format is supported directly by the SDK) and the text prompt string.
    -   The model processes both inputs to generate the response (in this case, a blog post about the image).

### Multimodal Generation with Gradio Interface

```python
import gradio as gr
# Other necessary imports (requests, pathlib, PIL.Image) assumed from previous cell

def generate_blog(image_input, prompt):
    # The 'image_input' from Gradio is already a PIL Image object if type="pil"
    if image_input is None:
        return "Please upload an image."

    # No need to save/reload if Gradio provides PIL object directly
    pil_image = image_input
    pil_image.thumbnail([512, 512]) # Optional resize for consistency

    # Call Gemini with the PIL image and text prompt
    try:
        response = client.models.generate_content(
            model=MODEL_ID, # Ensure model supports multimodal
            contents=[
                pil_image,
                prompt
            ]
        )
        return response.text
    except Exception as e:
        return f"Error processing request: {e}"


# Gradio UI for multimodal input
gr.Interface(
    fn=generate_blog,
    inputs=[
        gr.Image(type="pil", label="Upload an image"), # Image input component
        gr.Textbox(lines=2, placeholder="e.g., Write a blog post about this...", label="Prompt") # Text input
    ],
    outputs=gr.Markdown(label="Generated Blog Post"), # Text output
    title="AI Blog Generator from Image",
    description="Upload an image and let the Gemini model write a short blog post for you!"
).launch()
```

-   **Explanation:** This wraps the multimodal functionality in a Gradio interface.
    -   `generate_blog(image_input, prompt)`: This function now takes two arguments: `image_input` (from the Gradio image component) and `prompt` (from the textbox).
    -   `gr.Image(type="pil", ...)`: This Gradio input component allows users to upload an image. Setting `type="pil"` ensures that the `image_input` argument passed to our function is already a PIL Image object, simplifying the code.
    -   The rest of the function calls `generate_content` with the image and text, returning the generated text to be displayed in the `gr.Markdown` output component.
    -   *(Note on Scope):* While this example successfully uses Gradio for *image input*, recall the earlier point: reliably displaying *generated* images or audio from the model within Gradio *output* components can be complex and is considered outside the core scope of the required lab exercises. We focus on text/Markdown output for simplicity.

## Configure Model Parameters

API calls can include parameters to control the generation process.

### Generation with Custom Configuration

```python
# Make sure 'types' is imported: from google.genai import types

response = client.models.generate_content(
    model=MODEL_ID,
    contents="Tell me how the internet works, but pretend I'm a puppy who only understands squeaky toys.",
    config=types.GenerateContentConfig(
        temperature=0.4,       # Controls randomness (lower = more deterministic)
        candidate_count=1,     # Number of response candidates to generate
        seed=5,                # For reproducible results (if possible with model)
        max_output_tokens=100, # Maximum length of the response
    )
)

print(response.text)
```

*   **Explanation:** This demonstrates how to influence the model's output beyond just the prompt.
    *   The `config` argument takes a `GenerateContentConfig` object (from `google.genai.types`).
    *   Inside `GenerateContentConfig`, various parameters can be set:
        *   `temperature`: Controls creativity vs. focus. Lower values (e.g., 0.2) make output more predictable; higher values (e.g., 0.9) make it more random/creative.
        *   `top_p`, `top_k`: Alternative methods to control randomness by limiting the pool of tokens the model considers at each step.
        *   `max_output_tokens`: Limits response length.
        *   `stop_sequences`: Causes the model to stop generating if it produces one of these strings.
        *   `seed`: Allows for potentially reproducible outputs, though not guaranteed across all models/versions.
        *   `presence_penalty`, `frequency_penalty`: Help control repetitiveness.
    *   Experimenting with these parameters is key to tuning the model's behavior for specific needs.

### Configuration Control with Gradio Interface

```python
import gradio as gr
# Assume 'client', 'MODEL_ID', 'types' are available

def generate_response(prompt, temperature, top_p, top_k, seed, max_tokens, stop_seq, presence_penalty, frequency_penalty):
    # Prepare stop sequences list
    stop_sequences = [stop_seq] if stop_seq else None # Handle empty input

    # Create the configuration object from Gradio inputs
    config = types.GenerateContentConfig(
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        candidate_count=1,
        seed=int(seed) if seed is not None else None, # Handle potential None input
        max_output_tokens=int(max_tokens),
        stop_sequences=stop_sequences,
        presence_penalty=float(presence_penalty),
        frequency_penalty=float(frequency_penalty),
    )

    # Call the model
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=config
        )
        return response.text
    except Exception as e:
         return f"Error processing request: {e}"

# Gradio Interface with sliders and number inputs for parameters
gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(label="Prompt", lines=3, placeholder="e.g., Explain quantum physics to a cat..."),
        gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Temperature"),
        gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Top-p"),
        gr.Slider(1, 100, value=20, step=1, label="Top-k"),
        gr.Number(value=5, label="Seed", precision=0), # Use precision=0 for integer
        gr.Number(value=100, label="Max Output Tokens", precision=0),
        gr.Textbox(label="Stop Sequence (optional)", placeholder="e.g., STOP!"),
        gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Presence Penalty"),
        gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Frequency Penalty")
    ],
    outputs=gr.Markdown(label="Model Response"),
    title="Gemini Prompt with Custom Config",
    description="Customize generation settings and interact with the Gemini model."
).launch()
```

*   **Explanation:** This Gradio app allows interactive experimentation with the generation parameters.
    *   The `generate_response` function now takes the prompt and all the configuration parameters as arguments. These will come from the corresponding Gradio input components.
    *   Inside the function, it constructs the `GenerateContentConfig` object using the values passed from the UI. Note the type conversions (e.g., `float()`, `int()`) as Gradio inputs might be strings or floats that need to match the types expected by `GenerateContentConfig`.
    *   The `gr.Interface` uses various input components like `gr.Slider` and `gr.Number` to provide intuitive controls for the numerical parameters.
    * More about [top-p](https://rentry.co/samplers#top-p)
    * More about [top-k](https://rentry.co/samplers#top-k)



## Configure Safety Filters

The API includes safety filters to block potentially harmful content. These can be adjusted.

```python
# Assume 'client', 'MODEL_ID', 'types' are available

prompt = """
    Write a list of 2 disrespectful things that I might say to  friend after stubbing my toe in the dark.
"""

# Define safety settings configuration
# Example: Block only high-probability dangerous content
safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
    # Can add settings for other categories like HARASSMENT, HATE_SPEECH, SEXUALLY_EXPLICIT
]

# Call generate_content with safety_settings in the config
# Note: Safety settings are part of GenerateContentConfig
try:
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            safety_settings=safety_settings,
            # Can combine with other config parameters like temperature if needed
        ),
        # Alternative: safety_settings can sometimes be passed as a direct argument too
        # safety_settings=safety_settings
    )
    #Markdown(response.text)
    print(response.text)    
except Exception as e:
    # Responses might be blocked entirely if they violate stricter settings.
    # Check response.prompt_feedback for safety ratings/blocks
    print(f"An error or block occurred: {e}")
    # if hasattr(response, 'prompt_feedback'): print(response.prompt_feedback)
```

*   **Explanation:** This code demonstrates how to customize the API's built-in safety mechanisms.
    *   `safety_settings` is a list of `SafetySetting` objects. Each object specifies a `category` (e.g., `HARM_CATEGORY_DANGEROUS_CONTENT`) and a `threshold` (e.g., `BLOCK_NONE`, `BLOCK_LOW_AND_ABOVE`, `BLOCK_MEDIUM_AND_ABOVE`, `BLOCK_ONLY_HIGH`).
    *   These settings are passed within the `GenerateContentConfig` object (or sometimes directly as an argument) to the `generate_content` call.
    *   Adjusting these thresholds changes the likelihood that the API will block prompts or responses it deems potentially harmful according to its classifiers. It's important to configure these appropriately for the application's use case and target audience. If a response is blocked due to safety settings, the API might return an error or an empty response; detailed feedback is often available in `response.prompt_feedback`.

## Start a Multi-turn Chat

The SDK supports conversational interactions where context is maintained across turns.

### Basic Chat Interaction

```python
# Assume 'client', 'MODEL_ID', 'types' are available

# Optional: Define system instructions for the chat persona/behavior
system_instruction="""
  You are an expert software developer and a helpful coding assistant.
  You are able to generate high-quality code in any programming language.
"""

# Configure chat parameters (optional, can include temperature, etc.)
chat_config = types.GenerateContentConfig(
    system_instruction=system_instruction,
    temperature=0.5,
    # other config parameters can go here
)

# Start a new chat session
chat = client.chats.create(
    model=MODEL_ID,
    config=chat_config,
    # History can be pre-filled here if needed: history=[...]
)

# Send the first user message
response = chat.send_message("Write a function that checks if a year is a leap year.")
Markdown(response.text) # Display first response
```

```python
# Send a follow-up message; the chat object maintains history
response = chat.send_message("Okay, write a unit test of the generated function.")
Markdown(response.text) # Display second response
```


*   **Explanation:** This code sets up and conducts a multi-turn conversation.
    *   `system_instruction`: An optional initial instruction defining the AI's persona or core task for the entire chat session.
    *   `chat_config`: A `GenerateContentConfig` can be applied to the chat session, including the system instruction and generation parameters like temperature.
    *   `client.chats.create()`: Initializes a new chat session. It takes the model ID and optional configuration. You can also provide an initial `history` list here to start from a previous conversation.
    *   `chat.send_message()`: Sends a user message to the chat session. The SDK automatically manages the conversation history (previous user messages and model responses) and includes it in subsequent calls to the API, allowing the model to respond contextually.
    *   Each call to `send_message` returns the model's response for that turn.

### Chat Interaction with Gradio Interface

```python
import gradio as gr
# Assume 'client', 'MODEL_ID', 'types' are available

# Note: This Gradio example starts a *new* chat session for *each* interaction.
# For a persistent chat UI, you'd need to manage the 'chat' object state across calls,
# typically using gr.State or external storage, which adds complexity.
# This simplified version demonstrates passing system instructions and a single turn.

def chat_with_assistant(system_instruction, user_prompt, temperature):
    # Define chat config with system instruction and temperature for this turn
    chat_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=float(temperature),
    )

    # Create a *new* chat session for this interaction
    # (No history is carried over from previous interactions in this simple UI)
    try:
        chat = client.chats.create(
            model=MODEL_ID,
            config=chat_config,
        )
        # Send the user's message
        response = chat.send_message(user_prompt)
        return response.text
    except Exception as e:
        return f"Error processing request: {e}"

# Gradio Interface
gr.Interface(
    fn=chat_with_assistant,
    inputs=[
        gr.Textbox(label="System Instruction", lines=3, value="You are an expert software developer and a helpful coding assistant."),
        gr.Textbox(label="Your Message", lines=3, placeholder="e.g., Write a function that checks if a year is a leap year."),
        gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Temperature")
    ],
    outputs=gr.Markdown(label="Assistant Response"),
    title="Chat with Gemini (Custom System Instruction)",
    description="Define how the assistant should behave, then send a prompt to the Gemini model. (Note: Each interaction starts a new chat)."
).launch()
```

*   **Explanation:** This Gradio app provides an interface for interacting with the chat functionality, allowing users to set the system instruction.
    *   The `chat_with_assistant` function takes the system instruction, user prompt, and temperature from the UI.
    *   **Important Limitation:** As noted in the comments and description, this simple Gradio implementation creates a *new chat session* every time the user submits a prompt. It does not maintain conversation history between interactions in the UI. A true chatbot UI in Gradio would require state management (`gr.State`) to keep track of the `chat` object and its history across multiple turns. This example focuses only on demonstrating the passing of system instructions and single-turn interaction via Gradio.

-----

## Useful Links

- [https://ai.google.dev/gemini-api/docs/file-prompting-strategies](https://ai.google.dev/gemini-api/docs/file-prompting-strategies)
- [Gemini API: Getting started with Gemini 2.](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Get_started.ipynb)
- [Gemini API: Authentication Quickstart](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb)
- [Gemini Models](https://ai.google.dev/gemini-api/docs/models) 
- [Gemini QuickStart](https://ai.google.dev/gemini-api/docs/quickstart?lang=python) 
- [Free images](https://unsplash.com/images/stock/public-domain) 


<!-- 
- [How to train a new language model from scratch using Transformers and Tokenizers](https://huggingface.co/blog/how-to-train)  
-->
 