from langchain.llms import Ollama
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
import os

# Initialize the Ollama LLM
llm = Ollama(model="mistral")

# Define a tool to summarize text
def summarize_text(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text:\n\n{text}"
    )
    return llm(prompt.format(text=content))

summarize_tool = Tool(
    name="Summarizer",
    func=lambda file_path: summarize_text(file_path),
    description="Summarizes the content of a CSV file."
)

# Initialize the agent
tools = [summarize_tool]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Automate summarization for all text files in a directory
def automate_summarization(directory):
    for file_name in os.listdir(directory):
        if file_name.endswith(".csv"):
            file_path = os.path.join(directory, file_name)
            print(f"Summarizing {file_name}...")
            summary = agent.run(file_path)
            print(f"Summary for {file_name}:\n{summary}\n")

# Example usage
directory_path = "/Users/adityagupta/Desktop/projectm/projectM"  # Replace with your directory path
automate_summarization(directory_path)