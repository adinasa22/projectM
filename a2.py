#from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama

# Initialize the local LLM (Ollama)
llm = Ollama(model='mistral')  # Replace with your local model name

# Define a prompt template for summarization
prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text:\n\n{text}\n\nSummary:"
)

# Create a chain for summarization
summarization_chain = LLMChain(llm=llm, prompt=prompt)

# Input text to summarize
input_text = """
Artificial intelligence (AI) is a branch of computer science that aims to create machines 
that can perform tasks that would typically require human intelligence. These tasks include 
learning, reasoning, problem-solving, perception, and language understanding.
"""

# Run the summarization chain
summary = summarization_chain.run(text=input_text)

# Print the summary
print("Summary:", summary)