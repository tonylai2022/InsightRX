from transformers import pipeline
from langchain.prompts import PromptTemplate
import torch

def get_summary(text):
    # Load the model locally
    summarizer = pipeline(
        "summarization",
        model="google/flan-t5-small",
        device=0 if torch.cuda.is_available() else -1
    )

    prompt_template = """Summarize the following text in concise English:

    {content}

    Summary:"""
    
    prompt = PromptTemplate(
        input_variables=["content"],
        template=prompt_template
    )
    
    # Format the prompt with the text
    formatted_prompt = prompt.format(content=text[:2000])
    
    # Generate summary
    summary = summarizer(formatted_prompt, max_length=512, min_length=30, do_sample=False)[0]['summary_text']
    return summary