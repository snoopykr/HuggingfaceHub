import os
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

# ./cache/ 경로에 다운로드 받도록 설정
os.environ["TRANSFORMERS_CACHE"] = "./cache/"
os.environ["HF_HOME"] = "./cache/"


llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 256,
        "top_k": 50,
        "temperature": 0.1,
    },
)
print(llm.invoke("Hugging Face is"))


# ======================================================================================================================
template = """Summarizes TEXT in simple bullet points ordered from most important to least important.
TEXT:
{text}

KeyPoints: """

# 프롬프트 템플릿 생성
prompt = PromptTemplate.from_template(template)

# 체인 생성
chain = prompt | llm

text = """A Large Language Model (LLM) like me, ChatGPT, is a type of artificial intelligence (AI) model designed to understand, generate, and interact with human language. These models are "large" because they're built from vast amounts of text data and have billions or even trillions of parameters. Parameters are the aspects of the model that are learned from training data; they are essentially the internal settings that determine how the model interprets and generates language. LLMs work by predicting the next word in a sequence given the words that precede it, which allows them to generate coherent and contextually relevant text based on a given prompt. This capability can be applied in a variety of ways, from answering questions and composing emails to writing essays and even creating computer code. The training process for these models involves exposing them to a diverse array of text sources, such as books, articles, and websites, allowing them to learn language patterns, grammar, facts about the world, and even styles of writing. However, it's important to note that while LLMs can provide information that seems knowledgeable, their responses are generated based on patterns in the data they were trained on and not from a sentient understanding or awareness. The development and deployment of LLMs raise important considerations regarding accuracy, bias, ethical use, and the potential impact on various aspects of society, including employment, privacy, and misinformation. Researchers and developers continue to work on ways to address these challenges while improving the models' capabilities and applications."""
print(f"입력 텍스트:\n\n{text}")

# chain 실행
response = chain.invoke({"text": text})

# 결과 출력
print(response)