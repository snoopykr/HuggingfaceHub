import os
from langchain_huggingface import HuggingFaceEndpoint
from huggingface_hub import login, logout
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


HUGGINGFACEHUB_API_TOKEN = "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"

login(token=HUGGINGFACEHUB_API_TOKEN)

template = """<|system|>
You are a helpful assistant.<|end|>
<|user|>
{question}<|end|>
<|assistant|>"""

prompt = PromptTemplate.from_template(template)

# 사용할 모델의 저장소 ID를 설정합니다.
repo_id = "microsoft/Phi-3-mini-4k-instruct"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,  # 모델 저장소 ID를 지정합니다.
    max_new_tokens=256,  # 생성할 최대 토큰 길이를 설정합니다.
    temperature=0.1,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,  # 허깅페이스 토큰
)

# LLMChain을 초기화하고 프롬프트와 언어 모델을 전달합니다.
chain = prompt | llm | StrOutputParser()
# 질문을 전달하여 LLMChain을 실행하고 결과를 출력합니다.
response = chain.invoke({"question": "what is the capital of South Korea?"})
print(response)


# ======================================================================================================================
# Inference Endpoint URL을 아래에 설정합니다.
# hf_endpoint_url = "https://slcalzucia3n7y3g.us-east-1.aws.endpoints.huggingface.cloud"
#
# llm = HuggingFaceEndpoint(
#     # 엔드포인트 URL을 설정합니다.
#     endpoint_url=hf_endpoint_url,
#     max_new_tokens=512,
#     temperature=0.01,
# )
#
# # 주어진 프롬프트에 대해 언어 모델을 실행합니다.
# llm.invoke(input="대한민국의 수도는 어디인가요?")


# ======================================================================================================================
# A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
# Human: {prompt}
# Assistant:

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
        ),
        ("user", "Human: {question}\nAssistant: "),
    ]
)

chain = prompt | llm | StrOutputParser()


print(chain.invoke("대한민국의 수도는?"))

logout()