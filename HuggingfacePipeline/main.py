import os
# The class `HuggingFacePipeline` was deprecated in LangChain 0.0.37 and will be removed in 1.0.
# from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline

# ./cache/ 경로에 다운로드 받도록 설정
os.environ["TRANSFORMERS_CACHE"] = "./cache/"
os.environ["HF_HOME"] = "./cache/"

model_id = "beomi/llama-2-ko-7b"  # 사용할 모델의 ID를 지정합니다.
tokenizer = AutoTokenizer.from_pretrained(
    model_id
)  # 지정된 모델의 토크나이저를 로드합니다.

model = AutoModelForCausalLM.from_pretrained(model_id)  # 지정된 모델을 로드합니다.

# 텍스트 생성 파이프라인을 생성하고, 최대 생성할 새로운 토큰 수를 10으로 설정합니다.
pipe = pipeline("text-generation", model=model,
                tokenizer=tokenizer, max_new_tokens=512)

# HuggingFacePipeline 객체를 생성하고, 생성된 파이프라인을 전달합니다.
# hf = HuggingFacePipeline(pipeline=pipe)
hf = HuggingFacePipeline(pipeline=pipe)

template = """Answer the following question in Korean.
#Question: 
{question}

#Answer: """  # 질문과 답변 형식을 정의하는 템플릿
prompt = PromptTemplate.from_template(template)  # 템플릿을 사용하여 프롬프트 객체 생성

# 프롬프트와 언어 모델을 연결하여 체인 생성
chain = prompt | hf | StrOutputParser()

question = "대한민국의 수도는 어디야?"  # 질문 정의

print(
    chain.invoke({"question": question})
)  # 체인을 호출하여 질문에 대한 답변 생성 및 출력


# ======================================================================================================================
# GPU Inference
gpu_llm = HuggingFacePipeline.from_model_id(
    model_id = "beomi/llama-2-ko-7b",  # 사용할 모델의 ID를 지정합니다.
    task = "text-generation",  # 수행할 작업을 설정합니다. 여기서는 텍스트 생성입니다.
    # 사용할 GPU 디바이스 번호를 지정합니다. "auto"로 설정하면 accelerate 라이브러리를 사용합니다.
    # device와 device_map은 함께 지정해서는 안 되며, 예기치 않은 동작을 유발할 수 있습니다.
    device = -1, # cpu 사용...
    # 파이프라인에 전달할 추가 인자를 설정합니다. 여기서는 생성할 최대 토큰 수를 10으로 제한합니다.
    pipeline_kwargs = {"max_new_tokens": 64},
)

gpu_chain = prompt | gpu_llm  # prompt와 gpu_llm을 연결하여 gpu_chain을 생성합니다.

# 프롬프트와 언어 모델을 연결하여 체인 생성
gpu_chain = prompt | gpu_llm | StrOutputParser()

question = "대한민국의 수도는 어디야?"  # 질문 정의

# 체인을 호출하여 질문에 대한 답변 생성 및 출력
print(gpu_chain.invoke({"question": question}))


# ======================================================================================================================
# Batch GPU Inference
gpu_llm = HuggingFacePipeline.from_model_id(
    model_id = "beomi/llama-2-ko-7b",  # 사용할 모델의 ID를 지정합니다.
    task = "text-generation",  # 수행할 작업을 설정합니다.
    device = -1,  # GPU 디바이스 번호를 지정합니다. -1은 CPU를 의미합니다.
    batch_size = 2,  # 배치 크기s를 조정합니다. GPU 메모리와 모델 크기에 따라 적절히 설정합니다.
    model_kwargs = {
        "temperature": 0,
        "max_length": 256,
    },  # 모델에 전달할 추가 인자를 설정합니다.
)

# 프롬프트와 언어 모델을 연결하여 체인을 생성합니다.
gpu_chain = prompt | gpu_llm.bind(stop=["\n\n"])

questions = []
for i in range(4):
    # 질문 리스트를 생성합니다.
    questions.append({"question": f"숫자 {i} 이 한글로 뭐에요?"})

answers = gpu_chain.batch(questions)  # 질문 리스트를 배치 처리하여 답변을 생성합니다.
for answer in answers:
    print(answer)  # 생성된 답변을 출력합니다.