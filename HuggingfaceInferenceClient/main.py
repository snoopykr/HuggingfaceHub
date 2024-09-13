import json

from huggingface_hub import InferenceClient

# LLM의 repository ID 설정
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# InferenceClient 객체를 생성하여 모델을 지정
llm_client = InferenceClient(
    model = repo_id,
    timeout = 120,  # 초 단위로 timeout 설정
)


# LLM을 호출하는 함수 정의
def call_llm(inference_client: InferenceClient, prompt: str):
    # Inference client를 사용하여 모델에 텍스트 생성 요청을 보냄
    response = inference_client.post(
        json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 1000},
            "task": "text-generation",
        },
    )

    return json.loads(response.decode())[0]["generated_text"]


# 함수를 호출하여 테스트 프롬프트로 모델을 실행하고 출력 결과 확인
print(call_llm(llm_client, "Why is the sky blue"))

