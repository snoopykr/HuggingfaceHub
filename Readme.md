### Ollama Model 등록

[참조] https://www.youtube.com/watch?v=_7wnNTNQO1M

```bash
snoopy_kr@iMac models % ollama create Llama-3-Open-Ko-8B -f Modelfile
```

### Model file

[참조] https://github.com/ollama/ollama/blob/main/docs/modelfile.md

```bash
FROM  llama-3-Korean-Bllossom-8B.gguf

SYSTEM """당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다. You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner. 모든 대답은 한국어(Korean)으로 대답해주세요."""


TEMPLATE """{{- if .System }}
<s>{{ .System }}</s>
{{- end }}
<s>Human:
{{ .Prompt }}</s>
<s>Assistant:
"""

PARAMETER temperature 0.6
PARAMETER num_predict 3000
PARAMETER num_ctx 4096
PARAMETER stop <s>
PARAMETER stop </s>
PARAMETER stop <|eot_id|>
```

### requirements.txt to poetry
(huggingfacehub-py3.11) snoopy_kr@iMac HuggingfaceHub % poetry add $(cat requirements.txt)


### 정규식 '/site-packages’로 시작되는 라인을 삭제하는 방법
pattern = re.compile(r'^/site-packages.*$', re.MULTILINE)


### Build LLM Apps Easily
https://flowiseai.com/


### 뤼튼 무료 한글 GPT
https://wrtn.ai/


### HuggingfaceBERT 
- gpu가 없어서 에러 발생... 해결 방법 모색중...

```python
model_id = "beomi/Llama-3-Open-Ko-8B-Instruct-preview"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto", # TypeError: BFloat16 is not supported on MPS
    device_map="auto",
    # torch_dtype=torch.float16, # 메모리 에러...)
```

### KnowHow #1
langchain-huggingface 패키지는 pytorch가 없으면 설치가 되지 않는다.

torch = "2.2.2"
torchaudio = "2.2.2"
torchvision = "0.17.2"
langchain-huggingface = "^0.0.3"

### Update #1
- The class `HuggingFacePipeline` was deprecated in LangChain 0.0.37 and will be removed in 1.0. An updated version of the class exists in the langchain-huggingface package and should be used instead. To use it run `pip install -U langchain-huggingface` and import as `from langchain_huggingface import HuggingFacePipeline`.
  hf = HuggingFacePipeline(pipeline=pipe)

```python
# from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
```