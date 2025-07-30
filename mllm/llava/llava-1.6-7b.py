from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from PIL import Image
import torch

def main():
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf" # mistral <> vicuna
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id, 
        device_map="auto",
        torch_dtype=torch.float16  # 메모리 절약을 위해 추가
    )
    processor = LlavaNextProcessor.from_pretrained(model_id)

    img = Image.open("test_image.png")
    question = "What is in this image?"

    prompt = f"[INST] <image>\n{question} [/INST]"
    inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=500,
            do_sample=False,  # 안정적인 출력을 위해 추가
            temperature=0.0   # 결정적 출력을 위해 추가
        )
    
    # 입력 부분을 제외하고 새로 생성된 텍스트만 디코딩
    generated_text = processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    print("모델 답변:", generated_text)

if __name__ == "__main__":
    main()


# huggingface-cli download llava-hf/llava-v1.6-mistral-7b-hf --resume-download