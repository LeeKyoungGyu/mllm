from transformers import LlavaForConditionalGeneration, LlavaProcessor
from PIL import Image
import torch

def main():
    model_id = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map="auto")
    processor = LlavaProcessor.from_pretrained(model_id)

    img = Image.open("test_image.png")
    question = "What is in this image?"

    prompt = f"USER: <image>\n{question} ASSISTANT:"
    inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=500)
    answer = processor.decode(output[0], skip_special_tokens=True)

    print("모델 답변:", answer)

if __name__ == "__main__":
    main()

# huggingface-cli download llava-hf/llava-1.5-13b-hf --resume-download