from transformers import LlavaForConditionalGeneration, LlavaProcessor
from PIL import Image
import torch
import os

class LlavaModel:
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf"):
        """LLaVA 모델 초기화"""
        print(f"🔄 모델 로딩 중: {model_id}")
        
        # 디바이스 확인
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️ 사용 디바이스: {device}")
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.processor = LlavaProcessor.from_pretrained(model_id)
        self.model_id = model_id
        print("✅ 모델 로딩 완료!")
    
    def generate_response(self, image_path, question, max_tokens=500):
        """단일 이미지에 대한 응답 생성"""
        try:
            # 이미지 존재 확인
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            
            # 이미지 로드
            img = Image.open(image_path).convert("RGB")
            
            # 이미지 크기 확인 (너무 크면 리사이즈)
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            # 프롬프트 구성
            prompt = f"USER: <image>\n{question} ASSISTANT:"
            
            # 입력 준비
            inputs = self.processor(
                text=prompt, 
                images=img, 
                return_tensors="pt"
            ).to(self.model.device)
            
            # 추론 실행
            with torch.no_grad():
                output = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens,
                    do_sample=False,  # 일관성을 위해 greedy decoding
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    temperature=1.0,  # 명시적 설정
                    use_cache=True    # 속도 향상
                )
            
            # 응답 디코딩
            full_response = self.processor.decode(output[0], skip_special_tokens=True)
            
            # "ASSISTANT:" 이후 부분만 추출
            if "ASSISTANT:" in full_response:
                answer = full_response.split("ASSISTANT:")[-1].strip()
            else:
                answer = full_response.strip()
            
            # 빈 응답 체크
            if not answer:
                answer = "[모델이 빈 응답을 생성했습니다]"
            
            return {
                'success': True,
                'answer': answer,
                'image_path': image_path,
                'question': question
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'image_path': image_path,
                'question': question
            }
    
    def process_multiple_images(self, image_paths, questions):
        """여러 이미지 처리"""
        results = []
        
        # questions가 문자열이면 모든 이미지에 동일한 질문 적용
        if isinstance(questions, str):
            questions = [questions] * len(image_paths)
        
        # questions가 리스트지만 길이가 1이면 모든 이미지에 적용
        elif len(questions) == 1:
            questions = questions * len(image_paths)
        
        # 길이가 맞지 않으면 에러
        elif len(questions) != len(image_paths):
            raise ValueError(f"이미지 개수({len(image_paths)})와 질문 개수({len(questions)})가 맞지 않습니다.")
        
        print(f"🎯 총 {len(image_paths)}개 이미지 처리 시작")
        
        for i, (img_path, question) in enumerate(zip(image_paths, questions)):
            print(f"🖼️ [{i+1}/{len(image_paths)}] 처리 중: {os.path.basename(img_path)}")
            
            result = self.generate_response(img_path, question)
            results.append(result)
            
            if result['success']:
                # 응답 일부 미리보기
                preview = result['answer'][:50] + "..." if len(result['answer']) > 50 else result['answer']
                print(f"✅ 완료: {preview}")
            else:
                print(f"❌ 실패: {result['error']}")
            
            # 메모리 정리 (선택사항)
            if i % 10 == 0 and i > 0:
                torch.cuda.empty_cache()
        
        return results
    
    def cleanup(self):
        """메모리 정리"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("🗑️ 메모리 정리 완료")