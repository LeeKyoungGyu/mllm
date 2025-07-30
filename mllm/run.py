import glob
import os
import json
from datetime import datetime
import sys

sys.path.append('llava')
from m_llava_15_7b import LlavaModel

def main():
    # 설정
    image_folder = "png_images"  # PNG 이미지들이 있는 폴더
    output_file = "results.json"
    
    # 질문 설정 (옵션 선택)
    
    # 옵션 1: 모든 이미지에 같은 질문
    question = "What do you see in this image? Describe it accurately."
    
    # 옵션 2: 여러 질문 (각 이미지마다 순서대로 적용)
    # questions = [
    #     "What do you see in this image?",
    #     "Are there any visual illusions in this image?",
    #     "Describe the shapes and patterns you observe."
    # ]
    
    # 옵션 3: 이미지별 맞춤 질문
    # custom_questions = {
    #     "img_001.png": "Are the two lines the same length?",
    #     "img_002.png": "Do you see a triangle?",
    #     # 나머지는 기본 질문 사용
    # }
    
    print("🚀 착시 이미지 테스트 시작!")
    print(f"📁 이미지 폴더: {image_folder}")
    
    # 이미지 파일 목록 가져오기
    image_paths = glob.glob(os.path.join(image_folder, "*.png"))
    image_paths.sort()  # 파일명 순으로 정렬
    
    if not image_paths:
        print(f"❌ {image_folder}에서 PNG 파일을 찾을 수 없습니다.")
        return
    
    print(f"📊 총 {len(image_paths)}개 이미지 발견")
    
    # 모델 초기화
    model = LlavaModel()
    
    try:
        # 테스트 실행
        print("\n🔍 테스트 실행 중...")
        results = model.process_multiple_images(image_paths, question)
        
        # 결과 정리
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        print(f"\n📈 결과 요약:")
        print(f"✅ 성공: {len(successful_results)}개")
        print(f"❌ 실패: {len(failed_results)}개")
        
        # 실패한 경우 출력
        if failed_results:
            print(f"\n❌ 실패한 이미지들:")
            for fail in failed_results:
                print(f"  - {os.path.basename(fail['image_path'])}: {fail['error']}")
        
        # 결과 저장
        output_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model': 'llava-1.5-7b-hf',
                'total_images': len(image_paths),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'question_used': question
            },
            'results': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 결과 저장 완료: {output_file}")
        
        # 몇 개 결과 미리보기
        print(f"\n👀 결과 미리보기 (처음 3개):")
        for i, result in enumerate(successful_results[:3]):
            img_name = os.path.basename(result['image_path'])
            answer = result['answer'][:100] + "..." if len(result['answer']) > 100 else result['answer']
            print(f"\n{i+1}. {img_name}")
            print(f"   Q: {result['question']}")
            print(f"   A: {answer}")
        
    finally:
        # 메모리 정리
        model.cleanup()

if __name__ == "__main__":
    main()