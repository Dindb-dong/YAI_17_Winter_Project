# pip install torch transformers opencv-python pillow numpy google-genai openai python-dotenv
# 혹시 파이토치 보안 관련 에러 뜨면... 이거 해주면 됨 
# pip install --upgrade torch torchvision torchaudio
import os
import cv2
import torch
import json
import datetime
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, Blip2Processor, Blip2ForConditionalGeneration
from typing import List, Dict, Union, Tuple
from google import genai
import json
import re
from dotenv import load_dotenv

# ==========================================
# Gemini API 설정 (AdaptiveSearchEngine 내부 혹은 외부에 선언)
# ==========================================
# .env 파일 로드
load_dotenv()

# API 키는 환경변수
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY, http_options=genai.types.HttpOptions(api_version="v1"))
if client is not None:
    print("Gemini Client initialized successfully")
else:
    print("Gemini Client initialization failed")
    exit()
# # 현재 사용 가능한 모든 모델 리스트 출력
# for model in client.models.list():
#     print(f"Model Name: {model.name}, Supported Methods: {model.supported_actions}")
    
# ==========================================
# 1. Model Manager (CLIP & BLIP-2)
# ==========================================
class ModelManager:
    def __init__(self, use_blip=False, device=None):
        print("Initializing ModelManager...")
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading models on {self.device}...")

        # Load CLIP (Base Model)
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", 
            use_fast=True  # 이 옵션을 추가하면 Rust 기반의 빠른 전처리기를 사용합니다.
        )
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_model.eval()
        print("CLIP Model loaded")

        # Load BLIP-2 (Refinement Model) - Optional
        self.use_blip = use_blip
        self.blip_processor = None
        self.blip_model = None
        
        if self.use_blip:
            print("Loading BLIP-2 (this might take memory)...")
            self.blip_processor = Blip2Processor.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                use_fast=True)
            self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
            ).to(self.device)
            self.blip_model.eval()
            print("BLIP-2 Model loaded")
            

    def get_clip_scores(self, images: List[Image.Image], text_queries: List[str]) -> np.ndarray:
        """
        Computes cosine similarity matrix between images and texts.
        Returns: (n_images, n_queries) numpy array
        """
        inputs = self.clip_processor(text=text_queries, images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            # 특징 벡터(Embedding)를 직접 가져와서 정규화 후 코사인 유사도 계산
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
            
            # 코사인 유사도 (0.0 ~ 1.0)
            cosine_sim = torch.matmul(image_embeds, text_embeds.T)
        return cosine_sim.cpu().numpy()

    def generate_caption(self, image: Image.Image) -> str:
        """Generates caption using BLIP-2"""
        if not self.use_blip:
            return ""
        inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.blip_model.generate(**inputs)
        return self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    def get_text_features(self, text_list: List[str]):
        """텍스트를 CLIP 벡터로 변환 (텍스트 간 유사도 비교용)"""
        inputs = self.clip_processor(text=text_list, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """두 텍스트 간의 시맨틱 유사도 계산"""
        feat1 = self.get_text_features([text1])
        feat2 = self.get_text_features([text2])
        sim = torch.matmul(feat1, feat2.T)
        return float(sim.cpu().numpy())

# ==========================================
# 2. Video Processor
# ==========================================
class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        print("Video processor initialized")
    def extract_window_frames(self, start_sec, end_sec, num_samples_q, window_idx=None, total_windows=None) -> List[Image.Image]:
        """
        Extracts 'q' frames uniformly from the window [start_sec, end_sec].
        """
        frames = []
        start_frame = int(start_sec * self.fps)
        end_frame = int(end_sec * self.fps)
        
        # Uniform sampling indices
        indices = np.linspace(start_frame, end_frame - 1, num_samples_q, dtype=int)
        
        for frame_num, idx in enumerate(indices, 1):
            if window_idx is not None and total_windows is not None:
                print(f"  [Window {window_idx}/{total_windows}] 프레임 추출 중: {frame_num}/{num_samples_q}", end='\r')
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
            else:
                # Padding with black frame if read fails
                frames.append(Image.new('RGB', (224, 224), (0, 0, 0)))
        
        if window_idx is not None:
            print()  # 줄바꿈
        return frames

    def get_timestamp_str(self, seconds):
        return str(datetime.timedelta(seconds=int(seconds)))

# ==========================================
# 3. Adaptive Search Engine (Core Logic)
# ==========================================
class AdaptiveSearchEngine:
    def __init__(self, model_manager: ModelManager, video_processor: VideoProcessor):
        self.mm = model_manager
        self.vp = video_processor

    def split_query(self, text_query: str) -> tuple[list[str], str]:
        """
        Gemini API를 활용하여 한국어 쿼리를 시간 순서에 따른 동작 리스트로 분할합니다.
        API 호출 실패 시, 한국어 접속사 규칙을 기반으로 분할을 시도합니다.
        
        Returns:
            tuple: (분할된 쿼리 리스트, 분할 이유 설명)
        """
        print(f"Thinking with Gemini (Korean Mode)... Query: '{text_query}'")
        
        # ---------------------------------------------------------
        # 1. Gemini API 호출 (Primary Strategy)
        # ---------------------------------------------------------
        prompt = f"""
        당신은 비디오 검색 시스템을 위한 언어 분석기입니다.
        사용자의 검색어(Query)가 시간 순서에 따른 여러 동작(Sequence)을 포함하고 있다면, 이를 분할하세요.
        그 후, 그것들을 CLIP 모델이 가장 잘 인식할 수 있는 정교한 영어 문장으로 번역한 후 JSON 객체로 반환하세요.
        
        [규칙]
        1. 문맥상 시간의 흐름(예: ~하고 나서, ~한 뒤에, ~하다가)이 있으면 동작 단위로 쪼개세요.
        2. 단순한 묘사나 단일 동작이면 요소가 1개인 리스트를 반환하세요.
        3. 1회 이상 쪼개지 마세요. (요소는 2개 이하여야 합니다.)
        4. 반환값은 반드시 순수 JSON 객체 형식이어야 합니다. (Markdown 제외)
        5. 분할된 문장은 검색이 잘 되도록 기본형(예: '달리고' -> '달리는 사람')이나 명확한 문장으로 다듬어주면 더 좋습니다.
        6. 분할 여부와 그 이유를 함께 설명하세요.

        [반환 형식]
        {{
        "results": [
            {{"ko": "한국어 동작 1", "en": "English description 1"}},
            {{"ko": "한국어 동작 2", "en": "English description 2"}}
        ],
        "reason": "분할 및 번역 이유"
        }}

        [예시 1]
        Query: "공을 던지고 나서 넘어지는 사람"
        Output: {{
            "results": [
                {{"ko": "공을 던지는 사람", "en": "Person throwing a ball"}},
                {{"ko": "넘어지는 사람", "en": "Person falling down"}}
            ],
            "reason": "시간 순서를 나타내는 '던지고 나서'를 기준으로 두 개의 연속 동작으로 분할했습니다."
        }}

        [예시 2]
        Query: "웃고 있는 아기"
        Output: {{
            "results": [
                {{"ko": "웃고 있는 아기", "en": "Baby laughing"}}
            ],
            "reason": "단일 동작/상태를 묘사하는 쿼리로 분할하지 않았습니다."
        }}

        [예시 3]
        Query: "요리를 하다가 불이 나서 당황하는 남자"
        Output: {{
            "results": [
                {{"ko": "요리를 하는 남자", "en": "Man cooking"}},
                {{"ko": "불이 나서 당황하는 남자", "en": "Man panicking after fire"}}
            ],
            "reason": "'하다가'를 기준으로 시간 순서상 선행 동작과 후행 동작으로 분할했습니다."
        }}

        [실제 입력]
        Query: "{text_query}"
        """

        try:
            # API 호출
            response = client.models.generate_content(
                model='models/gemini-2.0-flash-lite',
                contents=prompt
            )
            result_text = response.text.strip()
            
            if result_text.startswith("```"):
                result_text = re.sub(r"```(json)?", "", result_text).strip()
                result_text = re.sub(r"```", "", result_text).strip()
            
            # JSON 파싱
            result = json.loads(result_text)
            
            if isinstance(result, dict) and "results" in result:
                actions = [x["en"] for x in result["results"]]
                reason = result.get("reason", "분할 완료")
                
                print(f" -> Gemini Split Result (EN): {actions} \n Reason: {reason}")
                return actions, f"[Gemini API] {reason}"
            
        except Exception as e:
            print(f"Gemini API Error: {e}. Switching to Fallback.")

        # ---------------------------------------------------------
        # 2. 규칙 기반 Fallback (한국어 접속사 처리)
        # ---------------------------------------------------------
        # API가 실패하거나 응답이 이상할 경우 작동하는 비상 로직입니다.
        # 한국어에서 순서를 나타내는 흔한 표현들을 기준으로 자릅니다.
        
        delimiters = [
            " 그리고 ", " 다음에 ", " 그 후 ", " 그 뒤에 ", " 나서 ", 
            " 하다가 ", "다가 "
        ]
        
        # 가장 먼저 발견되는 구분자로 1회만 분할 시도 (복잡성 방지)
        for delim in delimiters:
            if delim in text_query:
                parts = text_query.split(delim)
                # 빈 문자열 제거 및 공백 정리
                clean_parts = [p.strip() for p in parts if p.strip()]
                if len(clean_parts) > 1:
                    print(f" -> Rule-based Split Result: {clean_parts}")
                    return clean_parts, f"[Rule-based] '{delim.strip()}' 구분자를 기준으로 분할했습니다."

        # 분할 실패 시 원본 그대로 반환
        print("Final split result: ", [text_query])
        return [text_query], "[Rule-based] 시간 순서를 나타내는 표현이 없어 분할하지 않았습니다."

    def calculate_sequential_score(self, frames, sub_queries):
        """
        [변곡점 탐지 로직]
        쿼리가 A -> B로 나뉘었을 때, 프레임 시퀀스 내에서 최적의 분할 지점을 찾아
        (A유사도 + B유사도)가 최대가 되는 점수를 반환합니다.
        Returns: (max_score, scores_matrix, best_split_index)
        """
        # (q_frames, 2_sub_queries) matrix
        scores_matrix = self.mm.get_clip_scores(frames, sub_queries) 
        q_len = len(frames)
        max_score = -1.0
        best_split = -1
        
        # Linear Scan to find Change Point
        # 최소 20% 지점부터 80% 지점 사이에서 분할 시도
        start_idx = int(q_len * 0.2)
        end_idx = int(q_len * 0.8)

        if len(sub_queries) == 2:
            score_A = scores_matrix[:, 0] # Similarity curve for Query A
            score_B = scores_matrix[:, 1] # Similarity curve for Query B
            
            for t in range(start_idx, end_idx):
                # t 시점까지는 A, t 이후는 B
                avg_A = np.mean(score_A[:t])
                avg_B = np.mean(score_B[t:])
                combined_score = (avg_A + avg_B) / 2
                
                if combined_score > max_score:
                    max_score = combined_score
                    best_split = t
        else:
            max_score = np.mean(np.max(scores_matrix, axis=1))

        return float(max_score), scores_matrix, best_split

    def normalize_score(self, raw_score: float) -> float:
        """
        CLIP 코사인 유사도를 0-100 점수로 변환
        - 0.2 이하: 0점
        - 0.45 이상: 100점
        """
        lower_bound = 0.20
        upper_bound = 0.45
        
        # 정규화 계산
        normalized = (raw_score - lower_bound) / (upper_bound - lower_bound) * 100
        
        # 0~100 사이로 클리핑
        return float(np.clip(normalized, 0, 100))

    def search(self, original_query, sub_queries, p_sec, q_frames, k_top, weight_clip = 0.7, weight_semantic = 0.3):
        """
        Adaptive Search Engine 실행 메인 로직
        - 1. CLIP 기반 1차 검색 (Coarse-grained Search)
        - 2. BLIP-2 기반 2차 보정 (Fine-grained Refinement)
        - 3. 최종 점수 산출 및 정렬
        """
        is_sequential = len(sub_queries) > 1
        
        all_windows = []
        step_size = p_sec  # 윈도우가 겹치지 않도록 수정
        current_time = 0.0
        
        # 전체 윈도우 개수 계산
        total_windows = int((self.vp.duration - p_sec) / step_size) + 1
        
        print(f"\n{'='*60}")
        print(f"[검색 시작] 총 {total_windows}개 윈도우 처리 예정 (윈도우 크기: {p_sec}초, 프레임 샘플: {q_frames}개)")
        print(f"{'='*60}\n")
        
        # 1. CLIP 기반 1차 검색 (Coarse-grained Search)
        window_idx = 0
        current_top_window = None
        
        while current_time + p_sec <= self.vp.duration:
            window_idx += 1
            end_time = current_time + p_sec
            
            print(f"[Window {window_idx}/{total_windows}] 처리 중: {self.vp.get_timestamp_str(current_time)} - {self.vp.get_timestamp_str(end_time)}")
            
            frames = self.vp.extract_window_frames(current_time, end_time, q_frames, window_idx, total_windows)
            
            if is_sequential:
                raw_score, scores_matrix, best_split = self.calculate_sequential_score(frames, sub_queries)
                # 각 프레임별 점수 저장 (시퀀셜의 경우 두 쿼리에 대한 점수)
                frame_scores = {
                    f"query_{i}": scores_matrix[:, i].tolist() 
                    for i in range(len(sub_queries))
                }
                frame_scores["best_split_index"] = int(best_split) if best_split != -1 else None
            else:
                raw_scores_matrix = self.mm.get_clip_scores(frames, sub_queries)
                raw_score = float(np.mean(raw_scores_matrix))
                # 각 프레임별 점수 저장
                frame_scores = {
                    f"query_{i}": raw_scores_matrix[:, i].tolist() 
                    for i in range(len(sub_queries))
                }
            clip_score_norm = self.normalize_score(raw_score)
            print(f"  -> 정규화 CLIP 점수: {clip_score_norm:.2f}")
            
            window_data = {
                "start": current_time,
                "end": end_time,
                "timestamp": f"{self.vp.get_timestamp_str(current_time)} - {self.vp.get_timestamp_str(end_time)}",
                "raw_score": raw_score,           # 참고용 원본 점수 
                "clip_score_norm": clip_score_norm,    # 정규화된 점수 (JSON 저장용)
                "frame_scores": frame_scores,  # 프레임별 점수 추가
                "mid_frame": frames[len(frames)//2] # 보정을 위해 중간 프레임 저장
            }
            all_windows.append(window_data)
            
            # 현재까지 최고 점수 윈도우 추적
            if current_top_window is None or clip_score_norm > current_top_window['clip_score_norm']:
                current_top_window = window_data
                print(f"  ⭐ 새로운 Top 윈도우 발견! ({current_top_window['timestamp']})\n")
            else:
                print(f"  [현재 Top] {current_top_window['timestamp']} (점수: {current_top_window['clip_score_norm']:.4f})\n")
            
            current_time += step_size

        # CLIP 점수 기준 상위 K개 선별
        print(f"\n{'='*60}")
        print(f"[1차 검색 완료] CLIP 점수 기준 상위 {k_top}개 후보 선별")
        print(f"{'='*60}")
        
        all_windows.sort(key=lambda x: x["clip_score_norm"], reverse=True)
        top_k_candidates = all_windows[:k_top]
        
        for idx, item in enumerate(top_k_candidates, 1):
            print(f"{idx}. {item['timestamp']} - 점수: {item['clip_score_norm']:.4f}")

        # 2. BLIP-2 기반 2차 보정 (Fine-grained Refinement)
        if self.mm.use_blip:
            print(f"\n{'='*60}")
            print(f"[2차 보정 시작] BLIP-2를 사용하여 상위 {k_top}개 후보 보정 중...")
            print(f"{'='*60}\n")
            
            for idx, item in enumerate(top_k_candidates, 1):
                print(f"[후보 {idx}/{k_top}] {item['timestamp']}")
                
                # A. BLIP-2로 프레임 설명(Caption) 생성
                generated_caption = self.mm.generate_caption(item['mid_frame'])
                item['blip_caption'] = generated_caption
                
                # B. 사용자 쿼리와 생성된 캡션 간의 의미적 유사도 계산 (Text-to-Text)
                semantic_sim = self.mm.compute_text_similarity(original_query, generated_caption)
                item['semantic_consistency'] = semantic_sim

                # C. 최종 점수 산출 (앙상블)
                item['final_score'] = (item['clip_score_norm'] * weight_clip) + (semantic_sim * weight_semantic)
                
                print(f"  -> 생성된 캡션: {generated_caption}")
                print(f"  -> 의미 유사도: {semantic_sim:.4f}")
                print(f"  -> 최종 점수: {item['final_score']:.4f}\n")
            
            # 보정된 최종 점수로 다시 정렬
            top_k_candidates.sort(key=lambda x: x.get('final_score', x['clip_score_norm']), reverse=True)
            
            print(f"{'='*60}")
            print(f"[최종 순위]")
            print(f"{'='*60}")
            for idx, item in enumerate(top_k_candidates, 1):
                print(f"{idx}. {item['timestamp']} - 최종 점수: {item.get('final_score', item['clip_score_norm']):.4f}")
        
        print()

        # 결과 저장 전 이미지 객체 삭제 (메모리 확보)
        for item in top_k_candidates:
            if 'mid_frame' in item: del item['mid_frame']
            
        return top_k_candidates

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    # --- Configurations ---
    VIDEO_PATH = "sample_video.mp4" # 준비된 비디오 파일 경로
    QUERY = "바닥에 떨어진 신용카드"
    # "바닥에 떨어지는 카드를 보고 난감한 표정을 짓는 남자" # 테스트 쿼리
    
    # Experiments Parameters
    p_list = [2.0, 4.0]      # 윈도우 크기 (초)
    q_list = [24, 48]         # 샘플링 프레임 수
    k_list = [3, 5]          # Top-K 개수
    USE_BLIP = input("BLIP-2 사용 여부 (True/False): ")         # BLIP-2 사용 여부 (메모리 주의)
    USE_BLIP = USE_BLIP.lower() == 'true'
    if USE_BLIP:
        print("BLIP-2 사용 중...")
    else:
        print("BLIP-2 사용 안 함...")
    WEIGHT_CLIP = 0.7
    WEIGHT_SEMANTIC = 0.3
    USE_LOOP = False         # 반복 실행 여부

    # Initialize
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file '{VIDEO_PATH}' not found. Please place a dummy video.")
        return

    model_manager = ModelManager(use_blip=USE_BLIP)
    video_processor = VideoProcessor(VIDEO_PATH)
    engine = AdaptiveSearchEngine(model_manager, video_processor)
    
    print(f"\n[쿼리 분석] '{QUERY}'")
    sub_queries, split_reason = engine.split_query(QUERY)
    print(f"[분할된 쿼리] {sub_queries}\n")
    
    # Experiment Loop
    if USE_LOOP:
        for p in p_list:
            for q in q_list:
                for k in k_list:
                    print(f"\n--- Running Experiment: p={p}, q={q}, k={k} ---")
                    
                    # Perform Search
                    results = engine.search(QUERY, sub_queries, p, q, k, WEIGHT_CLIP, WEIGHT_SEMANTIC)
                    
                    # Construct Filename
                    model_name = "CB" if USE_BLIP else "Clip"
                    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{model_name}_{p}, {q}, {k}, {timestamp_str}.json"
                    
                    # Output Data Structure
                    output_data = {
                        "meta": {
                            "video_path": VIDEO_PATH,
                            "query": QUERY,
                            "sub_queries": sub_queries,
                            "split_reason": split_reason,
                            "parameters": {"p": p, "q": q, "k": k},
                            "model": model_name,
                            "timestamp": timestamp_str
                        },
                        "results": results
                    }
                    
                    # Save to JSON
                    with open(filename, "w", encoding='utf-8') as f:
                        json.dump(output_data, f, indent=4, ensure_ascii=False)
                    
                    print(f"\n[저장 완료] {filename}\n")
                    
    else:
        results = engine.search(QUERY, sub_queries, p_list[0], q_list[0], k_list[0], WEIGHT_CLIP, WEIGHT_SEMANTIC)
        model_name = "CB" if USE_BLIP else "Clip"
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{p_list[0]}, {q_list[0]}, {k_list[0]}, {timestamp_str}.json"
        
        # Output Data Structure
        output_data = {
            "meta": {
                "video_path": VIDEO_PATH,
                "query": QUERY,
                "sub_queries": sub_queries,
                "split_reason": split_reason,
                "parameters": {"p": p_list[0], "q": q_list[0], "k": k_list[0]},
                "model": model_name,
                "timestamp": timestamp_str
            },
            "results": results
        }
        
        # Save to JSON
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"[검색 완료] 결과가 {filename}에 저장되었습니다.")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()