# pip install torch transformers opencv-python pillow numpy google-genai python-dotenv matplotlib
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
import time
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

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
# exit()
    
# ==========================================
# 1. Model Manager (CLIP & BLIP-2)
# ==========================================
class ModelManager:
    def __init__(self, use_blip=False, device=None):
        start_time = time.time()
        print("Initializing ModelManager...")
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading models on {self.device}...")

        # Load CLIP (Base Model)
        clip_start = time.time()
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", 
            use_fast=True  # 이 옵션을 추가하면 Rust 기반의 빠른 전처리기를 사용합니다.
        )
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_model.eval()
        clip_time = time.time() - clip_start
        print(f"CLIP Model loaded ({clip_time:.2f}초)")

        # Load BLIP-2 (Refinement Model) - Optional
        self.use_blip = use_blip
        self.blip_processor = None
        self.blip_model = None
        blip_time = 0.0
        
        if self.use_blip:
            print("Loading BLIP-2 (this might take memory)...")
            blip_start = time.time()
            self.blip_processor = Blip2Processor.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                use_fast=True)
            self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
            ).to(self.device)
            self.blip_model.eval()
            blip_time = time.time() - blip_start
            print(f"BLIP-2 Model loaded ({blip_time:.2f}초)")
        
        self.init_time = time.time() - start_time
        self.clip_load_time = clip_time
        self.blip_load_time = blip_time
        print(f"ModelManager 초기화 완료 (총 {self.init_time:.2f}초)")
            

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
        start_time = time.time()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        self.init_time = time.time() - start_time
        print(f"Video processor initialized ({self.init_time:.2f}초)")
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
        # 타이밍 정보 저장
        self.timing_info = {
            "api_call_time": 0.0,
            "clip_inference_time": 0.0,
            "blip_inference_time": 0.0,
            "frame_extraction_time": 0.0,
            "total_search_time": 0.0
        }

    def _call_gemini_with_retry(self, prompt: str, max_retries: int = 3, timeout: int = 20) -> str:
        """
        Exponential Backoff과 Jitter를 사용한 재시도 로직 (Timeout 적용)
        
        Args:
            prompt: Gemini API에 전달할 프롬프트
            max_retries: 최대 재시도 횟수
            timeout: 각 요청의 최대 대기 시간 (초)
            
        Returns:
            API 응답 텍스트
        """
        import signal
        from contextlib import contextmanager
        
        @contextmanager
        def time_limit(seconds):
            """타임아웃 컨텍스트 매니저"""
            def signal_handler(signum, frame):
                raise TimeoutError(f"API 호출이 {seconds}초를 초과했습니다.")
            
            # macOS/Linux에서만 signal 사용 가능
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, signal_handler)
                signal.alarm(seconds)
                try:
                    yield
                finally:
                    signal.alarm(0)
            else:
                # Windows에서는 단순 timeout
                yield
        
        # 시도할 모델 리스트 (우선순위 순)
        models = [
            'models/gemini-2.0-flash-lite',
            'models/gemini-2.0-flash',
            'models/gemini-2.5-flash-lite'
        ]
        
        for model_idx, model in enumerate(models):
            for attempt in range(max_retries):
                try:
                    # Jitter 추가: 0.1~0.5초 랜덤 지연 (동시 요청 충돌 방지)
                    if attempt > 0:
                        jitter = random.uniform(0.1, 0.5)
                        time.sleep(jitter)
                    
                    if attempt == 0 and model_idx == 0:
                        print(f"  [API] {model} 호출 중... (Timeout: {timeout}초)")
                    else:
                        print(f"  [API] 재시도 {attempt + 1}/{max_retries} (모델: {model}, Timeout: {timeout}초)...")
                    
                    # API 호출 (Timeout 적용)
                    start_time = time.time()
                    try:
                        if hasattr(signal, 'SIGALRM'):
                            with time_limit(timeout):
                                response = client.models.generate_content(
                                    model=model,
                                    contents=prompt
                                )
                        else:
                            # Windows: 단순 호출
                            response = client.models.generate_content(
                                model=model,
                                contents=prompt
                            )
                            elapsed = time.time() - start_time
                            if elapsed > timeout:
                                raise TimeoutError(f"API 호출이 {timeout}초를 초과했습니다.")
                    except TimeoutError as e:
                        print(f"  [API] ⏱️ Timeout ({timeout}초 초과). 다음 모델로 전환...")
                        break  # 다음 모델로
                    
                    elapsed = time.time() - start_time
                    print(f"  [API] ✓ 성공! (소요 시간: {elapsed:.2f}초)")
                    return response.text.strip()
                    
                except Exception as e:
                    error_str = str(e)
                    
                    # 429 Rate Limit 에러 처리
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        print(f"  [API] ⚠ Rate Limit 도달. 다음 모델로 즉시 전환...")
                        break  # 대기하지 않고 바로 다음 모델로
                    
                    # 기타 에러
                    else:
                        print(f"  [API] ✗ 에러 발생: {error_str}")
                        if attempt < max_retries - 1:
                            continue
                        else:
                            if model_idx < len(models) - 1:
                                print(f"  [API] 다음 모델로 전환...")
                                break
                            else:
                                raise Exception(f"모든 모델 시도 실패: {error_str}")
        
        raise Exception("Gemini API 호출 실패: 모든 재시도 소진")

    def split_query(self, text_query: str) -> tuple[list[str], str]:
        """
        Gemini API를 활용하여 한국어 쿼리를 시간 순서에 따른 동작 리스트로 분할합니다.
        API 호출 실패 시, 한국어 접속사 규칙을 기반으로 분할을 시도합니다.
        
        Returns:
            tuple: (분할된 쿼리 리스트, 분할 이유 설명)
        """
        api_start_time = time.time()
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
            # API 호출 (Retry 로직 포함)
            result_text = self._call_gemini_with_retry(prompt)
            
            if result_text.startswith("```"):
                result_text = re.sub(r"```(json)?", "", result_text).strip()
                result_text = re.sub(r"```", "", result_text).strip()
            
            # JSON 파싱
            result = json.loads(result_text)
            
            if isinstance(result, dict) and "results" in result:
                actions = [x["en"] for x in result["results"]]
                reason = result.get("reason", "분할 완료")
                
                api_time = time.time() - api_start_time
                self.timing_info["api_call_time"] = api_time
                print(f" -> Gemini Split Result (EN): {actions} \n Reason: {reason}")
                print(f" -> API 호출 시간: {api_time:.2f}초")
                return actions, f"[Gemini API] {reason}"
            
        except Exception as e:
            print(f"Gemini API Error: {e}. Switching to Fallback.")

        # ---------------------------------------------------------
        # 2. 규칙 기반 Fallback (한국어 접속사 처리)
        # ---------------------------------------------------------
        # API가 실패하거나 응답이 이상할 경우 작동하는 비상 로직입니다.
        # 한국어에서 순서를 나타내는 흔한 표현들을 기준으로 자릅니다.
        
        api_time = time.time() - api_start_time
        self.timing_info["api_call_time"] = api_time
        
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
                    print(f" -> Fallback 처리 시간: {api_time:.2f}초")
                    return clean_parts, f"[Rule-based] '{delim.strip()}' 구분자를 기준으로 분할했습니다."

        # 분할 실패 시 원본 그대로 반환
        print("Final split result: ", [text_query])
        print(f" -> Fallback 처리 시간: {api_time:.2f}초")
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

    def search(self, original_query, sub_queries, p_sec, q_frames, k_top, weight_clip = 0.7, weight_semantic = 0.3, enable_visualization=True, save_path="results"):
        """
        Adaptive Search Engine 실행 메인 로직
        - 1. CLIP 기반 1차 검색 (Coarse-grained Search)
        - 2. BLIP-2 기반 2차 보정 (Fine-grained Refinement)
        - 3. 최종 점수 산출 및 정렬
        
        Args:
            enable_visualization: 실시간 시각화 활성화 여부
            save_path: 시각화 이미지 저장 경로
        """
        search_start_time = time.time()
        
        # 타이밍 초기화
        total_frame_extraction_time = 0.0
        total_clip_inference_time = 0.0
        total_blip_inference_time = 0.0
        
        is_sequential = len(sub_queries) > 1
        
        all_windows = []
        step_size = p_sec  # 윈도우가 겹치지 않도록 수정
        current_time = 0.0
        
        # 전체 윈도우 개수 계산 (정확한 계산)
        total_windows = int(np.ceil(self.vp.duration / step_size))
        
        # 실시간 시각화 초기화
        visualizer = None
        if enable_visualization:
            try:
                visualizer = RealTimeVisualizer(self.vp.duration, k_top, save_path)
                print(f"\n{'='*60}")
                print(f"[📊 실시간 시각화 활성화] 진행 상황을 실시간으로 그래프에 표시합니다!")
                print(f"{'='*60}\n")
            except Exception as e:
                print(f"시각화 초기화 실패: {e}. 시각화 없이 계속 진행합니다.")
                visualizer = None
        
        print(f"\n{'='*60}")
        print(f"[검색 시작] 총 {total_windows}개 윈도우 처리 예정 (윈도우 크기: {p_sec}초, 프레임 샘플: {q_frames}개)")
        print(f"{'='*60}\n")
        
        # 1. CLIP 기반 1차 검색 (Coarse-grained Search)
        window_idx = 0
        current_top_window = None
        
        while current_time < self.vp.duration:
            window_idx += 1
            end_time = min(current_time + p_sec, self.vp.duration)
            
            print(f"[Window {window_idx}/{total_windows}] 처리 중: {self.vp.get_timestamp_str(current_time)} - {self.vp.get_timestamp_str(end_time)}")
            
            # 프레임 추출 시간 측정
            frame_start = time.time()
            frames = self.vp.extract_window_frames(current_time, end_time, q_frames, window_idx, total_windows)
            frame_time = time.time() - frame_start
            total_frame_extraction_time += frame_time
            
            # CLIP 추론 시간 측정
            clip_start = time.time()
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
            clip_time = time.time() - clip_start
            total_clip_inference_time += clip_time
            
            clip_score_norm = self.normalize_score(raw_score)
            print(f"  -> 정규화 CLIP 점수: {clip_score_norm:.2f} (프레임 추출: {frame_time:.2f}초, CLIP 추론: {clip_time:.2f}초)")
            
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
            
            # 실시간 시각화 업데이트
            if visualizer:
                try:
                    visualizer.update({
                        'start': current_time,
                        'end': end_time,
                        'clip_score_norm': clip_score_norm
                    })
                except Exception as e:
                    print(f"  [시각화 경고] 업데이트 실패: {e}")
            
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
                
                # A. BLIP-2로 프레임 설명(Caption) 생성 - 시간 측정
                blip_start = time.time()
                generated_caption = self.mm.generate_caption(item['mid_frame'])
                blip_time = time.time() - blip_start
                total_blip_inference_time += blip_time
                
                item['blip_caption'] = generated_caption
                
                # B. 사용자 쿼리와 생성된 캡션 간의 의미적 유사도 계산 (Text-to-Text)
                semantic_start = time.time()
                semantic_sim = self.mm.compute_text_similarity(original_query, generated_caption)
                semantic_time = time.time() - semantic_start
                total_blip_inference_time += semantic_time
                
                item['semantic_consistency'] = semantic_sim

                # C. 최종 점수 산출 (앙상블)
                item['final_score'] = (item['clip_score_norm'] * weight_clip) + (semantic_sim * weight_semantic)
                
                print(f"  -> 생성된 캡션: {generated_caption}")
                print(f"  -> 의미 유사도: {semantic_sim:.4f}")
                print(f"  -> 최종 점수: {item['final_score']:.4f}")
                print(f"  -> BLIP-2 처리 시간: {blip_time + semantic_time:.2f}초\n")
            
            # 보정된 최종 점수로 다시 정렬
            top_k_candidates.sort(key=lambda x: x.get('final_score', x['clip_score_norm']), reverse=True)
            
            print(f"{'='*60}")
            print(f"[최종 순위]")
            print(f"{'='*60}")
            for idx, item in enumerate(top_k_candidates, 1):
                print(f"{idx}. {item['timestamp']} - 최종 점수: {item.get('final_score', item['clip_score_norm']):.4f}")
        else:
            # BLIP-2 없을 때도 최종 순위 출력
            print(f"\n{'='*60}")
            print(f"[최종 순위]")
            print(f"{'='*60}")
            for idx, item in enumerate(top_k_candidates, 1):
                print(f"{idx}. {item['timestamp']} - 점수: {item['clip_score_norm']:.4f}")
        
        print()

        # 실시간 시각화 최종 업데이트 (BLIP-2 보정 여부와 관계없이 여기서 호출)
        if visualizer:
            try:
                visualizer.finalize(top_k_candidates)
                print(f"\n{'='*60}")
                print(f"[📊 시각화 완료] 최종 Top-{k_top} 결과가 빨간색 별(★)로 표시되었습니다!")
                print(f"{'='*60}\n")
            except Exception as e:
                print(f"[시각화 오류] 최종 업데이트 실패: {e}")
                import traceback
                traceback.print_exc()
        
        # 결과 저장 전 이미지 객체 삭제 (메모리 확보)
        for item in top_k_candidates:
            if 'mid_frame' in item: del item['mid_frame']
        
        # all_windows에서도 이미지 객체 삭제
        for item in all_windows:
            if 'mid_frame' in item: del item['mid_frame']
        
        # 타이밍 정보 저장
        total_search_time = time.time() - search_start_time
        self.timing_info["frame_extraction_time"] = total_frame_extraction_time
        self.timing_info["clip_inference_time"] = total_clip_inference_time
        self.timing_info["blip_inference_time"] = total_blip_inference_time
        self.timing_info["total_search_time"] = total_search_time
        
        # visualizer 객체를 반환 (main에서 저장)
        return top_k_candidates, all_windows, visualizer

# ==========================================
# 4. Real-time Visualization
# ==========================================
class RealTimeVisualizer:
    def __init__(self, total_duration, k_top, save_path="results"):
        """
        실시간 시각화를 위한 클래스
        
        Args:
            total_duration: 비디오 총 길이 (초)
            k_top: Top-K 개수
            save_path: 그래프 이미지 저장 경로
        """
        self.total_duration = total_duration
        self.k_top = k_top
        self.save_path = save_path
        self.window_data = []
        self.current_top_k = []
        self.is_complete = False
        self.save_filename = None
        
        # 그래프 설정
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(figsize=(14, 6))
        self.fig.suptitle('Real-time Video Search Similarity Scores', fontsize=14, fontweight='bold')
        
    def update(self, window_info):
        """
        새로운 윈도우 정보로 그래프 업데이트
        
        Args:
            window_info: {'start': float, 'end': float, 'clip_score_norm': float, 'is_top_k': bool}
        """
        self.window_data.append(window_info)
        
        # 현재까지의 Top-K 계산
        sorted_windows = sorted(self.window_data, key=lambda x: x['clip_score_norm'], reverse=True)
        self.current_top_k = sorted_windows[:self.k_top]
        
        self._draw()
        
    def finalize(self, final_top_k):
        """
        검색 완료 후 최종 Top-K 표시
        
        Args:
            final_top_k: 최종 Top-K 윈도우 리스트
        """
        self.is_complete = True
        self.final_top_k = final_top_k
        self._draw()
        
    def _draw(self):
        """그래프 그리기"""
        self.ax.clear()
        
        if not self.window_data:
            return
        
        # 시간축과 점수 데이터 준비
        times = [(w['start'] + w['end']) / 2 for w in self.window_data]
        scores = [w['clip_score_norm'] for w in self.window_data]
        
        # 1. 기본 점수 선 그래프 (회색)
        self.ax.plot(times, scores, color='#CCCCCC', linewidth=1, alpha=0.6, zorder=1)
        
        # 2. 모든 윈도우 점 (작은 파란색)
        self.ax.scatter(times, scores, color='#4A90E2', s=30, alpha=0.5, zorder=2)
        
        # 3. 현재 Top-K 후보 (노란색 큰 점)
        if not self.is_complete:
            top_k_times = [(w['start'] + w['end']) / 2 for w in self.current_top_k]
            top_k_scores = [w['clip_score_norm'] for w in self.current_top_k]
            self.ax.scatter(top_k_times, top_k_scores, color='#FFD700', s=200, 
                          edgecolors='#FFA500', linewidths=2, zorder=4, 
                          label=f'Current Top-{self.k_top}', marker='o', alpha=0.9)
            
            # 반짝이는 효과를 위한 외곽선
            for t, s in zip(top_k_times, top_k_scores):
                circle = Circle((t, s), radius=0.3, color='#FFD700', alpha=0.3, zorder=3)
                self.ax.add_patch(circle)
        
        # 4. 최종 Top-K (빨간색 큰 점)
        if self.is_complete:
            final_times = [(w['start'] + w['end']) / 2 for w in self.final_top_k]
            final_scores = [w['clip_score_norm'] for w in self.final_top_k]
            self.ax.scatter(final_times, final_scores, color='#E74C3C', s=250, 
                          edgecolors='#C0392B', linewidths=3, zorder=5, 
                          label=f'Final Top-{self.k_top}', marker='*', alpha=1.0)
            
            # 순위 표시
            for idx, (t, s, w) in enumerate(zip(final_times, final_scores, self.final_top_k), 1):
                self.ax.annotate(f'#{idx}', xy=(t, s), xytext=(5, 5), 
                               textcoords='offset points', fontsize=10, 
                               fontweight='bold', color='#E74C3C',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#E74C3C', alpha=0.8))
        
        # 그래프 설정
        self.ax.set_xlabel('Video Time (seconds)', fontsize=11, fontweight='bold')
        self.ax.set_ylabel('Normalized Similarity Score', fontsize=11, fontweight='bold')
        self.ax.set_xlim(0, self.total_duration)
        self.ax.set_ylim(0, 105)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.legend(loc='upper right', fontsize=9)
        
        # 진행률 표시
        if self.window_data:
            progress = (self.window_data[-1]['end'] / self.total_duration) * 100
            status = "COMPLETE ✓" if self.is_complete else f"Processing... {progress:.1f}%"
            self.ax.text(0.02, 0.98, status, transform=self.ax.transAxes, 
                       fontsize=11, fontweight='bold', verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.pause(0.01)
        
    def save_and_close(self, filename_base):
        """그래프를 이미지로 저장하고 창 닫기"""
        plt.ioff()
        
        # 파일명 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_filename = f"viz_{filename_base}.png"
        save_path = os.path.join(self.save_path, self.save_filename)
        
        # 이미지로 저장
        self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [시각화] 그래프 저장 완료: {self.save_filename}")
        
        # 창 닫기
        plt.close(self.fig)
        
        return self.save_filename

# ==========================================
# 5. Main Execution
# ==========================================
def main():
    # 전체 실행 시간 측정 시작
    program_start_time = time.time()
    
    # --- Configurations ---
    VIDEO_PATH = "sample_video.mp4" # 준비된 비디오 파일 경로
    SAVE_PATH = "results"
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    else:
        print(f"Save path '{SAVE_PATH}' already exists. Results will be saved here.")
    QUERY = "바닥에 떨어진 신용카드"
    # "바닥에 떨어지는 카드를 보고 난감한 표정을 짓는 남자" # 테스트 쿼리
    
    # Experiments Parameters
    p_list = [2.0, 4.0]      # 윈도우 크기 (초)
    q_list = [12, 24, 48]         # 샘플링 프레임 수
    k_list = [3, 5]          # Top-K 개수
    USE_BLIP = False         # BLIP-2 사용 여부 (메모리 주의)
    WEIGHT_CLIP = 0.7
    WEIGHT_SEMANTIC = 0.3
    USE_LOOP = False         # 반복 실행 여부

    # Initialize
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file '{VIDEO_PATH}' not found. Please place a dummy video.")
        return

    # 초기화 시간 측정
    init_start_time = time.time()
    model_manager = ModelManager(use_blip=USE_BLIP)
    video_processor = VideoProcessor(VIDEO_PATH)
    engine = AdaptiveSearchEngine(model_manager, video_processor)
    total_init_time = time.time() - init_start_time
    
    print(f"\n[쿼리 분석] '{QUERY}'")
    sub_queries, split_reason = engine.split_query(QUERY)
    print(f"[분할된 쿼리] {sub_queries}\n")
    
    # Experiment Loop
    # 반복 실행 할 때
    if USE_LOOP:
        for p in p_list:
            for q in q_list:
                for k in k_list:
                    print(f"\n--- Running Experiment: p={p}, q={q}, k={k} ---")
                    
                    # Perform Search (실시간 시각화 활성화)
                    results, all_windows_data, visualizer = engine.search(QUERY, sub_queries, p, q, k, WEIGHT_CLIP, WEIGHT_SEMANTIC, enable_visualization=True, save_path=SAVE_PATH)
                    
                    # 전체 실행 시간 계산
                    total_elapsed_time = time.time() - program_start_time
                    
                    # Construct Filename
                    model_name = "CB" if USE_BLIP else "Clip"
                    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    # 유의미한 결과 나왔으면 _test.json 대신 .json 확장자 사용
                    filename = f"{model_name}_{p}, {q}, {k}, {USE_BLIP}, {WEIGHT_CLIP if USE_BLIP else ''}, {WEIGHT_SEMANTIC if USE_BLIP else ''}_{timestamp_str}_test.json"
                    filename_base = f"{model_name}_{p}, {q}, {k}, {USE_BLIP}, {WEIGHT_CLIP if USE_BLIP else ''}, {WEIGHT_SEMANTIC if USE_BLIP else ''}_{timestamp_str}_test"
                    
                    # 시각화 저장
                    if visualizer:
                        try:
                            viz_filename = visualizer.save_and_close(filename_base)
                        except Exception as e:
                            print(f"  [시각화] 저장 실패: {e}")
                            viz_filename = None
                    else:
                        viz_filename = None
                    
                    # 타이밍 정보 수집
                    timing_data = {
                        "total_time": round(total_elapsed_time, 2),
                        "init_time": round(total_init_time, 2),
                        "model_manager_init_time": round(model_manager.init_time, 2),
                        "clip_load_time": round(model_manager.clip_load_time, 2),
                        "blip_load_time": round(model_manager.blip_load_time, 2),
                        "video_processor_init_time": round(video_processor.init_time, 2),
                        "api_call_time": round(engine.timing_info["api_call_time"], 2),
                        "frame_extraction_time": round(engine.timing_info["frame_extraction_time"], 2),
                        "clip_inference_time": round(engine.timing_info["clip_inference_time"], 2),
                        "blip_inference_time": round(engine.timing_info["blip_inference_time"], 2),
                        "total_search_time": round(engine.timing_info["total_search_time"], 2)
                    }
                    
                    # Output Data Structure
                    output_data = {
                        "meta": {
                            "video_path": VIDEO_PATH,
                            "query": QUERY,
                            "sub_queries": sub_queries,
                            "split_reason": split_reason,
                            "parameters": {"p": p, "q": q, "k": k, "USE_BLIP": USE_BLIP, "WEIGHT_CLIP": WEIGHT_CLIP, "WEIGHT_SEMANTIC": WEIGHT_SEMANTIC},
                            "model": model_name,
                            "timestamp": timestamp_str
                        },
                        "time_used": timing_data,
                        "results": results
                    }
                    
                    # Save to JSON
                    with open(os.path.join(SAVE_PATH, filename), "w", encoding='utf-8') as f:
                        json.dump(output_data, f, indent=4, ensure_ascii=False)
                    
                    # 모든 윈도우의 상세 점수 저장
                    whole_score_filename = f"whole_score_{filename}"
                    whole_score_data = {
                        "meta": {
                            "video_path": VIDEO_PATH,
                            "query": QUERY,
                            "sub_queries": sub_queries,
                            "parameters": {"p": p, "q": q, "k": k, "USE_BLIP": USE_BLIP, "WEIGHT_CLIP": WEIGHT_CLIP, "WEIGHT_SEMANTIC": WEIGHT_SEMANTIC},
                            "total_windows": len(all_windows_data),
                            "timestamp": timestamp_str
                        },
                        "all_windows": all_windows_data
                    }
                    with open(os.path.join(SAVE_PATH, whole_score_filename), "w", encoding='utf-8') as f:
                        json.dump(whole_score_data, f, indent=4, ensure_ascii=False)
                    
                    print(f"\n[저장 완료] {filename}")
                    print(f"[상세 점수 저장 완료] {whole_score_filename}")
                    print(f"  -> 총 {len(all_windows_data)}개 윈도우의 상세 점수 저장됨")
                    if viz_filename:
                        print(f"[시각화 저장 완료] {viz_filename}")
                    print(f"[총 실행 시간] {total_elapsed_time:.2f}초\n")
                    
    # 반복 실행 아닐 때
    else:
        results, all_windows_data, visualizer = engine.search(QUERY, sub_queries, p_list[0], q_list[0], k_list[0], WEIGHT_CLIP, WEIGHT_SEMANTIC, enable_visualization=True, save_path=SAVE_PATH)
        
        # 전체 실행 시간 계산
        total_elapsed_time = time.time() - program_start_time
        
        model_name = "CB" if USE_BLIP else "Clip"
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{p_list[0]}, {q_list[0]}, {k_list[0]}, {USE_BLIP}, {WEIGHT_CLIP if USE_BLIP else ''}, {WEIGHT_SEMANTIC if USE_BLIP else ''}_{timestamp_str}_test.json"
        filename_base = f"{model_name}_{p_list[0]}, {q_list[0]}, {k_list[0]}, {USE_BLIP}, {WEIGHT_CLIP if USE_BLIP else ''}, {WEIGHT_SEMANTIC if USE_BLIP else ''}_{timestamp_str}_test"
        
        # 시각화 저장
        if visualizer:
            try:
                viz_filename = visualizer.save_and_close(filename_base)
            except Exception as e:
                print(f"  [시각화] 저장 실패: {e}")
                viz_filename = None
        else:
            viz_filename = None
        
        # 타이밍 정보 수집
        timing_data = {
            "total_time": round(total_elapsed_time, 2),
            "init_time": round(total_init_time, 2),
            "model_manager_init_time": round(model_manager.init_time, 2),
            "clip_load_time": round(model_manager.clip_load_time, 2),
            "blip_load_time": round(model_manager.blip_load_time, 2),
            "video_processor_init_time": round(video_processor.init_time, 2),
            "api_call_time": round(engine.timing_info["api_call_time"], 2),
            "frame_extraction_time": round(engine.timing_info["frame_extraction_time"], 2),
            "clip_inference_time": round(engine.timing_info["clip_inference_time"], 2),
            "blip_inference_time": round(engine.timing_info["blip_inference_time"], 2),
            "total_search_time": round(engine.timing_info["total_search_time"], 2)
        }
        
        # Output Data Structure
        output_data = {
            "meta": {
                "video_path": VIDEO_PATH,
                "query": QUERY,
                "sub_queries": sub_queries,
                "split_reason": split_reason,
                "parameters": {"p": p_list[0], "q": q_list[0], "k": k_list[0], "USE_BLIP": USE_BLIP, "WEIGHT_CLIP": WEIGHT_CLIP, "WEIGHT_SEMANTIC": WEIGHT_SEMANTIC},
                "model": model_name,
                "timestamp": timestamp_str
            },
            "time_used": timing_data,
            "results": results
        }
        
        # Save to JSON
        with open(os.path.join(SAVE_PATH, filename), "w", encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        # 모든 윈도우의 상세 점수 저장
        whole_score_filename = f"whole_score_{filename}"
        whole_score_data = {
            "meta": {
                "video_path": VIDEO_PATH,
                "query": QUERY,
                "sub_queries": sub_queries,
                "parameters": {"p": p_list[0], "q": q_list[0], "k": k_list[0], "USE_BLIP": USE_BLIP, "WEIGHT_CLIP": WEIGHT_CLIP, "WEIGHT_SEMANTIC": WEIGHT_SEMANTIC},
                "total_windows": len(all_windows_data),
                "timestamp": timestamp_str
            },
            "all_windows": all_windows_data
        }
        with open(os.path.join(SAVE_PATH, whole_score_filename), "w", encoding='utf-8') as f:
            json.dump(whole_score_data, f, indent=4, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"[검색 완료] 결과가 {filename}에 저장되었습니다.")
        print(f"[상세 점수 저장 완료] {whole_score_filename}")
        print(f"  -> 총 {len(all_windows_data)}개 윈도우의 상세 점수 저장됨")
        if viz_filename:
            print(f"[시각화 저장 완료] {viz_filename}")
        print(f"{'='*60}")
        print(f"\n📊 [전체 실행 시간 분석]")
        print(f"{'='*60}")
        print(f"  ⏱️  총 실행 시간: {total_elapsed_time:.2f}초")
        print(f"\n  🔧 초기화 단계:")
        print(f"     - ModelManager 초기화: {model_manager.init_time:.2f}초")
        print(f"       ├─ CLIP 로드: {model_manager.clip_load_time:.2f}초")
        print(f"       └─ BLIP-2 로드: {model_manager.blip_load_time:.2f}초")
        print(f"     - VideoProcessor 초기화: {video_processor.init_time:.2f}초")
        print(f"     - 전체 초기화: {total_init_time:.2f}초")
        print(f"\n  🔍 검색 단계:")
        print(f"     - API 호출 (쿼리 분석): {engine.timing_info['api_call_time']:.2f}초")
        print(f"     - 프레임 추출: {engine.timing_info['frame_extraction_time']:.2f}초")
        print(f"     - CLIP 추론: {engine.timing_info['clip_inference_time']:.2f}초")
        if USE_BLIP:
            print(f"     - BLIP-2 추론: {engine.timing_info['blip_inference_time']:.2f}초")
        print(f"     - 전체 검색: {engine.timing_info['total_search_time']:.2f}초")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()