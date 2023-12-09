import torch
import torch.nn as nn  
from transformers import AutoTokenizer, AutoModel 
import json
import sys
sys.path.append('/content/drive/My Drive/data')
from utils.model import WeekChallenge


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = '/home1/ohs/ongoing_proj/GPT-KoreaDiaryLabelling/ckpt/4_Adam_0.001_small.pth'
model_args = {"model_size": "small", "use_cuda": torch.cuda.is_available()}

mynet = WeekChallenge(model_args).to(device)
mynet.load_state_dict(torch.load(model_path, map_location=device))

# 사용자로부터 일기 내용 입력받기
user_diary_input = """
1000만원 짜리 예금 들었다.
은행가서 적금도 들었다.
그래도 오늘 운동을 못갔다. 아파서 운동을 못했다. 운동을 하고 나서 더 아픈 것 같아서 쉬었다.
그리고 나서 책을 읽었다.
"""
# 점수 예측
predicted_scores = mynet(user_diary_input)

# 결과 출력
print(f"Predicted Scores: {predicted_scores}")
