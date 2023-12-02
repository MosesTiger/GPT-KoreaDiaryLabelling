import torch
import torch.nn as nn  
from transformers import AutoTokenizer, AutoModel 
import json
import sys
sys.path.append('/content/drive/My Drive/data')
from model import BaseRegressor, SmallRegressor, LargeRegressor

class WeekChallenge(nn.Module):
    
    def __init__(self, model_args):
        super(WeekChallenge, self).__init__()

        self.args = model_args

        self.tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base", model_max_length=512)
        self.embed = AutoModel.from_pretrained("monologg/kobigbird-bert-base")        
        
        if self.args["model_size"] == "base":
            self.regressor = BaseRegressor()
        elif self.args["model_size"] == "small":
            self.regressor = SmallRegressor()
        elif self.args["model_size"] == "large":
            self.regressor = LargeRegressor()

        if self.args["use_cuda"]:
            self.regressor = self.regressor.cuda()
            self.embed = self.embed.cuda()

    # 사용자 입력 처리
    def process_input(self, user_input):
        tokenized = self.tokenizer(user_input, return_tensors="pt", padding="max_length", truncation=True)
        return tokenized

    # 예측 함수
    def predict_scores(self, user_input):
        self.eval()
        with torch.no_grad():
            inputs = self.process_input(user_input)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.embed(**inputs)
            # 마지막 은닉 상태를 regressor에 전달
            last_hidden_state = outputs.last_hidden_state[:, 0]
            scores = self.regressor(last_hidden_state)
            return scores

    # 평가 함수
    def evaluate_stars(self, pooler_output):
        avg_pool = pooler_output.mean(dim=1)
        # 0과 1 사이의 값으로 정규화
        normalized_score = torch.sigmoid(avg_pool)
        # 1에서 5 사이의 점수로 변환
        stars = 1 + 4 * normalized_score
        return stars

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = '/content/drive/My Drive/ckpt/4_Adam_0.01_base.pth'
model_args = {"model_size": "base", "use_cuda": torch.cuda.is_available()}
mynet = WeekChallenge(model_args).to(device)
mynet.load_state_dict(torch.load(model_path, map_location=device))

# 사용자로부터 일기 내용 입력받기
user_diary_input = input("Enter your weekly diary: ")

# 점수 예측
predicted_scores = mynet.predict_scores(user_diary_input)

# 결과 출력
print(f"Predicted Scores: {predicted_scores}")