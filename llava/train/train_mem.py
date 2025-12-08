from llava.train.train import train
import wandb

if __name__ == "__main__":
    
    wandb.login(key="9c446d73cfb7d7ef1b73bf79e4457cb5446c1ff4") 
    wandb.init(
    project="LLaVA1.5-7B-finetune-full",  # 여기에 프로젝트 이름 입력
    name="experiment_01"
)
    train(attn_implementation="flash_attention_2")