"""
使用GPRO训练Qwen2.5-0.5B-Instruct模型，使得其具有推理能力
"""

from transformers import AutoModelForCausalLM, AutoTokenizer




if __name__=="__main__":
    model_name = "Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "Joy can read 8 pages of a book in 20 minutes. How many hours will is take her to read 120 pages?"
    messages = [{"role":"user", "content":prompt}]

    text =  tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    model_inputs = tokenizer([text],return_tensors=".pt").to(model.device)

