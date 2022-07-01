from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    
class Summarizer:
    def __init__(self, pretrain_path="VietAI/vit5-large-vietnews-summarization"):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrain_path)
        
    def summary(self, text):
        summary_text = ""
        encoding = self.tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
        outputs = self.model.generate(
			input_ids=input_ids, attention_mask=attention_masks,
			max_length=512,
			early_stopping=True
		)
        for output in outputs:
            line = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            summary_text = summary_text + " " +line
        return summary_text
    
