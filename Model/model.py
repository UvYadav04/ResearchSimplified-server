class Model:
    def __init__(self, model_path, bucket):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.bucket = bucket
        pass

    def generate(self, input):
        return "generated"

    def get_model(self):
        if self.model is None:
            self.model = self.load_model(self.model_path)
        return self.model

    def get_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = self.load_tokenizer(self.tokenizer_path)
        return self.tokenizer

    def load_model(self, model_path):
        return "Model"

    def load_tokenizer(self, tokenizer_path):
        return "Model"

    def format_instruction(self, message):
        return [
            {
                "role": "system",
                "content": "You are a Research paper content simplifier. For each input you get, you need to simplify it in the following format, and if you think the input is already simple, return as it, else simplify the query.",
            },
            {"role": "user", "content": message},
        ]

    def apply_chat_template(self, message):
        formatted = self.format_instruction(message)
        return self.tokenizer.apply_chat_template(
            formatted, tokenize=True, return_tensors="pt", add_generation_prompt=True
        )
