from .llm import LLM


class Model:
    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device

    def stream_document(self, inputs, lastContent):
        # this generation method is for hf inference providers
        try:
            if not "content" in inputs:
                return None
            content = inputs["content"]
            # stream = self.model.chat.completions.create(
            #     messages=self.format_instruction(content, lastContent),
            #     model="meta-llama/Meta-Llama-3-8B-Instruct",
            #     stream=True,
            #     max_tokens=512,
            # )
            # return stream
            model = LLM()
            return model.stream(self.format_instruction(content, lastContent))

        except Exception as e:
            print(e)
            return {"error": str(e)}

    def stream_query(self, input, relatedContent, relatedChats, contextChunk):

        # contents = self.format_chunk_query(input, relatedContent)

        # response = self.model.models.generate_content_stream(
        #     model="gemini-2.5-flash",
        #     contents = contents,
        # )
        messages = self.format_query(input, relatedContent, relatedChats, contextChunk)
        stream = self.model.chat.completions.create(
            messages=messages,
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            stream=True,
            max_tokens=512,
        )
        return stream
        # return response
