import asyncio


class Bucket:
    def __init__(self, max_tokens: int, queue: asyncio.queue, early_batches: int = 2):
        self.max_tokens = max_tokens
        self.items = []
        self.current_tokens = 0
        self.batch_number = 0
        self.early_batches = early_batches

    def can_overflow(self, tokens: int) -> bool:
        return self.current_tokens + tokens >= self.max_tokens

    def should_flush_early(self) -> bool:
        if self.batch_number < self.early_batches:
            return len(self.items) >= 2
        return False

    async def add_to_bucket(self, item, tokens: int):
        if self.should_flush(tokens):
            self.flush()
        self.items.append(item)
        self.current_tokens += tokens

    def should_flush(self, incoming_tokens: int) -> bool:
        if not self.can_overflow(incoming_tokens):
            return True

        if self.should_flush_early():
            return True

        return False

    async def flush(self):
        await self.queue.put([self.items])
        self.items = []
        self.current_tokens = 0
        self.batch_number += 1

    def is_empty(self):
        return len(self.items) == 0
