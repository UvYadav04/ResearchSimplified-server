from collections import deque


class Queue:
    def __init__(self):
        self.queue = deque()

    def reset_queue(self):
        self.queue.clear()

    def add_to_queue(self, item):
        self.queue.append(item)

    def get_top_element(self):
        if self.is_empty():
            return None
        return self.queue.popleft()

    def is_empty(self):
        return len(self.queue) == 0
