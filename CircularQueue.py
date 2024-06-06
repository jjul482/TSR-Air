class CircularQueue():

    def __init__(self):
        self.k = 0
        self.queue = []
        self.head = -1

    # Insert an element into the circular queue
    def add(self, data):
        self.queue.append(data)

    def read(self):
        if len(self.queue) == 0:
            return None
        else:
            self.head += 1
            if self.head == len(self.queue):
                self.head = 0
            return self.queue[self.head]
