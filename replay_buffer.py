class ReplayBuffer:
    def __init__(self, tasks, max_task_size, num_classes):
        self.tasks = tasks
        self.task_examples = {}
        self.task_labels = {}
        self.max_task_size = max_task_size
        self.num_classes = num_classes
        self.num_replay = {}

    def add_example(self, example):
        label = example[1]
        task_index = self.tasks.task_map[label]
        if task_index not in self.task_examples:
            self.task_examples[task_index] = [example]
            self.task_labels[task_index] = [label]
            self.num_replay[task_index] = 0
        else:
            self.task_examples[task_index].append(example)
            self.task_labels[task_index].append(label)
            if len(self.task_examples[task_index]) > self.max_task_size:
                self.task_examples[task_index].pop(0)
                self.task_labels[task_index].pop(0)

    def add_examples(self, examples):
        label = examples[0][1]
        task_index = self.tasks.task_map[label]
        if task_index not in self.task_examples:
            self.task_examples[task_index] = examples[:min(len(examples), self.max_task_size)]
            self.task_labels[task_index] = [label] * min(len(examples), self.max_task_size)
            self.num_replay[task_index] = 0

    def get_task_examples(self, task):
        if task in self.task_examples.keys():
            return self.task_examples[task]
        else:
            return None

    def task_list(self):
        return self.tasks.tasks

