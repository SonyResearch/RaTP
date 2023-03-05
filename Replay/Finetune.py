class Finetune:
    def __init__(self, args):
        self.comment = 'Do nothing'

    def update_dataloader(self, dataloader=None):
        return None

    def update(self, model, task_id, dataloader):
        pass