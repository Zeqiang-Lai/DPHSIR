
class ProgressBar:
    def __init__(self, total):
        from tqdm import tqdm
        self.pbar = tqdm(total=total, dynamic_ncols=True)
    
    def __call__(self, **context):
        self.pbar.update()

class GatherIntermediates:
    def __init__(self, filter):
        self.intermediates = []
        self.filter = filter
    
    def __call__(self, **context):
        saved = self.filter(context)
        self.intermediates.append(saved)