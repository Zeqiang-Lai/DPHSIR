
class ProgressBar:
    def __init__(self, total):
        from tqdm import tqdm
        self.pbar = tqdm(total=total, dynamic_ncols=True)
    
    def __call__(self, **context):
        self.pbar.update()
    