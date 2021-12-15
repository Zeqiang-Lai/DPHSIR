
class Augment:
    def __init__(self, base_denoiser):
        self.base_denoiser = base_denoiser

    def __call__(self, x, sigma, iter):
        x = self.augment(x, iter % 8)

        x = self.base_denoiser.denoise(x, sigma)

        if iter % 8 == 3 or iter % 8 == 5:
            x = self.augment(x, 8 - iter % 8)
        else:
            x = self.augment(x, iter % 8)
            
        return x
        
    @staticmethod
    def augment(img, mode=0):
        if mode == 0:
            return img
        elif mode == 1:
            return img.rot90(1, [2, 3]).flip([2])
        elif mode == 2:
            return img.flip([2])
        elif mode == 3:
            return img.rot90(3, [2, 3])
        elif mode == 4:
            return img.rot90(2, [2, 3]).flip([2])
        elif mode == 5:
            return img.rot90(1, [2, 3])
        elif mode == 6:
            return img.rot90(2, [2, 3])
        elif mode == 7:
            return img.rot90(3, [2, 3]).flip([2])
