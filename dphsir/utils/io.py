import matplotlib.animation as animation
import matplotlib.pyplot as plt


def loadmat(path):
    from scipy.io import loadmat
    return loadmat(path)


def savemat(path, obj):
    from scipy.io import savemat
    savemat(path, obj)


def save_img(path, img):
    from imageio import imsave
    imsave(path, img)


def save_ani(img_list, filename='a.gif', fps=60):
    def animation_generate(img):
        ims_i = []
        im = plt.imshow(img, cmap='gray')
        ims_i.append([im])
        return ims_i

    ims = []
    fig = plt.figure()
    for img in img_list:
        ims += animation_generate(img)
    ani = animation.ArtistAnimation(fig, ims)
    ani.save(filename, fps=fps, writer='ffmpeg')


def show_gray(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def show_hsi(hsi, band=20):
    img = hsi[:, :, band]
    plt.imshow(img, cmap='gray')
    plt.show()
