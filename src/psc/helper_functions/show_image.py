import matplotlib.pyplot as plt

def show_image(array):
    plt.figure(figsize=(2, 2))
    plt.imshow(array, cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()