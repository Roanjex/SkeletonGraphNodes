from skimage.morphology import skeletonize
from skimage import color, img_as_bool
import sknw
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2 as cv
import json


def ndarray_to_json(ps):  # convert points to .json
    ps_correct = np.copy(ps)
    ps_correct[:, 0] = ps_correct[:, 1]
    ps_correct[:, 1] = ps[:, 0]
    temp_list = np.ndarray.tolist(ps_correct)
    with open("points.json", "w") as outfile:
        json.dump(temp_list, outfile)


def threshold(name, thresh):  # threshold
    img = cv.imread(name, 0)
    img = cv.medianBlur(img, 7)
    ret, img = cv.threshold(img, thresh, 255, cv.THRESH_BINARY)
    cv.imwrite("Cables_threshold_{}.jpg".format(thresh), img)


def graph_build():
    name = "Cables"
    thresh = 127
    threshold(name + ".jpg", thresh)
    img = img_as_bool(color.rgb2gray(io.imread("Cables_threshold_{}.jpg".format(thresh))))
    ske = skeletonize(~img).astype(np.uint16)
    # build graph from skeleton
    graph = sknw.build_sknw(ske)
    # draw image
    plt.imshow(img, cmap='gray')
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    # draw edges by pts
    for (s, e) in graph.edges():
        ps = graph[s][e]['pts']
        plt.plot(ps[:, 1], ps[:, 0], 'green')
    # draw node by o
    nodes = graph.nodes()
    ps = np.array([nodes[i]['o'] for i in nodes])
    plt.plot(ps[:, 1], ps[:, 0], 'r.')
    # title and show
    plt.title('')
    plt.axis('off')
    plt.show()
    #  plt.savefig("Skel.jpg")


def main():
    graph_build()


if __name__ == "__main__":
    main()
