import numpy as np
import glob as gb
from PIL import Image

def read_image(path1 = './test_images/', path2 = './test_annotation.annotation', data_size = 1050):
    data = list()
    with open(path2, 'r') as fileReader:
        # read all the data
        lines = fileReader.readlines()
        for line in lines:
            line = line.strip()
            # split the index and label
            line = line.split("\t")  
            data.append(line[:])
    data = np.array(data)
    data = data.astype(int)
    index = data[:, 0]
    label = data[:, 1:]
    label = label.reshape(-1)
    print("data loaded!")
    
    path3 = '.JPEG'
    num = 0
    row_data = []
    for ii in range(1, data_size + 1):
        num_str = str(ii)
        path = path1+num_str+path3
        imm = Image.open(path)
        l1 = []
        for n in range(64):
            l2 = []
            for m in range(64):
                l3 = imm.getpixel((m, n))
                if type(l3) is int:
                    l2.append(list([l3, l3, l3]))
                else:
                    l2.append(list(l3))
            l1.append(l2)
        one_image = [np.array(l1), label[num]]
        row_data.append(one_image)
        num = num + 1
    whole_x = np.zeros((data_size, 64, 64, 3))
    whole_y = np.zeros((data_size, 20), dtype=int)
    for i in range(data_size):
        whole_x[i] = row_data[i][0]
        label = row_data[i][1]
        if label!=21:
            whole_y[i][label - 1] = 1
    whole_data = [whole_x, whole_y]
    print("image loaded!")
    return whole_data


