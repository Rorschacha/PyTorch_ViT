import os
import random
import shutil
import json

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_path2(root_path, show=True):
    """
    返回路径下的文件名、文件路径等,迭代器版
    :param root_path:
    :param show:
    :return: a dict,key为filenames,filepaths,dirpath,dirnames
    """
    dirpath, dirnames, filenames = next(os.walk(root_path))  # 迭代器
    filepaths = []
    for y in filenames:
        filepaths.append(os.path.join(dirpath, y))
    if show:
        print('  dirpath :', dirpath)
        print(' dirnames :', dirnames)
        print('filenames :', filenames)
        print('filepaths :', filepaths)
    dict_result = dict()
    dict_result['dirpath'] = dirpath
    dict_result['dirnames'] = dirnames
    dict_result['filenames'] = filenames
    dict_result['filepaths'] = filepaths

    return dict_result


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


def create_dir(path):
    """create folder"""
    # format
    if True:
        path = path.strip()
        path = path.rstrip("\\")

    isExists = os.path.exists(path)
    if not isExists:
        os.mkdir(path)
        print('-Dir created-', path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print('-Dir already existed-')
        return False


def build_setfolder(folder_dir):
    """
    建立划分数据集文件夹
    :return:
    """
    dataset_dir = folder_dir
    split_dir = os.path.abspath(os.path.join(dataset_dir, "splitted_dataset"))
    # split_dir =os.path.join(dataset_dir, "splitted_dataset")
    train_dir = os.path.join(split_dir, "train")
    valid_dir = os.path.join(split_dir, "valid")
    test_dir = os.path.join(split_dir, "test")

    isExists = os.path.exists(split_dir)
    if not isExists:
        create_dir(split_dir)
        create_dir(train_dir)
        create_dir(valid_dir)
        create_dir(test_dir)
        print('-Dir created-', split_dir)

        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print('-Dir already existed-', split_dir)

        return False


def inspect_img(**kwargs):
    """path=...,image=...,plt版,接受bgr图片/数组 或者文件路径"""
    # import matplotlib.pyplot as plt
    # import matplotlib.image as mpimg

    path = kwargs.get('path')
    image = kwargs.get('image')
    show = kwargs.get('show')
    if path is not None:
        path = kwargs['path']
        image = mpimg.imread(path)
    elif image is not None:
        image = kwargs['image']

    if image is not None:
        print('  shape :', image.shape)
        print('   size :', image.size)
        if show:
            plt.imshow(image)
            plt.show()

    return image


def split_small_dataset():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    dataset_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "RMB_data"))
    split_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "rmb_split"))
    train_dir = os.path.join(split_dir, "train")
    valid_dir = os.path.join(split_dir, "valid")
    test_dir = os.path.join(split_dir, "test")

    if not os.path.exists(dataset_dir):
        raise Exception("\n{} 不存在，请下载 02-01-数据-RMB_data.rar 放到\n{} 下，并解压即可".format(
            dataset_dir, os.path.dirname(dataset_dir)))

    train_pct = 0.8
    valid_pct = 0.1
    test_pct = 0.1

    for root, dirs, files in os.walk(dataset_dir):
        for sub_dir in dirs:

            imgs = os.listdir(os.path.join(root, sub_dir))
            imgs = list(filter(lambda x: x.endswith('.jpg'), imgs))
            random.shuffle(imgs)
            img_count = len(imgs)

            train_point = int(img_count * train_pct)
            valid_point = int(img_count * (train_pct + valid_pct))

            if img_count == 0:
                print("{}目录下，无图片，请检查".format(os.path.join(root, sub_dir)))
                import sys
                sys.exit(0)
            for i in range(img_count):
                if i < train_point:
                    out_dir = os.path.join(train_dir, sub_dir)
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sub_dir)
                else:
                    out_dir = os.path.join(test_dir, sub_dir)

                makedir(out_dir)

                target_path = os.path.join(out_dir, imgs[i])
                src_path = os.path.join(dataset_dir, sub_dir, imgs[i])

                shutil.copy(src_path, target_path)

            print('Class:{}, train:{}, valid:{}, test:{}'.format(sub_dir, train_point, valid_point - train_point,
                                                                 img_count - valid_point))
            print("已在 {} 创建划分好的数据\n".format(out_dir))

    pass


class Create_Annotation():
    def __int__(self):
        pass

    def build_apc(self, data_folder=r"", anno_folder=r"",anno_type="train"):
        """
        build annotation for picture classification
        label in path
        :return:
        """
        pathd = get_path2(data_folder, show=False)
        filenames = pathd['filenames']  # 'cat.0.jpg'
        filepaths = pathd['filepaths']

        # extract label
        labels = []
        for index, filename in enumerate(filenames):
            sr = filename.split(".")
            type_p = sr[0]

            if type_p == "cat":
                label_txt = filepaths[index] + "    " + "0"
                labels.append(label_txt)

            elif type_p == "dog":
                label_txt = filepaths[index] + "    " + "1"
                labels.append(label_txt)
            else:
                raise Exception("unexpected data type")


        else:
            print("length of labels:", len(labels))  # summary

        if True:  # make annotation txt
            if os.path.exists(anno_folder):
                anno_name="annotation_"+anno_type+".txt"
                path_ls = os.path.join(anno_folder, anno_name)

                if True: #to txt
                    with open(path_ls, 'w') as fobj:
                        for line in labels:
                            fobj.write(line + "\n")
                        print('annotation created', path_ls)


                if False: # to json
                    with open(path_ls, 'w') as f_obj:
                        json.dump(labels, f_obj)
                        print('annotation created', path_ls)


            else:
                raise Exception("anno_folder path is not existed")



        pass


def build_a():
    ca = Create_Annotation()
    path_ds = r"D:\datasets\PracticeSets\cats and dogs\data"
    path_tr = r"D:\datasets\PracticeSets\cats and dogs\data\train"
    path_te = r"D:\datasets\PracticeSets\cats and dogs\data\test"

    path_at = r"D:\datasets\PracticeSets\cats and dogs\data"
    ca.build_apc(data_folder=path_tr, anno_folder=path_at,anno_type="train")
    #ca.build_apc(data_folder=path_te, anno_folder=path_at,anno_type="test")


def main():
    if False:
        raw_dataset_path = r''
        modified_dataset_path = r'test for DLT'
        os.mkdir(modified_dataset_path)
        build_setfolder(modified_dataset_path)
        # split_small_dataset()

    if True:
        build_a()

        pass


if __name__ == '__main__':
    main()
