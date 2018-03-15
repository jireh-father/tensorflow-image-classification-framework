from core import util
import os


def init(config):
    label_path = os.path.join(config.dataset_dir, "labels.txt")
    config.num_classes = util.count_label(label_path)
    if not config.num_classes:
        raise Exception("Check the label file : %s" % label_path)
    train_dataset_files = util.get_tfrecord_filenames(config.dataset_name, config.dataset_dir)
    if not train_dataset_files:
        raise Exception("There is no tfrecord files")
    config.num_samples_per_epoch = util.count_records(train_dataset_files)
