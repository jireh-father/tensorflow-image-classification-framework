from factory import model_factory
from core.dataset import Dataset


def train(config):
    # get model
    model = model_factory.build_model(config)
    if model is None:
        print("There is no model name.(%s)" % config.model_name)
        return False
    inputs, labels, logits, end_points, ops = model
    print(inputs, labels, logits, end_points, ops)

    # handle dataset & preprocessing
    train_dataset = Dataset(sess, config.use_shuffle, True, config)
    test_dataset = Dataset(sess, False, False, config)

    # logger setting

    # train

    # eval

    # visualization

    pass
