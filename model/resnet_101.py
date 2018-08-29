from model.common import resnet


def build_model(input, config, is_training):
    end_points = {}
    net = resnet.resnet_101(input, config.num_class, is_training)
    return net, end_points
