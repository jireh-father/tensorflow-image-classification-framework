from factory import model_factory, dataset_factory
from core.dataset import Dataset
import tensorflow as tf
from core import util, tfrecorder_builder
# from preprocessing import preprocessing_factory
import os, glob
from datetime import datetime
import numpy as np
import cv2
from tensorflow.contrib.tensorboard.plugins import projector
from visualizer.grad_cam_plus_plus import GradCamPlusPlus
from visualizer import embedding


class Trainer:
    def __init__(self, config):
        self.config = config
        self.sess = None
        self.train_writer = None
        self.validation_writer = None
        self.saver = None
        self.inputs = None
        self.labels = None
        self.logits = None
        self.end_points = None
        self.train_dataset = None
        self.validation_dataset = None
        self.cam = None
        self.global_step = None
        self.loss_op = None
        self.train_op = None
        self.accuracy_op = None
        self.learning_rate = None
        self.pred_idx_op = None
        self.is_training = None
        self.heatmap_imgs = {}
        self.num_current_cam = 0
        self.train_projector_config = projector.ProjectorConfig()
        self.validation_projector_config = projector.ProjectorConfig()
        self.train_embed_dataset = None
        self.train_embed_labels = None
        self.train_embed_activations = None
        self.validation_embed_dataset = None
        self.validation_embed_labels = None
        self.validation_embed_activations = None
        self.num_current_train_embed = 0
        self.num_current_validation_embed = 0
        self.validation_embed_dir = os.path.join(self.config.log_dir, "embedding/validation")
        self.train_embed_dir = os.path.join(self.config.log_dir, "embedding/train")
        self.summary_op = None
        self.avg_loss_pl = None
        self.avg_loss_summary = None
        self.avg_accuracy_pl = None
        self.avg_accuracy_summary = None
        self.dataset_path = tf.placeholder(tf.string, shape=(), name="dataset_path")
        self.input_size = tf.placeholder(tf.int32, shape=(), name="input_size")
        self.model_f = None

    def run(self):
        self.make_dataset()

        self.init_config()

        self.init_dataset()

        self.init_model()

        self.summary()

        self.init_session()

        self.restore_model()

        stop = False
        for epoch in range(self.config.epoch):
            if self.config.train:
                if self.config.num_gpus > 1:
                    stop = self.train_multiple(epoch)
                else:
                    stop = self.train(epoch)

            if self.config.validation:
                if self.config.num_gpus > 1:
                    self.validate_multiple(epoch)
                else:
                    self.validate(epoch)

                if epoch % self.config.validation_embed_visualization_interval == 0 and self.validation_embed_activations is not None:
                    embedding.add_embedding(self.validation_projector_config, sess=self.sess,
                                            embedding_list=[self.validation_embed_activations],
                                            embedding_path=self.validation_embed_dir, image_size=self.config.input_size,
                                            channel=self.config.num_channel, labels=self.validation_embed_labels,
                                            prefix="epoch" + str(epoch))

                if not self.config.train:
                    break
            if self.config.train and stop:
                break
        if self.config.validation and self.config.use_validation_embed_visualization and self.validation_embed_dataset is not None:
            embedding.write_embedding(self.validation_projector_config, self.sess, self.validation_embed_dataset,
                                      embedding_path=self.validation_embed_dir,
                                      image_size=self.config.input_size, channel=self.config.num_channel,
                                      labels=self.validation_embed_labels)

        self.sess.close()
        tf.reset_default_graph()

    def make_dataset(self):
        if self.config.dataset_name in dataset_factory.dataset_list:
            dataset_factory.download_and_make_tfrecord(self.config)
        else:
            tfrecorder_builder.make_tfrecord(self.config.dataset_name, self.config.dataset_dir,
                                             self.config.train_fraction,
                                             self.config.num_channel,
                                             self.config.num_dataset_split,
                                             self.config.remove_original_images)

    def summary(self):
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        for end_point in self.end_points:
            x = self.end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                            tf.nn.zero_fraction(x)))

        summaries.add(tf.summary.scalar('losses', self.loss_op))

        # accuracy
        summaries.add(tf.summary.scalar('accuracy', self.accuracy_op))

        for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            Trainer.variable_summaries(variable, summaries)

        summaries.add(tf.summary.scalar('learning_rate', self.learning_rate))

        self.summary_op = tf.summary.merge(list(summaries), name='summary_op')

        self.avg_loss_pl = tf.placeholder(tf.float32, (), "avg_loss_pl")
        self.avg_loss_summary = tf.summary.scalar("average_loss", self.avg_loss_pl)
        self.avg_accuracy_pl = tf.placeholder(tf.float32, (), "avg_accuracy_pl")
        self.avg_accuracy_summary = tf.summary.scalar("average_accuracy", self.avg_accuracy_pl)

    def init_config(self):
        label_path = os.path.join(self.config.dataset_dir, "labels.txt")
        num_class = util.count_label(label_path)
        self.config.num_class = num_class

        if not self.config.num_class:
            raise Exception("Check the label file : %s" % label_path)

        if self.config.train:
            train_filenames = util.get_tfrecord_filenames(self.config.dataset_name, self.config.dataset_dir)
            if not train_filenames:
                raise Exception("There is no tfrecord files")
            num_train_sample = util.count_records(train_filenames)

            self.config.num_train_sample = num_train_sample

        if self.config.validation:
            validation_filenames = util.get_tfrecord_filenames(self.config.dataset_name, self.config.dataset_dir,
                                                               'validation')
            if not validation_filenames:
                raise Exception("There is no tfrecord files")

            num_validation_sample = util.count_records(validation_filenames)

            self.config.num_validation_sample = num_validation_sample
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        self.is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
        from slim.model import model_factory as mf
        self.model_f = mf.get_network_fn(self.config.model_name, self.config.num_class,
                                         weight_decay=self.config.weight_decay,
                                         is_training=self.is_training)

        if not self.config.input_size:
            self.config.input_size = self.model_f.default_image_size

    def init_model(self):
        if self.config.num_gpus > 1:
            model = model_factory.build_model_multiple(self.config, self.train_dataset, self.model_f, self.is_training)
            self.train_op, self.is_training, self.global_step, default_last_conv_name, self.loss_op = model
            return
        else:
            model = model_factory.build_model(self.config)

        if model is None:
            raise Exception("There is no model name.(%s)" % self.config.model_name)

        self.inputs, self.labels, self.logits, self.end_points, self.is_training, self.global_step, default_last_conv_name, ops = model
        if ops:
            self.loss_op = ops["loss"]
            self.train_op = ops["train"]
            self.accuracy_op = ops["accuracy"]
            self.learning_rate = ops["learning_rate"]
            self.pred_idx_op = ops["pred_idx"]

        if (self.config.use_train_cam or self.config.self.use_validation_cam) and default_last_conv_name:
            self.cam = GradCamPlusPlus(self.logits, self.end_points[default_last_conv_name],
                                       self.inputs, self.is_training)

    def init_session(self):
        self.sess.run(tf.global_variables_initializer())

        summary_dir = os.path.join(self.config.log_dir, "summary")
        self.train_writer = tf.summary.FileWriter(summary_dir + '/train', self.sess.graph)
        self.validation_writer = tf.summary.FileWriter(summary_dir + '/validation')
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())

    def init_dataset(self):
        self.train_dataset = Dataset(self.sess, self.config.batch_size, self.config.use_train_shuffle, True,
                                     self.config, self.dataset_path, self.input_size)
        self.validation_dataset = Dataset(self.sess, self.config.validation_batch_size, False, False, self.config,
                                          self.dataset_path, self.input_size)

    def train(self, epoch):
        dataset_path = os.path.join(self.config.dataset_dir,
                                    "%s_%s*tfrecord" % (self.config.dataset_name, self.config.train_name))
        self.train_dataset.init(dataset_path, self.config.input_size)
        total_accuracy = .0
        total_loss = .0
        step = 0
        total_steps = self.config.num_train_sample // self.config.batch_size
        if self.config.num_train_sample % self.config.batch_size > 0:
            total_steps += 1
        self.num_current_train_embed = 0
        self.train_embed_labels = None
        self.train_embed_activations = None
        self.train_embed_dataset = None

        while True:
            try:
                batch_xs, batch_ys = self.train_dataset.get_next_batch()
                _, accuracy, loss, global_step, logits, pred_idx = self.sess.run(
                    [self.train_op, self.accuracy_op, self.loss_op, self.global_step, self.logits,
                     self.pred_idx_op],
                    feed_dict={self.inputs: batch_xs, self.labels: batch_ys,
                               self.is_training: True})
                total_accuracy += accuracy
                total_loss += loss

                if step % self.config.summary_interval == 0:
                    now = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                    print(
                        "[%s TRAIN] [Global Steps: %4d] [Epoch: %3d] [Current Step: %4d / %4d] [Accuracy: %10f] [Cost: %10f]" % (
                            now, global_step, epoch, step + 1, total_steps, accuracy, loss))

                    if self.config.use_summary:
                        summary = self.sess.run(self.summary_op,
                                                feed_dict={self.inputs: batch_xs, self.labels: batch_ys,
                                                           self.is_training: True})
                        self.train_writer.add_summary(summary, global_step)
                step += 1

                if self.config.use_train_cam and self.cam:
                    self.create_cam(batch_xs, batch_ys, logits)

                if epoch % self.config.train_embed_visualization_interval == 0:
                    self.add_embedding(batch_xs, batch_ys, logits, pred_idx)
                if self.config.total_steps and global_step >= self.config.total_steps:
                    return True
                if self.config.steps_per_epoch and step >= self.config.steps_per_epoch:
                    break
            except tf.errors.OutOfRangeError:
                break
        if step > 0:
            avg_accuracy = float(total_accuracy) / step
            avg_loss = float(total_loss) / step
            print("%d Epoch Avg Train Accuracy : %f" % (epoch, avg_accuracy))
            self.summary_average(epoch, avg_accuracy, avg_loss)

            if epoch % self.config.save_interval == 0:
                self.saver.save(self.sess, self.config.log_dir + "/model_epoch_%d.ckpt" % epoch, global_step)

            self.summary_cam(epoch)

            if epoch % self.config.train_embed_visualization_interval == 0 and self.train_embed_activations is not None:
                train_embed_dir = os.path.join(self.train_embed_dir, "epoch_" + str(epoch))
                embedding.add_embedding(self.train_projector_config, sess=self.sess,
                                        embedding_list=[self.train_embed_activations],
                                        embedding_path=train_embed_dir, image_size=self.config.input_size,
                                        channel=self.config.num_channel, labels=self.train_embed_labels,
                                        prefix="epoch" + str(epoch))
                embedding.write_embedding(self.train_projector_config, self.sess, self.train_embed_dataset,
                                          embedding_path=train_embed_dir,
                                          image_size=self.config.input_size, channel=self.config.num_channel,
                                          labels=self.train_embed_labels)
                self.train_projector_config = projector.ProjectorConfig()
        return False

    def train_multiple(self, epoch):
        self.train_dataset.init()
        # total_accuracy = .0
        total_loss = .0
        step = 0
        total_steps = self.config.num_train_sample // self.config.batch_size
        if self.config.num_train_sample % self.config.batch_size > 0:
            total_steps += 1
        self.num_current_train_embed = 0
        self.train_embed_labels = None
        self.train_embed_activations = None
        self.train_embed_dataset = None
        while True:
            try:
                _, loss, global_step = self.sess.run([self.train_op, self.loss_op, self.global_step],
                                                     feed_dict={self.is_training: True})
                total_loss += loss

                if step % self.config.summary_interval == 0:
                    now = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                    print(
                        "[%s TRAIN] [Epoch: %3d] [Current Step: %4d / %4d] [Cost: %10f]" % (
                            now, epoch, step + 1, total_steps, loss))

                    if self.config.use_summary:
                        summary = self.sess.run(self.summary_op,
                                                feed_dict={self.is_training: True})
                        self.train_writer.add_summary(summary, global_step)
                step += 1

                if self.config.total_steps and global_step >= self.config.total_steps:
                    return True
                if self.config.steps_per_epoch and step >= self.config.steps_per_epoch:
                    break
            except tf.errors.OutOfRangeError:
                break
        if step > 0:
            # avg_accuracy = float(total_accuracy) / step
            avg_loss = float(total_loss) / step
            # print("%d Epoch Avg Train Accuracy : %f" % (epoch, avg_accuracy))
            # self.summary_average(epoch, avg_accuracy, avg_loss)

            if epoch % self.config.save_interval == 0:
                self.saver.save(self.sess, self.config.log_dir + "/model_epoch_%d.ckpt" % epoch, global_step)

        return False

    def validate(self, epoch):
        dataset_path = os.path.join(self.config.dataset_dir,
                                    "%s_%s*tfrecord" % (self.config.dataset_name, self.config.validation_name))
        self.validation_dataset.init(dataset_path, self.config.input_size)
        total_accuracy = .0
        total_loss = .0
        step = 0
        total_steps = self.config.num_validation_sample // self.config.validation_batch_size
        if self.config.num_validation_sample % self.config.validation_batch_size > 0:
            total_steps += 1
        self.num_current_validation_embed = 0
        self.validation_embed_labels = None
        self.validation_embed_activations = None
        self.validation_embed_dataset = None

        while True:
            try:
                batch_xs, batch_ys = self.validation_dataset.get_next_batch()
                accuracy, loss, global_step, logits, pred_idx = self.sess.run(
                    [self.accuracy_op, self.loss_op, self.global_step, self.logits, self.pred_idx_op],
                    feed_dict={self.inputs: batch_xs, self.labels: batch_ys, self.is_training: False})
                total_accuracy += accuracy
                total_loss += loss
                now = datetime.now().strftime('%Y/%m/%d %H:%M:%S')

                print(
                    "[%s Validation] [Epoch: %3d] [Current Step: %4d / %4d] [Accuracy: %10f] [Cost: %10f]" % (
                        now, epoch, step + 1, total_steps,
                        accuracy, loss))
                if self.config.use_summary:
                    summary = self.sess.run(self.summary_op,
                                            feed_dict={self.inputs: batch_xs, self.labels: batch_ys,
                                                       self.is_training: False})
                    self.validation_writer.add_summary(summary, global_step + step)
                step += 1

                if self.config.use_validation_cam and self.cam:
                    self.create_cam(batch_xs, batch_ys, logits, False)

                if epoch % self.config.validation_embed_visualization_interval == 0:
                    self.add_embedding(batch_xs, batch_ys, logits, pred_idx, False)
            except tf.errors.OutOfRangeError:
                break

        if step > 0:
            avg_accuracy = float(total_accuracy) / step
            avg_loss = float(total_loss) / step
            print("%d Epoch - Validation, Avg Accuracy : %f, Avg Loss : %f" % (epoch, avg_accuracy, avg_loss))
            self.summary_average(epoch, avg_accuracy, avg_loss, False)
            self.summary_cam(epoch, False)

    def summary_average(self, epoch, avg_accuracy, avg_loss, is_train=True):
        if self.config.use_summary:
            # avg_accuracy_summary = tf.summary.scalar("average_accuracy", tf.convert_to_tensor(avg_accuracy))
            # avg_loss_summary = tf.summary.scalar("avg_loss", tf.convert_to_tensor(avg_loss))
            writer = self.train_writer if is_train else self.validation_writer
            writer.add_summary(self.sess.run(tf.summary.merge([self.avg_accuracy_summary, self.avg_loss_summary]),
                                             feed_dict={self.avg_accuracy_pl: avg_accuracy,
                                                        self.avg_loss_pl: avg_loss}), epoch)

    def add_embedding(self, xs, ys, logits, predict_idx, is_train=True):
        if is_train:
            if not self.config.use_train_embed_visualization:
                return
            batch_size = self.config.batch_size
            current_num = self.num_current_train_embed
            total_limit = self.config.num_train_embed_epoch
        else:
            if not self.config.use_validation_embed_visualization:
                return
            batch_size = self.config.validation_batch_size
            current_num = self.num_current_validation_embed
            total_limit = self.config.num_validation_embed_epoch

        if current_num >= total_limit:
            return

        num_embed = total_limit - current_num
        if num_embed > batch_size:
            num_embed = batch_size
        if is_train:
            self.num_current_train_embed += num_embed
        else:
            self.num_current_validation_embed += num_embed
        if num_embed < len(xs):
            xs = xs[:num_embed]
            ys = ys[:num_embed]
            logits = logits[:num_embed]
            predict_idx = predict_idx[:num_embed]

        if self.config.use_prediction_for_embed_visualization:
            predict_y = np.zeros((num_embed, self.config.num_class))
            predict_y[np.arange(num_embed), predict_idx] = 1
            tmp_labels = predict_y
        else:
            tmp_labels = ys

        if is_train:
            if self.train_embed_dataset is None:
                self.train_embed_dataset = xs
                self.train_embed_labels = tmp_labels
                self.train_embed_activations = logits
            else:
                self.train_embed_dataset = np.append(xs, self.train_embed_dataset, axis=0)
                self.train_embed_labels = np.append(tmp_labels, self.train_embed_labels, axis=0)
                self.train_embed_activations = np.append(logits, self.train_embed_activations, axis=0)
        else:
            if self.validation_embed_dataset is None:
                self.validation_embed_dataset = xs
                self.validation_embed_labels = tmp_labels
                self.validation_embed_activations = logits
            else:
                self.validation_embed_dataset = np.append(xs, self.validation_embed_dataset, axis=0)
                self.validation_embed_labels = np.append(tmp_labels, self.validation_embed_labels, axis=0)
                self.validation_embed_activations = np.append(logits, self.validation_embed_activations, axis=0)

    def inference(self):
        pass

    def restore_model(self):
        if self.config.restore_model_path and len(
          glob.glob(self.config.restore_model_path + ".data-00000-of-00001")) > 0:
            print("Restore model! %s" % self.config.restore_model_path)
            self.saver.restore(self.sess, self.config.restore_model_path)

    def create_cam(self, xs, ys, logits, is_train=True):
        if self.num_current_cam >= self.config.num_train_cam_epoch:
            return
        num_cam = self.config.num_train_cam_epoch - self.num_current_cam
        xs = xs[:num_cam]
        ys = ys[:num_cam]
        logits = logits[:num_cam]
        cam_imgs, class_indices = self.cam.create_cam_imgs(self.sess, xs, logits)
        for i in range(len(xs)):
            box_img = np.copy(xs[i])
            heatmap = self.cam.convert_cam_2_heatmap(cam_imgs[i][0])
            overlay_img = self.cam.overlay_heatmap(xs[i], heatmap)
            train_or_validation = "train" if is_train else "validation"

            if ys[i].argmax() == logits[i].argmax():
                key = "true/%s_label_%d" % (train_or_validation, ys[i].argmax())
            else:
                key = "false/%s_truth_%d_pred_%d" % (train_or_validation, ys[i].argmax(), logits[i].argmax())
            if key not in self.heatmap_imgs:
                self.heatmap_imgs[key] = []
            if len(xs[i].shape) != 3 or xs[i].shape[2] != 3:
                img = cv2.cvtColor(xs[i], cv2.COLOR_GRAY2BGR)[..., ::-1]
            else:
                img = xs[i]
            self.heatmap_imgs[key].append(img)
            self.heatmap_imgs[key].append(overlay_img[..., ::-1])
            if self.config.use_bounding_box_visualization:
                box_img = self.cam.draw_rectangle(box_img, cam_imgs[i][0], [255, 0, 0])
                self.heatmap_imgs[key].append(box_img)
            self.num_current_cam += 1

    def summary_cam(self, epoch, is_train=True):
        if is_train:
            writer = self.train_writer
        else:
            writer = self.validation_writer
        if self.config.use_train_cam and self.cam:
            for key in self.heatmap_imgs:
                self.cam.write_summary(writer, "grad_cam_epoch_%d_%s" % (epoch, key), self.heatmap_imgs[key],
                                       self.sess)
            self.heatmap_imgs = {}
            self.num_current_cam = 0

    @staticmethod
    def variable_summaries(var, summaries):
        name = var.op.name
        mean = tf.reduce_mean(var)
        summaries.add(tf.summary.scalar(name + '_mean', mean))
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        summaries.add(tf.summary.scalar(name + '_stddev', stddev))
        summaries.add(tf.summary.scalar(name + '_max', tf.reduce_max(var)))
        summaries.add(tf.summary.scalar(name + '_min', tf.reduce_max(var)))
        summaries.add(tf.summary.histogram(name, var))


def main(config):
    trainer = Trainer(config)
    trainer.run()
