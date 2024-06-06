import tensorflow as tf
import numpy as np
from OWMLayer_3Layers_ErrorCorrection import OWMLayer as MNIST_OWMLayer
from use_OWMLayer_2Layers import OWMLayer
import os
import random
import sys
import time
import datetime
from pyinstrument import Profiler
from tensorflow.python.client import device_lib
import math
from KSWIN import KSWIN
from skmultiflow.drift_detection import DDM
from CircularQueue import CircularQueue

class TSR:
    def __init__(self):
        self.name = "TSR"
        self.RVI_removal = 20
        random.seed(4)
        np.random.seed(4)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore warning

    def change_gpu(self, gpu):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    def init_parameters(self, dataset, batch_size=40, epochs=20, val_test_size=1, max_replay_per_batch = 20, alpha=0.0009):
        # Parameters
        # ==================================================
        tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")
        tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
        tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
        tf.app.flags.DEFINE_string("buckets", "", "")
        tf.app.flags.DEFINE_string("checkpointDir", "", "oss info")
        tf.flags.DEFINE_integer("num_class", dataset.num_classes, "")
        tf.flags.DEFINE_integer("batch_size", batch_size, "Batch Size (default: 64)")
        tf.flags.DEFINE_integer("epoch", epochs, "")
        self.FLAGS = tf.flags.FLAGS
        self.val_test_size = val_test_size
        self.no_ec_per_batch = max_replay_per_batch
        self.alpha = alpha
        # ==================================================
        test_type = "{}_{}_RESULTS.txt".format(self.name, dataset.name)
        new_dir = os.getcwd() + "/{} Results/".format(self.name)
        # new_dir = os.getcwd() + "/{} Ablation Results/".format(self.name)
        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)
        sys.stdout = open(new_dir + test_type, "w+")
        # ==================================================

    def running_mean(self, x, m):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[m:] - cumsum[:-m]) / float(m)

    def train(self, dataset):
        m = 2
        #Get ALL imagenet test set data in one_hot format for use with feed_dict
        n_values = np.max(dataset.y_test) + 1
        all_labels = np.eye(n_values)[dataset.y_test]
        all_labels = np.reshape(all_labels, (len(all_labels), dataset.num_classes))

        # Training
        # ==================================================
        g1 = tf.Graph()
        middle = 800
        with g1.as_default():
            if "MNIST" in dataset.name:
                OWM = MNIST_OWMLayer([[dataset.features + 1, middle], [middle + 1, middle], [middle + 1, dataset.num_classes]], seed_num=79)
            else:
                OWM = OWMLayer([[dataset.features + 1, middle], [middle + 1, dataset.num_classes]], seed_num=79, features=dataset.features, classes=dataset.num_classes)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # CEC dict for storing incorrect val examples from previous tasks
        dict_incorrect_valid = {}
        # TSR: Dict for storing a list of confidences for each task
        task_drift = {}

        with tf.Session(graph=g1, config=config) as sess1:
            # Initialize all variables
            init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
            sess1.run(init)
            task_num = dataset.num_tasks
            task_circ = CircularQueue(task_num)
            task_replay = {}
            for i in range(task_num):
                task_drift[i] = KSWIN(alpha=self.alpha)
                task_replay[i] = 0

            # For removing volatile instances
            variation_counts = [0] * task_num
            prev_val_TFs = [0] * task_num

            for j in range(0, task_num):
                start = time.time()
                print("Training Disjoint " + dataset.name + " %d" % (j + 1))
                #CC: Print timestamp
                print(datetime.datetime.now())
                # Update the parameters
                epoch_owm = self.FLAGS.epoch
                batch_size_owm = self.FLAGS.batch_size
                all_data = len(dataset.data_list[j].train.labels[:])
                print("No task training instances", all_data)

                no_training_per_batch = batch_size_owm
                all_step = all_data * epoch_owm // batch_size_owm

                # CEC: number of batches for current training task
                if all_data % batch_size_owm == 0:
                    no_batches = all_data // batch_size_owm
                else:
                    no_batches = all_data // batch_size_owm + 1

                # CEC: Checks if variation examples have all been added
                finished = False

                task_incorrect_xs = {}
                task_incorrect_ys = {}
                for k in range(j):
                    task_incorrect_xs[k], task_incorrect_ys[k] = dataset.data_list[k].validation.images[
                        dict_incorrect_valid[k][0]], dataset.data_list[k].validation.labels[dict_incorrect_valid[k][0]]
                incorrect_index = 0

                for current_step in range(all_step):
                    # TSR: Validation test for prior tasks
                    for _ in range(min(m, j)):
                        i_val = task_circ.read()
                        validation_test_indexes = random.sample(range(0, len(dataset.data_list[i_val].validation.images[:])),
                                                                self.val_test_size)
                        feed_dict = {
                            OWM.input_x: [dataset.data_list[i_val].validation.images[x] for x in validation_test_indexes],
                            OWM.input_y: [dataset.data_list[i_val].validation.labels[x] for x in validation_test_indexes],
                        }

                        _, _, _, logits = sess1.run(
                            [OWM.accuracy, OWM.loss, OWM.predicted_validation_examples, OWM.scores], feed_dict)
                        # TSR: Add logits to task's drift detector
                        logit_list = [logit[i_val] for logit in logits]
                        #for l in logit_list:
                        #    task_drift[i_val].add_element(i)
                        task_drift[i_val].add_element(logit_list)
                    lamda = current_step/all_step
                    batch_xs, batch_ys = dataset.data_list[j].train.next_batch(no_training_per_batch)
                    batch_incorrect_xs = np.empty(shape=(0, batch_xs.shape[1]))
                    batch_incorrect_ys = np.empty(shape=(0, batch_ys.shape[1]))
                    for i_val in task_drift:
                        replays = 0
                        # TSR: If task drift then add replay examples to batch
                        if finished == False and j > 0 and task_drift[i_val].detected_change():
                            # TSR: Get batch of incorrect examples
                            if (incorrect_index + self.no_ec_per_batch) > len(task_incorrect_xs[i_val]):
                                batch_incorrect_xs = np.vstack(
                                    [batch_incorrect_xs, task_incorrect_xs[i_val][incorrect_index:]])
                                batch_incorrect_ys = np.vstack(
                                    [batch_incorrect_ys, task_incorrect_ys[i_val][incorrect_index:]])
                                finished = True
                                incorrect_index = 0
                                replays += len(task_incorrect_xs[i_val][incorrect_index:])
                            else:
                                batch_incorrect_xs = np.vstack([batch_incorrect_xs, task_incorrect_xs[i_val][
                                                                                    incorrect_index:(
                                                                                            incorrect_index + self.no_ec_per_batch)]])
                                batch_incorrect_ys = np.vstack([batch_incorrect_ys, task_incorrect_ys[i_val][
                                                                                    incorrect_index:(
                                                                                            incorrect_index + self.no_ec_per_batch)]])
                                incorrect_index += self.no_ec_per_batch
                                replays += len(
                                    task_incorrect_xs[i_val][incorrect_index:(incorrect_index + self.no_ec_per_batch)])
                        task_replay[i_val] += replays
                    p = np.random.permutation(len(batch_incorrect_xs))
                    batch_incorrect_xs = batch_incorrect_xs[p]
                    batch_incorrect_ys = batch_incorrect_ys[p]

                    # CEC: Combine training batch and incorrect batch to get full batch
                    batch_xs = np.vstack([batch_xs, batch_incorrect_xs])
                    batch_ys = np.vstack([batch_ys, batch_incorrect_ys])

                    # CEC: Reset Finished for new epoch
                    if current_step % no_batches == 0:
                        finished = False

                    # 3 or 2 layer
                    if "MNIST" in dataset.name:
                        alpha_arr = np.array([[0.9 * 0.001 ** lamda, 1.0 * 0.1 ** lamda, 0.6]])
                        lr_arr = np.array([[0.2]])
                    else:
                        lr_arr = np.array([[0.1]])
                        alpha_arr = np.array([[1.0 * 0.005 ** lamda, 1.0]])
                    feed_dict = {
                        OWM.input_x: batch_xs,
                        OWM.input_y: batch_ys,
                        OWM.input_x_validation_incorrect: batch_incorrect_xs,
                        OWM.input_y_validation_incorrect: batch_incorrect_ys,
                        OWM.lr_array: lr_arr,
                        OWM.alpha_array: alpha_arr,
                    }
                    acc, loss,  _, = sess1.run([OWM.accuracy, OWM.loss, OWM.back_forward], feed_dict,)
                end = time.time()
                task_circ.add(j)
                print("Training time for task {}: {}".format(j + 1, end - start))
                print("Test on Previous Datasets:")
                for i_test in range(j + 1):
                    feed_dict = {
                        OWM.input_x: dataset.data_list[i_test].test.images[:],
                        OWM.input_y: dataset.data_list[i_test].test.labels[:],
                    }
                    accu, loss = sess1.run([OWM.accuracy, OWM.loss], feed_dict)
                    print("Test:->>>[{:d}/{:d}], acc: {:.2f} % Replays: {:d}".format(i_test + 1, task_num, accu * 100, task_replay[i_test]))
                #DBP validation accuracy
                print("Validation on Previous Datasets:")
                for i_val in range(j + 1):
                    feed_dict = {
                        OWM.input_x: dataset.data_list[i_val].validation.images[:],
                        OWM.input_y: dataset.data_list[i_val].validation.labels[:],
                    }
                    #DBP include correct and incorrect validation examples indices
                    accu, loss, predicted_validation_examples = sess1.run([OWM.accuracy, OWM.loss, OWM.predicted_validation_examples], feed_dict)
                    print("Validation:->>>[{:d}/{:d}], acc: {:.2f} %".format(i_val + 1, task_num, accu * 100))
                    list_val_TFs = list(predicted_validation_examples)
                    # CC: Update variation counts
                    if i_val != j:
                        for i_TF in range(len(list_val_TFs)):
                            if list_val_TFs[i_TF] != prev_val_TFs[i_val][i_TF]:
                                variation_counts[i_val][i_TF] += 1
                    else:
                        variation_counts[j] = [0] * len(list_val_TFs)

                    prev_val_TFs[i_val] = list_val_TFs
                    dict_incorrect_valid[i_val] = np.where((predicted_validation_examples == False))
            feed_dict = {
                OWM.input_x: dataset.x_test[:],
                OWM.input_y: all_labels[:],
            }
            accu, loss = sess1.run([OWM.accuracy, OWM.loss], feed_dict)
            print("Average Final Test Accuracy over all Tasks {:g} %\n".format(accu * 100))
            print(datetime.datetime.now(), "END")