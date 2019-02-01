from graph.restore_model import RestoreModel
import tensorflow as tf


class ModelBase(RestoreModel):

    def __init__(self,
                 file_path,
                 session: tf.Session,
                 batch_size,
                 epochs,
                 data_device,
                 net_device,
                 summary_dir,
                 save_dir
                 ):

        self._file_path = file_path
        self._batch_size = batch_size
        self._epochs = epochs
        self._data_device = data_device
        self._net_device = net_device
        self._summary_dir = summary_dir
        self._save_dir = save_dir

        self._global_step = None
        self._batch_count_per_epoch = None
        self._total_batch = None

        self._default_graph = session.graph
        self._default_session = session
        self._data_init = None
        self._net_init = None
        self._writer = None

    def mini_batch(self, batch_name):
        with tf.get_default_graph().as_default():
            with tf.get_default_session().as_default():
                with tf.name_scope('Data'):
                    data = self.data(batch_name=batch_name).batch()
        return data

    def forward_pass(self, features):
        if tf.get_default_graph().get_collection('Conv2D'):
            first_conv2d_layer = tf.get_default_graph().get_collection('Conv2D')[0]
            with tf.control_dependencies([first_conv2d_layer.assign(features)]):
                print("New Batch of Images Passed to old graph...")
                print('Calculating the Logits for features ')
                return tf.get_default_graph().get_collection('FinalFullyConnect')[0]
        else:
            with tf.name_scope('Forward_Pass'):
                return self.logits(features=features)

    def backward_pass(self, logits, labels):
        if tf.get_default_graph().get_collection('target_ops'):
            pred_ops = tf.get_default_graph().get_collection('pred')[0]
            tf.control_dependencies(pred_ops)
        else:
            with tf.name_scope(name='Backward_Pass'):
                with tf.name_scope(name='Prediction'):
                    pred = tf.nn.softmax(logits=logits)
                target = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
                with tf.name_scope(name='Cost'):
                    cost = tf.reduce_mean(target)

                # specify optimizer
                with tf.name_scope('Train'):
                    # optimizer is an "operation" which we can execute in a session
                    train_op = tf.train.GradientDescentOptimizer(
                        learning_rate=.0001).minimize(cost, global_step=self._global_step)

                print('shape prediction :%s , labels :%s ' % (logits.shape, labels.shape))

                with tf.name_scope('Accuracy'):
                    # Accuracy
                    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
                tf.summary.histogram('Loss Histogram', cost)

                tf.add_to_collection('train_op', train_op)
                tf.add_to_collection('cost_op', cost)
                tf.add_to_collection('target', target)
                tf.add_to_collection('pred', pred)

            return target, pred, cost, accuracy, train_op

    @staticmethod
    def scalar_summary(cost, accuracy):
        # create a summary for our cost and accuracy
        return tf.summary.scalar("Cost", cost), tf.summary.scalar("Accuracy", accuracy)

    @staticmethod
    def image_summary(image, max_outputs):
        # create a summary for our image batch
        return tf.summary.image('Images', image, max_outputs=max_outputs)

    @staticmethod
    def summary_merged_ops():
        # create the summary ops explicitly
        # merge all summaries into a single "operation" which we can execute in a session
        return tf.summary.merge_all()

    def summary_writer(self, summary_path, sess):
        # creates a summary writer file and passes it to the model to
        # save the model summary such as feature, pred, cost and other
        # tensors and scalars
        if self._writer is None:
            self._writer = tf.summary.FileWriter(summary_path, sess.graph)
        return self._writer

    def init_net_data_and_global_step(self):
        if self._data_init is None:
            self._data_init = Data.__init__(self,
                                            file=self._file_path,
                                            number_examples=self._batch_size,
                                            number_epochs=self._epochs,
                                            device=self._data_device)

        if self._net_init is None:
            self._net_init = Net.__init__(self,
                                          filters=self._filters_shape_list,
                                          device=self._net_device)

        if self._global_step is None:
            self._global_step = tf.get_variable(name='global_step',
                                                initializer=0,
                                                dtype=tf.int32,
                                                trainable=False)

    def run(self,
            restore_point,
            end_point,
            cost,
            accuracy,
            summary_ops,
            train_ops,
            writer=None,
            session=None,
            ):

        if restore_point == 1:
            epoch = 0
        else:
            epoch = int(restore_point / self._batch_count_per_epoch)
        saver_all = tf.train.Saver(max_to_keep=4)

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        if session:
            sess = session
        else:
            sess = self._default_session.as_default()
        writer_was_none = False

        if writer is None:

            if self._writer:
                writer = self._writer
            else:
                writer_was_none = True
                writer = tf.summary.FileWriter(logdir=self._summary_dir, session=sess)
        else:
            writer = writer

        for i in range(restore_point, end_point):
            batch_number = i
            try:
                # cost = tf.Print(cost, [cost], 'Cost :')
                # accuracy = tf.Print(accuracy, [accuracy], 'Accuracy :')
                cost_v, accuracy_v, summary_v, _ = sess.run([cost, accuracy, summary_ops, train_ops],
                                                            options=run_options, run_metadata=run_metadata)

                writer.add_summary(summary_v, batch_number)
                writer.add_run_metadata(run_metadata=run_metadata, tag='step_%s' % str(batch_number))

                if batch_number % self._batch_count_per_epoch == 0:
                    epoch += 1
                    if epoch % 10 == 0 and epoch > 0:
                        saver_all.save(sess=sess, save_path=self._save_dir + '/save.ckpt',
                                       global_step=self._global_step)

                        # writing different versions of the graph model in protobuffer file
                        self.write_graph(batch_number=batch_number)

                        # Create the Timeline object, and write it to a json
                        tl = timeline.Timeline(run_metadata.step_stats)
                        ctf = tl.generate_chrome_trace_format()
                        with open('timeline_%s.json' % str(batch_number), 'w') as f:
                            f.write(ctf)

                    print('Epoch :', epoch)
                    print('Batch_Number :', batch_number)
                    print('Cost :', cost_v, 'Train_Accuracy :', accuracy_v)

            except tf.errors.OutOfRangeError:
                print('Data Exhausted!')
                break
            if writer_was_none:
                writer.close()

    def __call__(self, model_name, batch_name, modify_feature=None):

        restore_point, end_point = self.restore_points()
        self.restore_model(model_name=model_name, restore_point=restore_point)
        with self._default_graph.as_default():
            with self._default_session.as_default() as sess:
                if self._net_init is not None:
                    model_scope_name = model_name+'/'
                else:
                    model_scope_name = model_name

                with tf.variable_scope(model_scope_name):
                    self.init_net_data_and_global_step()
                    if modify_feature is None:
                        modify_feature = True
                    else:
                        modify_feature = modify_feature

                    data = self.mini_batch(batch_name=batch_name)
                    features, labels = data
                    logits = self.forward_pass(features=features)
                    init_g = tf.global_variables_initializer()
                    sess.run(init_g)
                    target, pred, cost, accuracy, train_ops = self.backward_pass(logits=logits, labels=labels)
                    self.scalar_summary(cost=cost, accuracy=accuracy)
                    self.image_summary(image=images, max_outputs=self._batch_size)
                    summary_ops = self.summary_merged_ops()
                    writer = self.summary_writer(summary_path=self._summary_dir, sess=sess)
                    self.run(restore_point=restore_point, end_point=end_point, cost=cost,
                             accuracy=accuracy, summary_ops=summary_ops, train_ops=train_ops,
                             session=sess, writer=writer)

                    writer.close()

    def default_graph(self):
        return self._default_graph

    def default_session(self):
        return self._default_session

    def file(self):
        return self._file_path

    def summary_dir(self):
        return self._summary_dir

    def save_dir(self):
        return self._save_dir