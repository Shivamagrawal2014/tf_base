from graph.import_graph import ImportGraph
import tensorflow as tf


class RestoreModel(ImportGraph):

    def __init__(self,
                 chk_pt_path,
                 tf_record_path,
                 batch_size,
                 epochs,
                 session
                 ):
        """
        :param chk_pt_path:
        :param batch_size:
        :param epochs:
        :param save_dir:
        :param session:
        """

        self._chk_pt_path = chk_pt_path
        self._tf_record_path = tf_record_path
        self._batch_size = batch_size
        self._epochs = epochs

        self._global_step = None
        self._batch_count_per_epoch = None
        self._total_batch = None

        self._default_session = session
        self._default_graph = session.graph
        super().__init__(dir_path=self._chk_pt_path, session=self._default_session)

        self._data_init = None
        self._net_init = None
        self._writer = None

    def batch_count(self):

        number_of_examples = sum([1 for _ in
                                  tf.python_io.tf_record_iterator(self._tf_record_path, options='GZIP')])
        print('number_of_examples :', number_of_examples)
        print('batch size selected :', self._batch_size)
        batch_count_per_epoch = int(number_of_examples / self._batch_size)
        print('batch_count_per_epoch :', batch_count_per_epoch)

        if self._batch_count_per_epoch is None:
            self._batch_count_per_epoch = batch_count_per_epoch
        elif self._batch_count_per_epoch is not None and \
                self._batch_count_per_epoch != batch_count_per_epoch:
            self._batch_count_per_epoch = batch_count_per_epoch

        total_batch = self._batch_count_per_epoch * self._epochs
        if self._total_batch is None:
            self._total_batch = total_batch
        elif self._total_batch is not None and self._total_batch != total_batch:
            self._total_batch = total_batch

        print('number_of_batches :', total_batch)
        return self._batch_count_per_epoch, self._total_batch

    def restore_points(self):
        # checks if there exits the checkpoint already and tries to
        # find next iteration parameters
        latest_ckpt = tf.train.latest_checkpoint(self._dir)
        if latest_ckpt is None:
            self.batch_count()
            start_iter = 1
            end_iter = self._total_batch + 1
        else:
            last_hyphen_pos = latest_ckpt.rfind('-')
            batch_count = int(latest_ckpt[last_hyphen_pos + 1:])
            start_iter = batch_count + 1
            self.batch_count()
            end_iter = start_iter + self._total_batch + 1
        return start_iter, end_iter

    def read_graph_from_chk_pt(self, restore_point, model_name=None, graph_net=None):
        # tries to restore the checkpoints of the Variables as well attempts
        # to remake the graph from meta file if it exits

        try:
            # This part of model checks whether there exists a saved
            #  model as a meta file of graph and tires to restore it
            self.import_meta_graph_from_checkpoint_meta()
        except (FileNotFoundError, NotADirectoryError):
            if graph_net is not None:
                # if no meta file of pre defined graph is found then
                # a new net for the model is created based on the predefined net filters
                graph = self._graph
                sess = self._session
                with graph.as_default():
                    with sess.as_default():
                        with tf.variable_scope(model_name):
                            if self._net_init is None:
                                self._net_init = graph_net()

                            graph_nodes_name_list = [i.name for i in
                                                     self._default_graph.as_graph_def().node]
                            if self._global_step is None:
                                if model_name + '/global_step' in graph_nodes_name_list:
                                    self._global_step = tf.get_default_graph().get_tensor_by_name(
                                        name=model_name + '/global_step:0')
                            start = self._global_step.eval()  # get last global_step
                            assert start == (restore_point - 1)
                            print("Starting after:", start)
                            print('Session resuming with batch :', restore_point)
            else:
                raise
        # Now that the model has restored the graph attempting for
        # restoring all variables/weights of the model
        self.import_weights_from_checkpoint()
        print('model restored....')

    def read_graph_from_proto(self, proto_dir, batch_number, as_constant=None):
        self.read_graph_proto(proto_dir=proto_dir, batch_number=batch_number,
                              as_constants=as_constant)

    def write_graph(self, proto_dir, batch_number, as_constant=None):
        if as_constant in (None, False):
            self.write_graph_and_var_to_proto(proto_dir=proto_dir,
                                              batch_number=batch_number)
        else:
            self.write_graph_and_const_to_proto(proto_dir=proto_dir,
                                                batch_number=batch_number)

    def restore_model_chk_pt(self, restore_point, model_name=None):
        if restore_point != 1:
            print('Checkpoints Found!')
            print('Attempting to restore the Model! ')
            print('Restoring the Model...')
            self.read_graph_from_chk_pt(restore_point=restore_point,
                                        model_name=model_name)
            print('Model Restored!, Resuming Old Training!...')
        elif restore_point == 1:
            print('Starting New Training...')

    def restore_model_proto(self,
                            proto_dir,
                            batch_number,
                            as_constant):
        print('ProtoBuf Found!')
        print('Attempting to restore the Model! ')
        print('Restoring the Model...')
        self.read_graph_from_proto(proto_dir=proto_dir,
                                   batch_number=batch_number,
                                   as_constant=as_constant)

    def chk_pt_path(self):
        return self._chk_pt_path

    def __call__(self,
                 restore_from_proto,
                 summary_dir=None,
                 proto_dir=None,
                 batch_number=None,
                 as_constant=None):
        restore_point, end_point = self.restore_points()

        if not restore_from_proto:
            self.restore_model_chk_pt(restore_point=restore_point)
        else:
            self.restore_model_proto(proto_dir=proto_dir,
                                     batch_number=batch_number,
                                     as_constant=as_constant)
        writer = self.write_summary(summary_dir=summary_dir)
        writer.close()


if __name__ == '__main__':
    from pprint import pprint
    ckpt_dir = r'C:/Test_Net/'
    tf_record_path = r'C:\Users\shivam.agarwal\PycharmProjects\TopicAPI\record\testing_data.tfr'
    summary_dir = '../../summary/'
    proto_dir = r'C:\Users\shivam.agarwal\PycharmProjects\TopicAPI\proto_dir'
    as_constants = True
    batch_number = 3
    session = tf.Session()

    restorer = RestoreModel(chk_pt_path=ckpt_dir,
                            tf_record_path=tf_record_path,
                            batch_size=batch_number,
                            epochs=2000,
                            session=session)

    restorer(restore_from_proto=False,
             summary_dir=summary_dir,
             proto_dir=proto_dir,
             batch_number=1620,
             as_constant=False)
    pprint(restorer.session.graph.get_operations())

    print(restorer.batch_count())
