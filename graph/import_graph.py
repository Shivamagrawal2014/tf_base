import tensorflow as tf
import os
from memory_profiler import profile


class ImportGraph(object):

    def __init__(self, dir_path, session):
        self._dir = dir_path
        self._graph = None
        self._session = None
        self._meta_graph_imported = None
        self._weights_imported = None
        self._convert_var_to_const = None

        self._input_maps = None
        self._constants_maps = None
        self.graph = session.graph
        self.session = session

    def dir(self):
        return self._dir

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, graph=None):
        assert isinstance(graph, tf.Graph)
        self._graph = graph

    @property
    def session(self):
        return self._session

    @session.setter
    def session(self, session=None):
        assert isinstance(session, tf.Session)
        self._session = session

    @profile
    def import_meta_graph_from_checkpoint_meta(self):
        assert self._meta_graph_imported is None
        graph = self.graph
        sess = self.session
        with graph.as_default():
            with sess.as_default():
                if tf.gfile.Exists(self._dir):
                    ckpt = tf.train.get_checkpoint_state(self._dir)
                    print('looking for file : ', ckpt.model_checkpoint_path + '.meta')
                    if tf.gfile.Exists(ckpt.model_checkpoint_path + '.meta'):
                        print(ckpt.model_checkpoint_path + '.meta', 'exists')
                    else:
                        raise FileNotFoundError(
                            '{file} not found'.format(file=ckpt.model_checkpoint_path + '.meta'))

                    print(ckpt.model_checkpoint_path)
                    print(os.path.dirname(os.path.dirname(self._dir)))
                    if ckpt and ckpt.model_checkpoint_path:
                        loader = tf.train.import_meta_graph(
                            os.path.join(os.path.dirname(os.path.dirname(self._dir)),
                                         '.'.join([ckpt.model_checkpoint_path, 'meta'])))
                        loader.restore(self.session, ckpt.model_checkpoint_path)
                        sess.run(tf.tables_initializer())
                        self._meta_graph_imported = True

                else:
                    raise NotADirectoryError("{directory} not found".format(directory=self._dir))

    @profile
    def import_weights_from_checkpoint(self):
        assert self._weights_imported is None
        graph = self.graph
        sess = self.session
        with graph.as_default():
            with sess.as_default():
                if tf.gfile.Exists(self._dir):
                    ckpt = tf.train.get_checkpoint_state(self._dir)
                    if ckpt and ckpt.model_checkpoint_path:
                        saver = tf.train.Saver()
                        saver.restore(sess=self.session,
                                      save_path=os.path.join(self._dir, ckpt.model_checkpoint_path))

                        self._weights_imported = True
                else:
                    raise NotADirectoryError("{directory} not found".format(directory=self._dir))

    def convert_variables_to_constants(self, collection_names=None):
        assert self._convert_var_to_const is None
        graph = self._graph
        sess = self._session
        with graph.as_default():
            with sess.as_default():
                if collection_names is None:
                    collection_names = [tf.GraphKeys.VARIABLES]
                output_node_names = []
                _ = [output_node_names.extend(self.graph.get_collection(coll)) for coll in collection_names]
                output_node_names = [i.name.split(':0')[0] for i in output_node_names]
                print('output_node_names', output_node_names)
                tf.graph_util.convert_variables_to_constants(sess=self.session,
                                                             input_graph_def=self.graph.as_graph_def(),
                                                             output_node_names=output_node_names)
        self._convert_var_to_const = True

    @profile
    def input_maps_of_var_from_restored_variables(self, return_input_maps=None):
        if return_input_maps is None:
            return_input_maps = True
        graph = self.graph
        sess = self.session
        with graph.as_default():
            with sess.as_default():
                if tf.gfile.Exists(self._dir):
                    ckpt = tf.train.get_checkpoint_state(self._dir)
                    if ckpt and ckpt.model_checkpoint_path:
                        if self._meta_graph_imported is None:
                            self.import_meta_graph_from_checkpoint_meta()
                        if return_input_maps:
                            print('attempting to map input_map from trainable variables')
                            input_maps = {}

                            for v in tf.trainable_variables():
                                input_maps[v.value().name] = self.session.run(v)
                            print('completed mapping input_map from trainable variables')

                            return input_maps
                        else:
                            return True
                else:
                    raise FileNotFoundError('%s not Found' % self._dir)

    def _constant_maps_of_variables_from_input_maps(self):

        # Get the Graph Variables mapped for input
        if self._input_maps is None:
            self._input_maps = self.input_maps_of_var_from_restored_variables()
        # self.tf.reset_default_graph()
        if self._constants_maps is None:
            self._constants_maps = {}
        self.graph = tf.Graph()
        if self._input_maps:
            # Loading New Graph along with the
            for key in self._input_maps:
                print(key.split(':0')[0])
                self._constants_maps[key] = tf.constant(self._input_maps[key], name=key.split(':0')[0])

                print(self._constants_maps[key].name)

        num_of_nodes = 0
        for node in self.session.graph_def.node:
            if node.name.endswith('read_1'):
                node.name = node.name.split('read_1')[0] + 'read:0'
            print(node.name)
            num_of_nodes += 1
        print('number_of_nodes :', str(num_of_nodes))

        return self._constants_maps

    @profile
    def write_graph_and_var_to_proto(self, proto_dir=None, batch_number=None):
        if proto_dir is None:
            proto_dir = self._dir
        if self._meta_graph_imported is None:
            self.import_meta_graph_from_checkpoint_meta()
        if self._weights_imported is None:
            self.import_weights_from_checkpoint()
        graph_proto_name = self._graph_proto_name(batch_number=batch_number, as_constants=False)
        print(graph_proto_name)
        print([node.name for node in self.session.graph_def.node])
        return tf.train.write_graph(self.session.graph_def, proto_dir, graph_proto_name, False)

    @profile
    def write_graph_and_const_to_proto(self, proto_dir=None, batch_number=None):
        if proto_dir is None:
            proto_dir = self._dir

        _ = self._constant_maps_of_variables_from_input_maps()
        del self._constants_maps
        del self._input_maps
        graph_proto_name = self._graph_proto_name(batch_number=batch_number, as_constants=True)
        print(graph_proto_name)
        print('Total Nodes :', sum([1 for _ in self.session.graph_def.node]))
        return tf.train.write_graph(self.session.graph_def, proto_dir, graph_proto_name, False)

    @staticmethod
    def _graph_proto_name(batch_number=None, as_constants=None):

        if as_constants:
            if batch_number is None:
                graph_proto_name = 'graph_as_consts.pb'
            else:
                graph_proto_name = 'graph_%s_as_consts.pb' % str(batch_number)
        else:
            if batch_number is None:
                graph_proto_name = 'graph_as_vars.pb'
            else:
                graph_proto_name = 'graph_%s_as_vars.pb' % str(batch_number)
        return graph_proto_name

    @profile
    def read_graph_proto(self, proto_dir=None, batch_number=None, as_constants=None, new_filename_queue=None):
        import os

        if proto_dir is None:
            proto_dir = self._dir
        else:
            proto_dir = proto_dir

        if not os.path.isabs(proto_dir):
            proto_dir = os.path.abspath(proto_dir)
        print('Proto Dir :', proto_dir)

        graph_proto_name = self._graph_proto_name(batch_number=batch_number, as_constants=as_constants)
        pb_file = os.path.join(proto_dir, graph_proto_name)
        print('pb File : ', pb_file)
        if tf.gfile.Exists(pb_file):
            with tf.gfile.GFile(pb_file, mode='rb') as f:
                graph_def = self.graph.as_graph_def()
                graph_def.ParseFromString(f.read())

                if new_filename_queue:
                    if tf.get_default_graph().get_collection('input_filename_queue'):
                        filename_queue = tf.get_default_graph().get_collection('input_filename_queue')[0]
                        tf.import_graph_def(graph_def=graph_def,
                                            input_map={filename_queue.name: new_filename_queue})
                    else:
                        tf.import_graph_def(graph_def=graph_def)
                else:
                    tf.import_graph_def(graph_def=graph_def)
                print([node for node in self.session.graph_def.node])

        else:
            raise NotADirectoryError('{directory} not Found'.format(directory=os.path.dirname(pb_file)))

    def read_graph_def_from_proto(self, proto_dir, batch_number=None, as_constants=None):

        if proto_dir is None:
            proto_dir = self._dir
        graph_proto_name = self._graph_proto_name(batch_number=batch_number, as_constants=as_constants)
        if tf.gfile.Exists(proto_dir):
            with tf.gfile.GFile(os.path.join(proto_dir, graph_proto_name), mode='rb') as f:
                graph_def = self.graph.as_graph_def()
                graph_def.ParseFromString(f.read())

            return graph_def
        else:
            raise FileNotFoundError('File %s does not exists!' % self._dir)

    def read_graph_from_graph_def(self, graph_def=None, as_constant=None):

        if tf.gfile.Exists(self._dir):
            if as_constant is not None:
                if as_constant is True:
                    input_map = self._constant_maps_of_variables_from_input_maps()
                else:
                    input_map = self.input_maps_of_var_from_restored_variables()

            else:
                input_map = None
        else:
            input_map = None

        if graph_def is None:
            _graph_def = self.session.graph.as_graph_def()
        else:
            _graph_def = graph_def
        if input_map is not None:
            tf.import_graph_def(graph_def=_graph_def,
                                input_map={key: input_map[key] for key in input_map.keys()})
        else:
            tf.import_graph_def(graph_def=_graph_def)

    @profile
    def write_summary(self, summary_dir=None):
        if summary_dir is None:
            summary_dir = '../../summary_dir/'

        writer = tf.summary.FileWriter(summary_dir, graph=self.graph)
        return writer


def restore_graph(checkpoint_dir=None, batch_number=None, as_constants=None):
    if checkpoint_dir is None:
        ckpt_dir = 'C:/Test_Net/'
        checkpoint_dir = ckpt_dir
    session = tf.Session()
    summary_dir = '../../summary/'
    proto_dir = '../../proto_dir/'
    saved_graph = ImportGraph(checkpoint_dir, session)

    try:
        print('Trying to read a Maybe-Proto File ...')
        saved_graph.read_graph_proto(proto_dir=proto_dir, batch_number=batch_number, as_constants=as_constants)
    except Exception as e:
        print('Writing New Protobuf File :')
        if as_constants:
            saved_graph.write_graph_and_const_to_proto(proto_dir=proto_dir, batch_number=batch_number)
        else:
            saved_graph.write_graph_and_var_to_proto(proto_dir=proto_dir, batch_number=batch_number)
    finally:
        return saved_graph
        # saved_graph.write_summary(summary_dir=summary_dir)


def check_restore_graph():
    from pprint import pprint
    save_g = restore_graph(as_constants=True)
    with save_g.graph.as_default():
        with save_g.session.as_default():
            pprint([i.name for i in save_g.graph.get_operations() if i.name.endswith('W')])


if __name__ == '__main__':
    ckpt_dir = 'C:/Test_Net/'
    checkpoint_dir = ckpt_dir
    as_constants = True
    summary_dir = '../../summary/'
    proto_dir = '../../proto_dir/'
    batch_number = 1620
    session = tf.Session()
    saved_graph = ImportGraph(checkpoint_dir, session)
    try:
        print('Trying to read a Maybe-Proto File ...')
        saved_graph.read_graph_proto(proto_dir=proto_dir, batch_number=batch_number, as_constants=as_constants)
    except Exception as e:
        print('Writing New Protobuf File :')
        if as_constants:
            saved_graph.write_graph_and_const_to_proto(proto_dir=proto_dir, batch_number=batch_number)
        else:
            saved_graph.write_graph_and_var_to_proto(proto_dir=proto_dir, batch_number=batch_number)
    finally:
        writer = saved_graph.write_summary()
        writer.close()
