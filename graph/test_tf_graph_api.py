import tensorflow as tf
from graph.tf_graph_api import GraphAPI
from six import add_metaclass


def main():
    from unittest import TestCase
    tf.reset_default_graph()
    graph_api = GraphAPI(reuse_variables=True, log=False)

    @add_metaclass(graph_api())
    class Api(object):

        def __init__(self):
            super(Api, self).__init__()

        @property
        def graph(self):
            return super(Api, self).graph

        @property
        def session(self):
            return super(Api, self).session

    class Meta(Api):

        def meta_method1(self, value, name='Test'):
            return tf.Variable(initial_value=value, name=name)

        def meta_method2(self, value, name='Test'):
            return tf.Variable(initial_value=value, name=name)

    class Reader(Api):

        def trial(self, value, name='Test'):
            return tf.Variable(initial_value=value, name=name)

    class TestAPI(TestCase):

        def __init__(self):
            super(TestAPI, self).__init__()
            self._meta = Meta()
            self._reader = Reader()

            self._meta_method1 = self._meta.meta_method1(1, 'one')
            self._meta_method2 = self._meta.meta_method2(2, 'two')
            self._reader_trial = self._reader.trial(2, 'two')

        def test_names(self):
            self.assertEqual(self._meta_method1.name, 'Meta/meta_method_1/one:0')
            self.assertEqual(self._meta_method2.name, 'Meta/meta_method_2/two:0')

        def test_graph(self):
            self.assertEqual(self._meta.graph, tf.get_default_graph())
            self.assertEqual(self._reader.graph, tf.get_default_graph())
            self.assertEqual(self._meta.graph, self._reader.graph)

        def test_session(self):
            with self._meta.session.as_default():
                self.assertEqual(self._meta.session, tf.get_default_session())

            with self._reader.session.as_default():
                self.assertEqual(self._reader.session, tf.get_default_session())

        def test_operations(self):
            self.assertEqual(self._meta.graph.get_operations(), self._reader.graph.get_operations())

    test = TestAPI()
    test.test_graph()
    test.test_names()
    test.test_session()
    test.test_operations()


if __name__ == '__main__':
    main()
    import cProfile
    cProfile.run('main()')
