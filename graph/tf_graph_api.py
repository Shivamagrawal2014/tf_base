import os
import tensorflow as tf
from graph.tf_sonnet import reuse_variables
from functools import wraps

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Graph(object):
    def __init__(self):
        self._graph = None
        self._session = None
        self._default_graph = None
        self._default_session = None

    def _load_graph(self, session_or_graph=None):
        if self._graph is None:
            if session_or_graph is None:
                if hasattr(tf.get_default_graph(), 'as_default'):
                    self._graph = tf.get_default_graph()
                else:
                    self._graph = tf.Graph()
            else:
                if isinstance(session_or_graph, tf.Graph):
                    self._graph = session_or_graph
                elif isinstance(session_or_graph, tf.Session):
                    self._graph = session_or_graph.graph
                else:
                    raise AssertionError('session_or_graph must be of type either {Graph} or {Session} give {Given}'
                                         .format(Graph=type(tf.Graph).__name__, Session=type(tf.Session).__name__,
                                                 Given=type(session_or_graph)))
        return self._graph

    def _load_session(self, session_or_graph=None):
        if self._session is None:
            if session_or_graph is None:
                if hasattr(tf.get_default_session(), 'as_default'):
                    self._session = tf.get_default_session()
                else:
                    self._session = tf.Session()
            else:
                if isinstance(session_or_graph, tf.Graph):
                    self._session = tf.Session(graph=session_or_graph)
                elif isinstance(session_or_graph, tf.Session):
                    self._session = session_or_graph
                else:
                    raise AssertionError('session_or_graph must be of type either {Graph} or {Session} give {Given}'
                                         .format(Graph=repr(tf.Graph).__name__, Session=repr(tf.Session).__name__,
                                                 Given=type(session_or_graph).__name__))

        return self._session

    def _load_default_graph(self, session_or_graph=None):
        if self._default_graph is None:
            if self._load_graph(session_or_graph=session_or_graph) != tf.get_default_graph():
                print('Resetting the old graph....')
                tf.reset_default_graph()
                with self._load_graph(session_or_graph=session_or_graph).as_default() as dg:
                    self._default_graph = dg
            else:
                self._default_graph = self._load_graph(session_or_graph=session_or_graph)
        return self._default_graph

    def _load_default_session(self, session_or_graph=None):
        if self._default_session is None:
            with self._load_session(session_or_graph=session_or_graph).as_default() as ds:
                self._default_session = ds

        return self._default_session


class With(Graph):
    def __init__(self):
        super(With, self).__init__()
        self._graph = None
        self._session = None
        self._default_graph = None
        self._default_session = None
        import functools
        self._wraps = functools.wraps

    def logger(self):
        import logging as lg

        def method(class_method):
            @self._wraps(class_method)
            def wrapper(*args, **kwargs):
                lg.basicConfig(filename='{}.log'.format(class_method.__qualname__), level=lg.INFO)
                lg.info('Ran {} with args:{} and kwargs:{} '.format(class_method.__qualname__, args, kwargs))
                result = class_method(*args, **kwargs)
                return result

            return wrapper

        return method

    def reuse_variables(self):
        def method(class_method):
            @self._wraps(class_method)
            def wrapper(*args, **kwargs):
                _result = reuse_variables(class_method)(*args, **kwargs)
                return _result

            return wrapper

        return method

    def trial(self):
        def method(class_method):
            @self._wraps(class_method)
            def wrapper(*args, **kwargs):
                print('Entering Trial Context')
                _result = class_method(*args, **kwargs)
                print('Exiting Trial Context')
                return _result

            return wrapper

        return method

    def with_graph(self, session_or_graph=None):
        with super(With, self)._load_graph(session_or_graph=session_or_graph) as graph:
            if self._graph is None:
                self._graph = graph

        def method(class_method):
            @self._wraps(class_method)
            def wrapper(*args, **kwargs):
                with self._graph:
                    _result = class_method(*args, **kwargs)
                return _result

            return wrapper

        return method

    def with_session(self, session_or_graph=None):
        with super(With, self)._load_session(session_or_graph=session_or_graph) as session:
            if self._session is None:
                self._session = session

        def method(class_method):
            @self._wraps(class_method)
            def wrapper(*args, **kwargs):
                with self._session:
                    _result = class_method(*args, **kwargs)
                return _result

            return wrapper

        return method

    def with_default_graph(self, session_or_graph=None):
        with super(With, self)._load_graph(session_or_graph=session_or_graph) as default_graph:
            if self._default_graph is None:
                self._default_graph = default_graph

        def method(class_method):
            @self._wraps(class_method)
            def wrapper(*args, **kwargs):
                with self._default_graph.as_default():
                    _result = class_method(*args, **kwargs)
                return _result

            return wrapper

        return method

    def with_default_session(self, session_or_graph=None):
        with super(With, self)._load_session(session_or_graph=session_or_graph) as default_session:
            if self._default_session is None:
                self._default_session = default_session

        def method(class_method):
            @self._wraps(class_method)
            def wrapper(*args, **kwargs):
                with default_session.as_default():
                    _result = class_method(*args, **kwargs)
                return _result

            return wrapper

        return method

    def with_default_graph_and_default_session(self, session_or_graph=None):
        graph = self._load_graph(session_or_graph=session_or_graph)
        sess = self._load_session(session_or_graph=graph)
        # print('isinstance(graph, tf.Graph) :', isinstance(graph, tf.Graph))
        # print('type(graph):', graph)
        with graph.as_default() as default_graph:
            with sess.as_default() as default_session:
                if self._default_graph is None:
                    self._default_graph = default_graph
                if self._default_session is None:
                    self._default_session = default_session

        def method(class_method):
            @self._wraps(class_method)
            def wrapper(*args, **kwargs):
                with self._default_graph.as_default():
                    with self._default_session.as_default():
                        _result = class_method(*args, **kwargs)
                return _result

            return wrapper

        return method

    def _g(self):
        return self._default_graph or self._graph

    def _s(self):
        return self._default_session or self._session


def make_context_meta_class(decorators, graph=None, session=None):
    class MakeContext(type):

        def __new__(mcs, name, bases, namespace):
            use_variable_scope = None
            if not isinstance(decorators, (list, tuple)):
                _decorators = [decorators]
            else:
                _decorators = decorators

            for decorator in _decorators:
                decorator_name = decorator.__qualname__.split('.')[1]
                assert decorator_name in With.__dict__
                if decorator_name == 'reuse_variables':
                    use_variable_scope = True
                del decorator_name
            for names in namespace:

                if names not in ['graph', 'session', 'tf', '__qualname__', '__classcell__'] or \
                        not (names.startswith(('__', '_')) or names.endswith(('__', '_'))):
                    for decorator in _decorators[::-1]:
                        if not isinstance(names, property):
                            namespace[names] = decorator(namespace[names])
                        else:
                            namespace[names] = property(decorator(namespace[names].__get__),
                                                        namespace[names].__set__,
                                                        namespace[names].__delattr__)
            # namespace['__qualname__'] = mcs.__qualname__ # __qualname__ is a property

            cls = type.__new__(mcs, name, bases, namespace)
            if use_variable_scope:
                with tf.variable_scope(name_or_scope=cls.__name__) as vs:
                    cls.variable_scope = vs
                    del use_variable_scope
            if graph is not None:
                setattr(cls, '_graph', graph)
            else:
                setattr(cls, '_graph', None)

            if session is not None:
                setattr(cls, '_session', session)
            else:
                setattr(cls, '_session', None)
            cls.graph = getattr(cls, '_graph')
            cls.session = getattr(cls, '_session')
            return cls

        def __init__(cls, name, bases, namespace):
            super(MakeContext, cls).__init__(name, bases, namespace)

    return MakeContext


class ContextClass(With):
    def __init__(self):
        super(ContextClass, self).__init__()
        self._graph = super(ContextClass, self)._g()
        self._session = super(ContextClass, self)._s()
        self._context = None

    @property
    def context(self):
        return make_context_meta_class(
            self._context, graph=self._graph, session=self._session)

    @context.setter
    def context(self, value):
        if self._context is None:
            self._context = list()
        assert isinstance(value, (list, tuple))
        for dec in value:
            if any([dec]):
                assert dec.__qualname__.split('.')[1] in With.__dict__
                self._context.append(dec)
        # print(self._context)

    def with_graph(self, session_or_graph=None):
        return super(ContextClass, self).with_graph(session_or_graph=session_or_graph)

    def with_session(self, session_or_graph=None):
        return super(ContextClass, self).with_session(session_or_graph=session_or_graph)

    def with_default_graph(self, session_or_graph=None):
        return super(ContextClass, self).with_default_graph(session_or_graph=session_or_graph)

    def with_default_session(self, session_or_graph=None):
        return super(ContextClass, self).with_default_session(session_or_graph=session_or_graph)

    def with_default_graph_and_default_session(self, session_or_graph=None):
        return super(ContextClass, self).with_default_graph_and_default_session(session_or_graph=session_or_graph)

    def __call__(self):
        return self.context


class GraphAPI(ContextClass):

    def __init__(self, session_or_graph=None, reuse_variables=None, log=None):
        super(GraphAPI, self).__init__()
        self._session_or_graph = session_or_graph
        self._reuse_variables = reuse_variables
        self._log = log

    def __call__(self):
        if self._context is None:
            graph_session = self.with_default_graph_and_default_session(session_or_graph=self._session_or_graph)
            reuse = self.reuse_variables() if any([self._reuse_variables]) else self._reuse_variables
            logger = self.logger() if any([self._log]) else self._log
            self.context = (graph_session, reuse, logger)
        return self.context


def do_not_reuse_variables(cls_method):

    @wraps(cls_method)
    def result(self, *args, **kwargs):
        return cls_method(self, *args, **kwargs)
    result.do_not_reuse_variables = True
    return result
