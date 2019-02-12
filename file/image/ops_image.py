import tensorflow as tf
from file.image.process_image import ConvolutionalBatchNormalizer
from graph.tf_graph_api import do_not_reuse_variables

__all__ = ['Variable', 'Weight', 'Bias', 'BiasAdd', 'ReLU', 'MaxPool',
           'Conv2D', 'FullyConnect', 'FinalFullyConnect', 'BatchNormalize', 'LayerStack']


class Variable(object):
    def __init__(self, shape):
        self._shape = shape

    def weight(self,
               name,
               stddev=0.02,
               seed=1
               ):
        return tf.get_variable(
            name=name,
            shape=self._shape,
            initializer=tf.truncated_normal_initializer(stddev=stddev, seed=seed))

    def bias(self,
             name=None,
             constant_init_value=0):
        return tf.get_variable(name=name,
                               shape=self._shape[-1],
                               initializer=tf.constant_initializer(constant_init_value)
                               )


class Weight(object):

    def __init__(self,
                 shape,
                 weight_name=None,
                 stddev=0.02,
                 seed=1
                 ):
        if weight_name is None:
            self._name = 'W'
        else:
            self._name = weight_name
        self._stddev = stddev
        self._seed = seed
        self._var = Variable(shape=shape)

    @property
    def __call__(self):
        w = self._var.weight(name=self._name, stddev=self._stddev, seed=self._seed)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, w)
        return w

    def __str__(self):
        return 'Weight'


class Bias(object):
    def __init__(self,
                 shape,
                 bias_name=None,
                 constant_init_value=0):
        if bias_name is None:
            self._name = 'B'
        else:
            self._name = bias_name

        self._init_val = constant_init_value
        self._var = Variable(shape=shape)

    @property
    def __call__(self):
        b = self._var.bias(name=self._name, constant_init_value=self._init_val)
        tf.add_to_collection(tf.GraphKeys.BIASES, b)
        return b

    def __str__(self):
        return 'Bias'


class BiasAdd(object):
    def __init__(self,
                 bias,
                 name='bias_add'
                 ):
        self._name = name
        self._bias = bias
        self._bias_add = tf.nn.bias_add

    def __call__(self, images):
        with tf.name_scope(self._name):
            bias_add = self._bias_add(images, self._bias)
        tf.add_to_collection('BiasAdd', bias_add)
        return bias_add

    def __str__(self):
        return 'BiasAdd'


class ReLU(object):

    def __init__(self, relu_fuction=None, scope_name='ReLU'):
        """
        :param relu_fuction: (optional) Function for the ReLU supplied
        :param scope_name: (optional) Name of the Variable Scope
        """

        self._scope_name = scope_name
        self._relu = relu_fuction or tf.nn.relu

    def __call__(self, image, *args, **kwargs):
        with tf.name_scope(self._scope_name):
            image = self._relu(image, *args, **kwargs)
        tf.add_to_collections([tf.GraphKeys.ACTIVATIONS, 'ReLU'], image)
        return image

    def __str__(self):
        return 'ReLU'

    def __repr__(self):
        return '<{name}>'.format(name=self._scope_name)

    @property
    def scope(self):
        return self._scope_name


class MaxPool(object):

    def __init__(self, ksize=2, strides=2, padding='SAME', scope_name='MaxPooling'):
        """
        :param ksize:
        :param strides:
        :param padding:
        :param scope_name:
        """
        self._scope_name = scope_name
        self._strides = [1, strides, strides, 1]
        self._ksize = [1, ksize, ksize, 1]
        self._padding = padding
        self._max_pool = tf.nn.max_pool

    def __call__(self, image):

        with tf.name_scope(self._scope_name):
            image = self._max_pool(image, ksize=self._ksize,
                                   strides=self._strides,
                                   padding=self._padding)
        tf.add_to_collections([tf.GraphKeys.ACTIVATIONS, 'MaxPool'], image)
        return image

    def __str__(self):
        return 'MaxPool'

    def __repr__(self):
        return '<{name}>'.format(name=self._scope_name)

    @property
    def scope(self):
        return self._scope_name


class BatchNormalize(object):

    def __init__(self,
                 name,
                 filter_channels):
        self._bn_scope_name = None
        self._normalize = None

        with tf.variable_scope('normalize_'+name) as bn_scope:
            if self._normalize is None:
                ewma = tf.train.ExponentialMovingAverage(decay=0.99)
                bn = ConvolutionalBatchNormalizer(filter_channels, 0.001, ewma, True)
                update_assignments = bn.get_assigner()
                self._normalize = bn.normalize
            if self._bn_scope_name is None:
                self._bn_scope_name = bn_scope.name if hasattr(bn_scope, 'name') else bn_scope

    def __call__(self, images):

        with tf.name_scope(self._bn_scope_name):
            images = self._normalize(images, train=False)
        tf.add_to_collections([tf.GraphKeys.ACTIVATIONS, 'BatchNormalize'], images)
        return images

    def __str__(self):
        return 'BatchNormalize'

    def __repr__(self):
        return '<{batch_norm}>'.format(batch_norm=self._bn_scope_name)

    @property
    def scope(self):
        return self._bn_scope_name


class Conv2D(object):

    def __init__(self,
                 name,
                 filters_shape,
                 weight_name=None,
                 bias_name=None,
                 weight=None,
                 bias=None,
                 strides=1,
                 padding='SAME'):
        """
        :param x: The Tensor of image batch on which the convolution filter is traversed;
        :param W: The Filter used for the convolution in the layer;
        :param b: The Bias term;
        :param strides: Number of steps for filter to stride;
        :return conv2d->relu:
        """
        self._name = name
        self._filter = filters_shape
        self._weight_name = weight_name
        self._bias_name = bias_name
        self._weight = weight
        self._bias = bias
        self._strides = strides
        self._padding = padding
        self._conv2d = tf.nn.conv2d
        self._bias_add = tf.nn.bias_add
        self._conv_scope = None

        with tf.variable_scope(name_or_scope=self._name) as conv_scope:
            if self._conv_scope is None:
                self._conv_scope = conv_scope.name

            if self._weight is None:
                self._W = Weight(shape=self._filter, weight_name=self._weight_name).__call__
            else:
                self._W = self._weight.__call__

            if self._bias is None:
                self._B = Bias(shape=self._filter, bias_name=self._bias_name).__call__
            else:
                self._B = self._bias.__call__
        tf.summary.histogram(self._name+'/weight', self._W)
        tf.summary.histogram(self._name+'/bias', self._B)

    def __call__(self, images):
        with tf.variable_scope(name_or_scope=self._conv_scope + '/'):
            with tf.variable_scope('conv2d'):
                images = self._conv2d(images, self._W,
                                      strides=[1, self._strides, self._strides, 1],
                                      padding=self._padding)
            tf.summary.histogram(self._name + '/conv2d', images)
            tf.add_to_collections([tf.GraphKeys.ACTIVATIONS, 'Conv2D'], images)
            images = BiasAdd(self._B)(images)
            tf.summary.histogram(self._name+'/bias_add', images)
        # print(self._conv_scope)
        return images

    @property
    def scope(self):
        return self._conv_scope

    def __str__(self):
        return 'Conv2D'

    def __repr__(self):
        return "<{name}, {conv}>".format(name=self._conv_scope,
                                         conv=self._conv2d.__qualname__)


class FullyConnect(object):

    def __init__(self,
                 name,
                 filter_shape,
                 weight_name=None,
                 bias_name=None,
                 weight=None,
                 bias=None,
                 ):
        self._name = name
        self._filter = filter_shape
        self._weight_name = weight_name
        self._bias_name = bias_name
        self._weight = weight
        self._bias = bias
        self._fully_connect_scope = None

        with tf.variable_scope(name_or_scope=self._name) as fully_connect_scope:
            if self._fully_connect_scope is None:
                self._fully_connect_scope = fully_connect_scope.name

            if self._weight is None:
                self._W = Weight(shape=self._filter, weight_name=self._weight_name).__call__
            else:
                self._W = self._weight.__call__

            if self._bias is None:
                self._B = Bias(shape=self._filter, bias_name=self._bias_name).__call__
            else:
                self._B = self._bias.__call__
        tf.summary.histogram(self._name + '/weight', self._W)
        tf.summary.histogram(self._name + '/bias', self._B)

    def __call__(self, images):
        with tf.variable_scope(name_or_scope=self._fully_connect_scope + '/'):
            reshape_filter = [-1, self._W.shape[0]]

            with tf.variable_scope(name_or_scope='reshape_convoluted_images'):
                images = tf.reshape(images, reshape_filter)
                tf.summary.histogram(self._name+'/reshape', images)
            with tf.variable_scope(name_or_scope='fullyconnect'):
                with tf.variable_scope(name_or_scope='mat_multiply'):
                    images = tf.matmul(images, self._W)
                tf.summary.histogram(self._name+'/FullyConnect', images)
                tf.add_to_collections([tf.GraphKeys.ACTIVATIONS, 'FullyConnect'], images)
                images = BiasAdd(self._B)(images)
                tf.summary.histogram(self._name + '/bias_add', images)
        # print(self._fully_connect_scope)
        return images

    @property
    def scope(self):
        return self._fully_connect_scope

    def __str__(self):
        return 'FullyConnect'

    def __repr__(self):
        return "<{name}, {fc}>".format(name=self._fully_connect_scope,
                                       fc='fully_connect')


class FinalFullyConnect(object):

    def __init__(self,
                 name,
                 filter_shape,
                 weight_name=None,
                 bias_name=None,
                 weight=None,
                 bias=None
                 ):
        self._name = name
        self._filter = filter_shape
        self._weight_name = weight_name
        self._bias_name = bias_name
        self._weight = weight
        self._bias = bias
        self._final_fully_connect_scope = None

        with tf.variable_scope(name_or_scope=self._name) as final_fully_connect_scope:
            if self._final_fully_connect_scope is None:
                self._final_fully_connect_scope = final_fully_connect_scope.name

            if self._weight is None:
                self._W = Weight(shape=self._filter, weight_name=self._weight_name).__call__
            else:
                self._W = self._weight.__call__

            if self._bias is None:
                self._B = Bias(shape=self._filter, bias_name=self._bias_name).__call__
            else:
                self._B = self._bias.__call__
        tf.summary.histogram(self._name + '/weight', self._W)
        tf.summary.histogram(self._name + '/bias', self._B)

    def __call__(self, images):
        with tf.variable_scope(name_or_scope=self._final_fully_connect_scope + '/'):
            with tf.variable_scope(name_or_scope='finalfullyconnect'):
                with tf.variable_scope(name_or_scope='mat_mul'):
                    images = tf.matmul(images, self._W)
                    tf.summary.histogram(self._name+'/finalfullyconnect', images)
                    tf.add_to_collections([tf.GraphKeys.ACTIVATIONS, 'FinalFullyConnect'], images)
                images = BiasAdd(self._B)(images)
                tf.summary.histogram(self._name + '/bias_add', images)
        # print(self._final_fully_connect_scope)
        return images

    @property
    def scope(self):
        return self._final_fully_connect_scope

    def __str__(self):
        return 'FinalFullyConnect'

    def __repr__(self):
        return '<{name}, {full_final}>'.format(name=self._final_fully_connect_scope,
                                               full_final='final_fully_connect')


class LayerName(object):

    def __init__(self,
                 layer_name_prefix='Layer',
                 limit=None):
        self._counter = 0
        self._limit = limit
        self._name_prefix = layer_name_prefix
        self._names = list()
        for i in range(self._limit):
            self._names.append(self())

    def __call__(self):

        if self._counter == 0:
            name = 'Input'
        else:
            if self._limit is None:
                name = 'Hidden' + str(self._counter)
            else:
                if self._counter < self._limit - 2:
                    name = 'Hidden' + str(self._counter)
                elif self._counter == self._limit - 2:
                    name = 'FullyConnect'
                else:
                    assert self._counter == self._limit - 1
                    name = 'Output'

        layer_name = '_'.join([self._name_prefix, name])
        self._counter += 1
        return layer_name

    @property
    def name(self):
        return self._names


class ConvolutionCell(object):

    def __init__(self, conv_fn=None):
        self._counter = 0
        self._conv_fn = conv_fn or Conv2D
        self._cell_scope = None

    def __call__(self, name,
                 filters_shape: list,
                 layer_num: int=None,
                 batch_norm_at_layer: int = None):

        self._counter += 1
        layer_num = layer_num or self._counter

        stack = list()
        conv = self._conv_fn(name=name, filters_shape=filters_shape)

        # this scope is
        self._cell_scope = conv.scope

        stack.append(conv)
        with tf.name_scope(conv.scope):
            relu = ReLU()
            stack.append(relu)
            max_pool = MaxPool()
            stack.append(max_pool)
            if batch_norm_at_layer is not None and \
                    self._counter % batch_norm_at_layer == 0:
                batch_norm = BatchNormalize(name=name,
                                            filter_channels=filters_shape[-1])
                stack.append(batch_norm)

        return stack

    @property
    def cell_scope(self):
        return self._cell_scope


class FullyConnectCell(object):

    def __init__(self, fully_conn_fn=None):
        self._counter = 0
        self._fully_conn_fn = fully_conn_fn or FullyConnect

    def __call__(self, name, filters_shape):
        stack = list()
        fully_conn = self._fully_conn_fn(name=name,
                                         filter_shape=filters_shape)
        stack.append(fully_conn)
        with tf.variable_scope(fully_conn.scope):
            relu = ReLU()
            stack.append(relu)
        return stack


class FinalFullyConnectCell(object):

    def __init__(self, final_fully_conn_fn=None):
        self._counter = 0
        self._final_fully_conn_fn = final_fully_conn_fn or FinalFullyConnect

    def __call__(self, name, filters_shape):
        stack = list()
        final_layer = self._final_fully_conn_fn(name, filters_shape)
        stack.append(final_layer)
        return stack


class LayerStack(object):

    def __init__(self, conv_fn=None, fully_conn_fn=None, final_fully_conn_fn=None):
        self._conv_fn = ConvolutionCell(conv_fn)
        self._fully_conn_fn = FullyConnectCell(fully_conn_fn)
        self._final_fully_conn_fn = FinalFullyConnectCell(final_fully_conn_fn)
        self._counter = 0

    def _convolution(self, name, filters_shape: list, layer_num: int=None, batch_norm_at_layer=None):
        return self._conv_fn(name=name,
                             filters_shape=filters_shape,
                             layer_num=layer_num,
                             batch_norm_at_layer=batch_norm_at_layer)

    def _fully_connect(self, name, filters_shape):
        return self._fully_conn_fn(name=name, filters_shape=filters_shape)

    def _final_fully_connect(self, name, filters_shape):
        return self._final_fully_conn_fn(name=name, filters_shape=filters_shape)

    @do_not_reuse_variables
    def image_classification(self, layer_name, filters_list, batch_norm_at_layer=None):
        layer = list()
        names_list = LayerName(layer_name_prefix=layer_name, limit=len(filters_list)).name
        names_filters_shapes = list(zip(names_list, filters_list))

        with tf.name_scope('convolutions'):
            for l_name, l_filter_shape in names_filters_shapes[:-2]:
                print(l_name, l_filter_shape)
                layer.extend(self._convolution(name=l_name,
                                               filters_shape=l_filter_shape,
                                               batch_norm_at_layer=batch_norm_at_layer))

        with tf.name_scope('fully_connect'):
            l_name, l_filter_shape = names_filters_shapes[-2:][0]
            print(l_name, l_filter_shape)
            layer.extend(self._fully_connect(name=l_name, filters_shape=l_filter_shape))

        with tf.name_scope('final_fully_connect'):
            l_name, l_filter_shape = names_filters_shapes[-2:][1]
            print(l_name, l_filter_shape)
            layer.extend(self._final_fully_connect(name=l_name, filters_shape=l_filter_shape))

        return layer

    def image_segmentation(self, ):
        pass


conv_1 = [5, 5, 3, 32]
conv_2 = [5, 5, 32, 64]
conv_3 = [7, 7, 64, 128]
conv_4 = [9, 9, 128, 256]
fully_conn = [16 * 16 * 256, 1024]
final_fc = [1024, 10]

FILTERS_SHAPE_LIST = [conv_1, conv_2, conv_3, conv_4, fully_conn, final_fc]

if __name__ == '__main__':
    from file.image.record_image import ImageTFRecordReader as ImageReader
    layer_stack = LayerStack()
    layer_stack = layer_stack.image_classification(net_name='TestImage', layer_name='Image',
                                                   filters_list=FILTERS_SHAPE_LIST,
                                                   batch_norm_at_layer=2)

    image_reader = ImageReader()
    reader = image_reader.batch(tf_path=r'C:\Users\shivam.agarwal\PycharmProjects\TopicAPI\data\image\image_record.tfr',
                                batch_size=5,
                                epochs_size=20)
    data = reader.make_one_shot_iterator()
    # init = data.initializer
    sess = image_reader.session
    # sess.run(init)
    data = data.get_next()
    summarizer = image_reader.summary_writer('../summary')
    _, i = data[0]['image_pixel']
    print(i.shape)
    for layer in layer_stack:
        i = layer(i)
        print(i)
    glob_init_op = tf.global_variables_initializer()
    loc_init_op = tf.local_variables_initializer()

    sess.run(loc_init_op)
    sess.run(glob_init_op)
    print(i.shape)
    with summarizer:
        try:
            for _ in range(10):
                i_ = sess.run(i)
                print(i_.shape)
            print('Completed!')
        except tf.errors.OutOfRangeError:
            print('Data Exhausted!')
