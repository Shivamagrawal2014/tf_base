# tf_base
[![Build Status](https://travis-ci.com/Shivamagrawal2014/tf_base.svg?branch=master)](https://travis-ci.com/Shivamagrawal2014/tf_base)

High level API to make task of Tensorflow Coding Easy   

Tensorflow coding, simplified.

[Source Code](http://github.com/shivamagrawal2014/tf_base/)

Library under development. Contains rough edges/unfinished functonality. API subject to changes.

This library has support for making features from a single call. 
```python 
  from tf_base.file.record.protofy import protofy
  
  features = protofy(int_dict={'testing_int': [[1], [1, 3, 5]]}))
  print(features)
  # returns
feature_list {
  key: "testing_int"
  value {
    feature {
      int64_list {
        value: 1
      }
    }
    feature {
      int64_list {
        value: 1
        value: 3
        value: 5
      }
    }
  }
}
}
```
Also Graph functionality acts as base graph 
```python 
    from tf_base.graph import GraphAPI
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
            
            
```
Now all classes and functions act as variable_scope to the graph and there is no explicit need to call
```python 

with.variabl_scope('Meta'+'method')
    one = tf.Variable(initial_value=1, name='one'))
```
to name the variable as **Meta/meta_method_1/one:0**
