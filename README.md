# tf_base
[![Build Status](https://travis-ci.com/Shivamagrawal2014/tf_base.svg?branch=master)](https://travis-ci.com/Shivamagrawal2014/tf_base)

High level API to make task of Tensorflow Coding Easy   

Tensorflow coding, simplified.

[Source Code](http://github.com/shivamagrawal2014/tf_base/)

Library under development. Contains rough edges/unfinished functonality. API subject to changes.

### Dictionary to Featues 
  >This library has support for making features from a single call. 
```python 
  from tf_base.file.record.protofy import protofy
  
  >>features = protofy(int_dict={'testing_int': [[1], [1, 3, 5]]}))
  >>print(features)
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
### Image Folder to TFRecord
  > With a single call tfrecord file for the images can be created, image folders will be taken as labels. Compression
  formats can be specified as boolen or types. 
  
  ```python 
    from tf_base.file.image import ImageTFRecordWriter
    
    images = ImageTFRecordWriter('/home/shivam/Documents/', ['jpg'],
                                 size=(20, 20, 0), show=False)
    record = images.to_tfr(tfrecord_name='images',
                           save_folder='/home/shivam/Documents/', allow_compression=True)
  
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

    class Convolution(Api):

        def W(self, value, name='weight'):
            return tf.Variable(initial_value=value, name=name)

        def B(self, value, name='bias'):
            return tf.Variable(initial_value=value, name=name)
            
            
```


Now all classes and functions act as variable_scope to the graph.
So there is no explicit need to call
```python 

with.variabl_scope('Concolution'+'/'+'W')
    one = tf.Variable(initial_value=
                      tf.truncated_normal_initializer(mean=0.0, stddev=1.0), 
                      name='weight'))
```
to name the variable as **Convolution/W/weight:0**

just 
```python 
  convolution = Convolution()
  weight = convolution.W(tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
  
  will do the task
  
