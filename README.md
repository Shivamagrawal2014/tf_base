tf_base
===
[![Build Status](https://travis-ci.com/Shivamagrawal2014/tf_base.svg?branch=master)](https://travis-ci.com/Shivamagrawal2014/tf_base)
[![Coverage Status](https://coveralls.io/repos/github/Shivamagrawal2014/tf_base/badge.svg?branch=master)](https://coveralls.io/github/Shivamagrawal2014/tf_base?branch=master)

>High level API to make task of Tensorflow API coding, simplified.

[Source Code](http://github.com/shivamagrawal2014/tf_base/)

## Requirements
- [Tensorflow](http://www.tensorflow.org)

## Basic Usage
To make creation of Datasets and Graph a bit easier.

Library under development. Contains rough edges/unfinished functonality. API subject to changes.

### Dictionary to Featues 
  >This library has support for making features with a single call. 
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
### Writing Images from Folder to TFRecord
  > With simple call tfrecord file for the images can be created, image folders will be taken as labels. Compression
  formats can be specified as boolen or types. Resizing of images also supported with **size** parameter.
  
  ```python 
    from tf_base.file.image import ImageTFRecordWriter
    
    images = ImageTFRecordWriter('/home/shivam/Documents/', ['jpg'],
                                 size=(20, 20, 0), show=False)
    record = images.to_tfr(tfrecord_name='images',
                           save_folder='/home/shivam/Documents/', allow_compression=True)
  
  ```
 ### Reading Images from TFRecord
 
 > This API is based on tf.dataset API so it can simply read the TFrecord  
 ```python 
    reader = ImageTFRecordReader()
    tf_record_path = '/path/to/image_folder'
    data = reader.batch(tf_record_path=tf_record_path, batch_size=2, epochs_size=1)
    data = data.make_one_shot_iterator()
    sess = reader.session
    data = data.get_next()
    summarizer = reader.summary_writer('../summary', sess.graph)
    try:
        for _ in range(21):
            image, label = sess.run(data)
            print(image.shape, label)
        print('Completed!')
    except tf.errors.OutOfRangeError:
        print('Data Exhausted!')
    finally:
        summarizer.close()
 
 ```


### Adding with tf.Graph functionality with GraphAPI to class
> With GraphAPI classes and functions act as variable_scope to the graph. based on **tf.sonnet** backend 
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
            
    convolution = Convolution()
    weight = convolution.W(tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
    
    # is same as 
    with.variabl_scope('Convolution'+'/'+'W')
      one = tf.Variable(initial_value=
                        tf.truncated_normal_initializer(mean=0.0, stddev=1.0), 
                        name='weight'))

```


