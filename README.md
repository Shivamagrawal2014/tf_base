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
