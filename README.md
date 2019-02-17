# tf_base
[![Build Status](https://travis-ci.com/Shivamagrawal2014/tf_base.svg?branch=master)](https://travis-ci.com/Shivamagrawal2014/tf_base)

High level API to make task of Tensorflow Coding Easy   

Tensorflow coding, simplified.

[Source Code](http://github.com/shivamagrawal2014/tf_base/)

Library under development. Contains rough edges/unfinished functonality. API subject to changes.

This library has support for making features from a single call. 
```python 
  from tf_base.file.record import protofy
  features = protofy(byte_dict={'A': [b'a', b'c', b'd', b'e', b'f'], 'B': [b'b']},
                     int_dict={'A': [1, 2, 3], 'B': [4, 5]},
                     float_dict={'C': [1.1, 2.1, 3.1], 'D': [4.1, 5.1]})
  # it returns 
  feature {
  key: "A"
  value {
    bytes_list {
      value: "a"
      value: "c"
      value: "d"
      value: "e"
      value: "f"
    }
  }
}
feature {
  key: "B"
  value {
    bytes_list {
      value: "b"
    }
  }
}
feature {
  key: "C"
  value {
    float_list {
      value: 1.1
      value: 2.1
      value: 3.1
    }
  }
}
feature {
  key: "D"
  value {
    float_list {
      value: 4.1
      value: 5.1
    }
  }
}
feature {
  key: "E"
  value {
    int64_list {
      value: 1
      value: 2
      value: 3
    }
  }
}
feature {
  key: "F"
  value {
    int64_list {
      value: 4
      value: 5
    }
  }
}
}
```
