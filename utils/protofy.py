from utils.proto_funcs import *
from utils.proto_funcs import _List
from utils.container import descriptor

data_type = dict


class DataList(object):
    def __init__(self, merge_all=None):
        if merge_all is None:
            self._merge_all = True
        else:
            self._merge_all = merge_all
        self._int_list = descriptor(data_type)
        self._float_list = descriptor(data_type)
        self._bytes_list = descriptor(data_type)
        self._list_func = _List

    @property
    def int_list(self):
        return data_type(self._int_list.value)

    @int_list.setter
    def int_list(self, value):
        int_list_key = 'int_list'
        if isinstance_int(value):
            self._int_list[int_list_key] = self._int_list.new_val(
                value=self._list_func.int64_list(value=value))

        if isinstance_int64_list(value):
            self._int_list[int_list_key] = self._int_list.new_val(value=value)

        if isinstance_list(value):
            assert each_tuple_or_list_elem_type_a_b(value, int, list)
            if each_list_elem(isinstance_int, value):
                if self._merge_all:
                    self._int_list[int_list_key] = self._int_list.new_val(
                        value=self._list_func.int64_list(value))
                else:
                    self._int_list[int_list_key] = self._int_list.new_val(
                        value=apply_to_list_elem(self._list_func.int64_list, value))
            else:
                assert apply_to_list_elem(lambda x: each_list_elem(isinstance_int, x), value)
                if self._merge_all:
                    self._int_list[int_list_key] = self._int_list.new_val(
                        value=self._list_func.int64_list(value))
                else:
                    self._int_list[int_list_key] = self._int_list.new_val(
                        value=[self._list_func.int64_list(i) for i in value])

        if isinstance_tuple(value):
            assert each_tuple_elem(isinstance_int, value)
            if self._merge_all:
                self._int_list[int_list_key] = self._int_list.new_val(
                    value=self._list_func.int64_list([value]))
            else:
                self._int_list[int_list_key] = self._int_list.new_val(
                    value=apply_to_list_elem(self._list_func.int64_list, value))

        if isinstance_dict(value):
            assert each_dict_elem_type_a_b(value, int, list)
            if self._merge_all:
                if each_dict_elem(isinstance_int, value):
                    self._int_list[int_list_key] = self._int_list.new_val(
                        value=apply_to_dict_elem(self._list_func.int64_list, value))
                else:
                    self.int_list[int_list_key] = self._int_list.new_val(
                        value=apply_to_dict_elem(self._list_func.int64_list, value))
            else:
                for name, val in value.items():
                    self._int_list[name] = self._int_list.new_val(
                        value=self._list_func.int64_list(value=val))


if __name__ == '__main__':
    data = DataList(merge_all=True)
    data.int_list = [[1, 3, 5], [7, 8, 9]]
    print(data.int_list['int_list'])
    print(type(data.int_list['int_list']))

