from collections import namedtuple as nt


class Descriptor(object):

    def __init__(self, datatype):
        self._value = None

        if self._value is None:
            if callable(datatype):
                self._value = datatype()
            else:
                self._value = datatype

    @property
    def value(self):
        return self._value

    class Put(nt('put', 'value')):
            pass

    class New(object):
        def __init__(self):
            self._value = None

        def new_set(self, value):
            if isinstance(value, Descriptor.Put):
                self._value = value
            else:
                print(type(value))
                raise ValueError('Unsupported type')

        def new_get(self):
            return self._value.value

        def new_del(self):
            del self

        def __str__(self):
            return "{value}".format(value=str(self._value))

        newly = property(new_get, new_set, None, 'This is New Property')


class Data(object):

    def __init__(self, datatype, allow_sparse=True, edit=False):

        self._value = Descriptor(datatype).value
        self._edit = edit
        self._sparse = allow_sparse

    def __getitem__(self, token):
        print('Getting Token [{token}]...'.format(token=token))
        if isinstance(self._value.__getitem__(token), Descriptor.Put):
            return self._value.__getitem__(token).value
        else:
            return self._value.__getitem__(token)

    def __setitem__(self, token, value):
        try:
            _ = self._value[token]
            assert isinstance(value, Descriptor.Put), 'Unsupported DataType'
            if isinstance(self._value.__getitem__(token), Descriptor.New):
                assert self._edit is True, 'Currently Editing Option is in False mode'
                print('Editing Token [{token}] to Value [{value}]'.format(token=token, value=value.value))
                self._value.__getitem__(token).newly = value
            else:
                print('Editing Token [{token}] to Value [{value}]'.format(token=token, value=value.value))
                self._value.__setitem__(token, value)
        except (KeyError, IndexError):
            assert isinstance(value, Descriptor.New), "Unsupported Datatype"
            print('Setting Token [{token}] to Value [{value}]...'.format(token=token, value=value.newly))
            try:
                self._value.__setitem__(token, value)
            except IndexError as ie:
                if self._sparse:
                    assert isinstance(token, int)
                    index = token-(len(self._value)-1)
                    if index > 1:
                        self._value.extend([None]*(index-1))

                    self._value.append(value)
                else:
                    try:
                        assert len(self._value) == token
                        self._value.append(value)
                    except Exception:
                        raise ie

    def __delitem__(self, token):
        if token in self._value and isinstance(self._value.__getitem__(token), Descriptor.New):
            value = self._value.__getitem__(token).newly
            print('Deleting Token [{token}] and Value [{value}]'.format(token=token, value=value))
            del self._value.__delitem__(token).newly
        else:
            self._value.__delitem__(token)

    @property
    def value(self):
        if hasattr(self._value, 'keys'):
            for k, v in self._value.items():
                if isinstance(v, Descriptor.New):
                    self._value[k] = v.newly if not isinstance(v.newly, Descriptor.Put) else v.newly.value
                elif isinstance(v, Descriptor.Put):
                    self._value[k] = v.value
                else:
                    self._value[k] = v

            return self._value
        else:
            if hasattr(self._value, 'index'):
                for idx, v in enumerate(self._value):
                    if isinstance(self._value[idx], Descriptor.New):
                        self._value[idx] = v.newly if not isinstance(v.newly, Descriptor.Put) else v.newly.value
                    elif isinstance(v, Descriptor.Put):
                        self._value[idx] = v.value
                    else:
                        self._value[idx] = v

            return self._value


def descriptor(dtype, allow_sparse=True, edit=False):
    class Desc(object):
        def __init__(self, datatype, allow_sparse, edit):
            self._val = Data(datatype, allow_sparse, edit)
            self._edit = edit

        def new_val(self, value):
            val = Descriptor.New()
            val.newly = self._put(value)
            return val

        def edit_val(self, value):
            return self._put(value)

        def __getitem__(self, token):
            return self._val.__getitem__(token)

        def __setitem__(self, token, value):
            self._val.__setitem__(token, value)

        def __delitem__(self, token):
            self._val.__delitem__(token)

        @staticmethod
        def _put(value):
            return Descriptor.Put(value)

        @property
        def edit(self):
            return self._edit

        @property
        def value(self):
            return self._val.value

        def __str__(self):
            return str(self.value)
    return Desc(dtype, allow_sparse, edit)


if __name__ == '__main__':
    value = descriptor(list, allow_sparse=True, edit=True)
    value[0] = value.new_val(2)
    value[1] = value.new_val(300)
    value[1] = value.edit_val(0.000)
    print(value)

