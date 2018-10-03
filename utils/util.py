# Python 2-3 Compatibility
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from six.moves import range, zip

import os.path


class ReadOnlyDict(dict):
  """Provides a read-only version of a python dict."""

  def __readonly__(self, *args, **kwargs):
    raise RuntimeError("Cannot modify `ReadOnlyDict`")

  __setitem__ = __readonly__
  __delitem__ = __readonly__
  pop = __readonly__
  popitem = __readonly__
  clear = __readonly__
  update = __readonly__
  setdefault = __readonly__
  del __readonly__


def parser_path(parser, path):
  if not os.path.exists(path):
    parser.error("The file %s does not exist!" % path)
  else:
    return
