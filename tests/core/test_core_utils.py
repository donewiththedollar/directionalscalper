import unittest

from directionalscalper.core.utils import BlankResponse


class TestCoreUtils(unittest.TestCase):
    def test_BlankResponse(self):
        blank = BlankResponse()
        assert blank.content == ""
